#!/usr/bin/env python3
"""Refine a shared two-court court-to-world Sim(3), adjacent-court gap, and train-camera pose offsets.

This tool aligns a two-court tennis-court ground-line geometry to MASt3R train
cameras using distance-transform supervision from line masks.

The two courts are constrained to:
- lie on the same plane
- stay parallel
- differ only by a single unknown adjacent-court gap in court coordinates

The first court is the base court at the original court coordinate frame.
The second court is a translated copy of the same line model.

Optimization schedule:
1. Fix all camera pose offsets to zero, optimize the shared Sim(3) and court gap.
2. Alternate:
   - optimize per-camera SE(3) pose offsets with Sim(3)+gap fixed
   - optimize Sim(3)+gap with pose offsets fixed

Inputs are read from a scene directory:
- images_train.txt
- mast3r/camera_intrinsics.npy
- mast3r/camera_poses.npy
- mast3r/court_line_masks.npy

Outputs are written under the requested output directory:
- sim3_refined.npz
- camera_pose_offsets.npy
- camera_poses_refined.npy
- world_to_camera_refined.npy
- metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.court_scheme import COURT_SKELETON, DOUBLES_WIDTH, court_keypoints_3d  # noqa: E402


@dataclass
class Config:
    scene_dir: Path
    output_dir: Path
    init_sim3_path: Path | None = None
    mask_path: Path | None = None
    mast3r_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "float32"
    max_image_size: int = 1024
    camera_batch_size: int = 8
    line_sample_step: float = 0.20
    dt_trunc: float = 24.0
    dt_sigma: float = 4.0
    charbonnier_eps: float = 1.0e-3
    z_min: float = 1.0e-4
    sim_lr: float = 5.0e-3
    pose_lr: float = 2.0e-3
    sim_phase_iters: int = 400
    local_phase_iters: int = 120
    global_phase_iters: int = 160
    num_alternations: int = 5
    lambda_pose: float = 1.0
    lambda_plane: float = 2.0
    lambda_init: float = 0.5
    lambda_visibility: float = 4.0
    pose_rot_sigma_deg: float = 3.0
    pose_trans_sigma: float | None = None
    plane_rot_sigma_deg: float = 5.0
    plane_height_sigma: float = 0.25
    init_rot_sigma_deg: float = 12.0
    init_trans_sigma: float = 3.0
    init_scale_sigma_log: float = 0.20
    init_gap_sigma: float = 1.5
    min_visible_fraction: float = 0.08
    adjacent_court_direction: str = "+x"
    init_adjacent_gap: float = 3.0
    court_base_width: float = 512.0
    log_interval: int = 20


@dataclass
class InitialGlobalPrior:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    adjacent_gap: float
    plane_normal: np.ndarray | None
    plane_d: float | None
    enabled: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("/root/repos/3rgs/data/tennis_court"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <scene-dir>/mast3r/court_alignment.",
    )
    parser.add_argument(
        "--init-sim3-path",
        type=Path,
        default="data/tennis_court/mast3r/court_alignment/init_sim3.json",
        help="Optional .npy/.npz/.json initial court->world similarity transform.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Override mask array path. Defaults to <mast3r-dir>/court_line_masks.npy.",
    )
    parser.add_argument(
        "--mast3r-dir",
        type=Path,
        default=None,
        help="Override mast3r directory. Defaults to <scene-dir>/mast3r.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1024,
        help="Resize masks so the longer side is at most this value. Use <=0 to keep original size.",
    )
    parser.add_argument("--camera-batch-size", type=int, default=8)
    parser.add_argument("--line-sample-step", type=float, default=0.20)
    parser.add_argument("--dt-trunc", type=float, default=24.0)
    parser.add_argument("--dt-sigma", type=float, default=4.0)
    parser.add_argument("--charbonnier-eps", type=float, default=1.0e-3)
    parser.add_argument("--z-min", type=float, default=1.0e-4)
    parser.add_argument("--sim-lr", type=float, default=5.0e-3)
    parser.add_argument("--pose-lr", type=float, default=2.0e-3)
    parser.add_argument("--sim-phase-iters", type=int, default=400)
    parser.add_argument("--local-phase-iters", type=int, default=120)
    parser.add_argument("--global-phase-iters", type=int, default=160)
    parser.add_argument("--num-alternations", type=int, default=5)
    parser.add_argument("--lambda-pose", type=float, default=1.0)
    parser.add_argument("--lambda-plane", type=float, default=2.0)
    parser.add_argument("--lambda-init", type=float, default=0.5)
    parser.add_argument("--lambda-visibility", type=float, default=4.0)
    parser.add_argument("--pose-rot-sigma-deg", type=float, default=3.0)
    parser.add_argument(
        "--pose-trans-sigma",
        type=float,
        default=None,
        help="World-space translation sigma for pose regularization. "
            "Defaults to 2%% of train camera RMS spread.",
    )
    parser.add_argument("--plane-rot-sigma-deg", type=float, default=5.0)
    parser.add_argument("--plane-height-sigma", type=float, default=0.25)
    parser.add_argument("--init-rot-sigma-deg", type=float, default=12.0)
    parser.add_argument("--init-trans-sigma", type=float, default=3.0)
    parser.add_argument("--init-scale-sigma-log", type=float, default=0.20)
    parser.add_argument("--init-gap-sigma", type=float, default=1.5)
    parser.add_argument("--min-visible-fraction", type=float, default=0.08)
    parser.add_argument(
        "--adjacent-court-direction",
        choices=("+x", "-x", "+y", "-y"),
        default="+x",
        help="Direction of the second court center in the base court coordinate frame.",
    )
    parser.add_argument(
        "--init-adjacent-gap",
        type=float,
        default=3.0,
        help="Initial edge-to-edge gap in meters between the two neighboring doubles sidelines.",
    )
    parser.add_argument(
        "--court-base-width",
        type=float,
        default=512.0,
        help="MASt3R camera intrinsics are assumed to be expressed at this image width.",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    mast3r_dir = args.mast3r_dir.expanduser().resolve() if args.mast3r_dir is not None else scene_dir / "mast3r"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else mast3r_dir / "court_alignment"
    )
    return Config(
        scene_dir=scene_dir,
        output_dir=output_dir,
        init_sim3_path=args.init_sim3_path.expanduser().resolve() if args.init_sim3_path is not None else None,
        mask_path=args.mask_path.expanduser().resolve() if args.mask_path is not None else None,
        mast3r_dir=mast3r_dir,
        device=args.device,
        dtype=args.dtype,
        max_image_size=args.max_image_size,
        camera_batch_size=max(1, args.camera_batch_size),
        line_sample_step=max(1.0e-3, args.line_sample_step),
        dt_trunc=max(0.0, args.dt_trunc),
        dt_sigma=max(1.0e-6, args.dt_sigma),
        charbonnier_eps=max(1.0e-12, args.charbonnier_eps),
        z_min=max(1.0e-8, args.z_min),
        sim_lr=max(1.0e-6, args.sim_lr),
        pose_lr=max(1.0e-6, args.pose_lr),
        sim_phase_iters=max(0, args.sim_phase_iters),
        local_phase_iters=max(0, args.local_phase_iters),
        global_phase_iters=max(0, args.global_phase_iters),
        num_alternations=max(0, args.num_alternations),
        lambda_pose=max(0.0, args.lambda_pose),
        lambda_plane=max(0.0, args.lambda_plane),
        lambda_init=max(0.0, args.lambda_init),
        lambda_visibility=max(0.0, args.lambda_visibility),
        pose_rot_sigma_deg=max(1.0e-6, args.pose_rot_sigma_deg),
        pose_trans_sigma=args.pose_trans_sigma,
        plane_rot_sigma_deg=max(1.0e-6, args.plane_rot_sigma_deg),
        plane_height_sigma=max(1.0e-6, args.plane_height_sigma),
        init_rot_sigma_deg=max(1.0e-6, args.init_rot_sigma_deg),
        init_trans_sigma=max(1.0e-6, args.init_trans_sigma),
        init_scale_sigma_log=max(1.0e-6, args.init_scale_sigma_log),
        init_gap_sigma=max(1.0e-6, args.init_gap_sigma),
        min_visible_fraction=float(np.clip(args.min_visible_fraction, 0.0, 1.0)),
        adjacent_court_direction=args.adjacent_court_direction,
        init_adjacent_gap=max(0.0, args.init_adjacent_gap),
        court_base_width=max(1.0, args.court_base_width),
        log_interval=max(1, args.log_interval),
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float64":
        return torch.float64
    return torch.float32


def load_train_stems(scene_dir: Path) -> list[str]:
    path = scene_dir / "images_train.txt"
    if not path.exists():
        raise FileNotFoundError(f"Train split not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_first_image(scene_dir: Path, stem: str) -> Path:
    image_dir = scene_dir / "images"
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image for train stem {stem!r} under {image_dir}")


def load_initial_global_prior(
    path: Path | None,
    fallback_gap: float,
) -> InitialGlobalPrior:
    if path is None:
        return InitialGlobalPrior(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            adjacent_gap=fallback_gap,
            plane_normal=None,
            plane_d=None,
            enabled=False,
        )

    suffix = path.suffix.lower()
    plane_normal: np.ndarray | None = None
    plane_d: float | None = None
    if suffix == ".npy":
        matrix = np.load(path)
        gap = fallback_gap
    elif suffix == ".npz":
        payload = np.load(path)
        gap = float(payload["adjacent_gap"]) if "adjacent_gap" in payload else fallback_gap
        if "plane_normal" in payload:
            plane_normal = np.asarray(payload["plane_normal"], dtype=np.float64)
        if "plane_d" in payload:
            plane_d = float(payload["plane_d"])
        if "matrix" in payload:
            matrix = payload["matrix"]
        elif {"scale", "rotation", "translation"} <= set(payload.files):
            return InitialGlobalPrior(
                scale=float(payload["scale"]),
                rotation=np.asarray(payload["rotation"], dtype=np.float64),
                translation=np.asarray(payload["translation"], dtype=np.float64),
                adjacent_gap=gap,
                plane_normal=plane_normal,
                plane_d=plane_d,
                enabled=True,
            )
        else:
            raise ValueError(f"Unsupported npz structure for initial Sim(3): {path}")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        gap = float(payload.get("adjacent_gap", fallback_gap))
        if "plane_normal" in payload:
            plane_normal = np.asarray(payload["plane_normal"], dtype=np.float64)
        if "plane_d" in payload:
            plane_d = float(payload["plane_d"])
        if "matrix" in payload:
            matrix = np.asarray(payload["matrix"], dtype=np.float64)
        elif {"scale", "rotation", "translation"} <= set(payload):
            return InitialGlobalPrior(
                scale=float(payload["scale"]),
                rotation=np.asarray(payload["rotation"], dtype=np.float64),
                translation=np.asarray(payload["translation"], dtype=np.float64),
                adjacent_gap=gap,
                plane_normal=plane_normal,
                plane_d=plane_d,
                enabled=True,
            )
        else:
            raise ValueError(f"Unsupported json structure for initial Sim(3): {path}")
    else:
        raise ValueError(f"Unsupported init-sim3 format: {path}")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 similarity matrix, got {matrix.shape}")
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(np.linalg.norm(linear, axis=0)))
    rotation = linear / scale
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    translation = matrix[:3, 3]
    return InitialGlobalPrior(
        scale=scale,
        rotation=rotation,
        translation=translation,
        adjacent_gap=gap,
        plane_normal=plane_normal,
        plane_d=plane_d,
        enabled=True,
    )


def rodrigues_np(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotvec_from_matrix_np(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(math.acos(trace))
    if theta < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - math.pi) < 1.0e-5:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1.0e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= axis_norm
        return axis * theta
    skew = (rotation - rotation.T) / (2.0 * math.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    return axis * theta


def skew_matrix(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            torch.stack((zeros, -v[..., 2], v[..., 1]), dim=-1),
            torch.stack((v[..., 2], zeros, -v[..., 0]), dim=-1),
            torch.stack((-v[..., 1], v[..., 0], zeros), dim=-1),
        ),
        dim=-2,
    )


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta_safe = theta.clamp_min(1.0e-12)
    theta2_safe = theta_safe * theta_safe
    K = skew_matrix(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(rotvec.shape[:-1] + (3, 3))

    A_exact = torch.sin(theta_safe) / theta_safe
    B_exact = (1.0 - torch.cos(theta_safe)) / theta2_safe
    A = torch.where(
        theta > 1.0e-6,
        A_exact,
        1.0 - theta2 / 6.0 + theta4 / 120.0,
    )
    B = torch.where(
        theta > 1.0e-6,
        B_exact,
        0.5 - theta2 / 24.0 + theta4 / 720.0,
    )
    return eye + A[..., None] * K + B[..., None] * (K @ K)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    omega = xi[..., :3]
    tau = xi[..., 3:]
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta_safe = theta.clamp_min(1.0e-12)
    theta2_safe = theta_safe * theta_safe
    K = skew_matrix(omega)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    eye = eye.expand(xi.shape[:-1] + (3, 3))

    A_exact = torch.sin(theta_safe) / theta_safe
    B_exact = (1.0 - torch.cos(theta_safe)) / theta2_safe
    C_exact = (theta_safe - torch.sin(theta_safe)) / (theta_safe * theta2_safe)
    A = torch.where(
        theta > 1.0e-6,
        A_exact,
        1.0 - theta2 / 6.0 + theta4 / 120.0,
    )
    B = torch.where(
        theta > 1.0e-6,
        B_exact,
        0.5 - theta2 / 24.0 + theta4 / 720.0,
    )
    C = torch.where(
        theta > 1.0e-6,
        C_exact,
        1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0,
    )

    rot = eye + A[..., None] * K + B[..., None] * (K @ K)
    V = eye + B[..., None] * K + C[..., None] * (K @ K)
    trans = torch.matmul(V, tau.unsqueeze(-1)).squeeze(-1)

    transform = torch.eye(4, dtype=xi.dtype, device=xi.device)
    transform = transform.expand(xi.shape[:-1] + (4, 4)).clone()
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = trans
    return transform


def so3_relative_angle(rotation: torch.Tensor) -> torch.Tensor:
    trace = (torch.trace(rotation) - 1.0) * 0.5
    return torch.acos(torch.clamp(trace, -1.0 + 1.0e-7, 1.0 - 1.0e-7))


def vector_angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(u) * torch.linalg.norm(v)
    cos = torch.dot(u, v) / denom.clamp_min(1.0e-8)
    return torch.acos(torch.clamp(cos, -1.0 + 1.0e-7, 1.0 - 1.0e-7))


def softplus_inverse(x: float) -> float:
    x = max(float(x), 1.0e-9)
    return float(np.log(np.expm1(x)))


def adjacent_direction_vector(direction: str) -> np.ndarray:
    if direction == "+x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if direction == "-x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if direction == "+y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if direction == "-y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported adjacent court direction: {direction}")


def build_ground_line_samples(step: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    keypoints = court_keypoints_3d().cpu().numpy().astype(np.float64)
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    segments_meta: list[dict[str, Any]] = []

    for seg_idx, (i0, i1) in enumerate(COURT_SKELETON):
        p0 = keypoints[i0]
        p1 = keypoints[i1]
        if not (abs(p0[2]) < 1.0e-9 and abs(p1[2]) < 1.0e-9):
            continue
        length = float(np.linalg.norm(p1 - p0))
        num = max(2, int(math.ceil(length / step)) + 1)
        alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
        samples = (1.0 - alpha[:, None]) * p0[None, :] + alpha[:, None] * p1[None, :]
        sample_weight = np.full((num,), length / max(num, 1), dtype=np.float64)
        points.append(samples)
        weights.append(sample_weight)
        segments_meta.append(
            {
                "segment_index": seg_idx,
                "kp0": int(i0),
                "kp1": int(i1),
                "length_m": length,
                "num_samples": int(num),
            }
        )

    if not points:
        raise RuntimeError("No ground-plane court line segments were found.")
    return np.concatenate(points, axis=0), np.concatenate(weights, axis=0), segments_meta


def resize_mask_and_intrinsics(mask: np.ndarray, K_full: np.ndarray, max_image_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    if max_image_size <= 0 or max(height, width) <= max_image_size:
        return mask, K_full
    scale = float(max_image_size) / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    K_resized = K_full.copy()
    K_resized[0, :] *= new_w / width
    K_resized[1, :] *= new_h / height
    return resized, K_resized


def compute_distance_transform(mask: np.ndarray, truncation: float) -> np.ndarray:
    line_mask = mask > 0
    if not np.any(line_mask):
        return np.full(mask.shape, truncation, dtype=np.float32)
    dt = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 3)
    if truncation > 0.0:
        dt = np.minimum(dt, truncation)
    return dt.astype(np.float32, copy=False)


def scale_intrinsics_to_fullres(K: np.ndarray, image_width: int, base_width: float) -> np.ndarray:
    scale = float(image_width) / float(base_width)
    K_full = np.asarray(K, dtype=np.float64).copy()
    K_full[:2, :] *= scale
    K_full[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return K_full


def world_to_camera_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w)


class CourtAlignmentProblem:
    def __init__(
        self,
        dt_images: torch.Tensor,
        intrinsics: torch.Tensor,
        world_to_cam_init: torch.Tensor,
        single_court_points: torch.Tensor,
        single_point_weights: torch.Tensor,
        active_cameras: torch.Tensor,
        *,
        dt_sigma: float,
        charbonnier_eps: float,
        z_min: float,
        lambda_pose: float,
        lambda_plane: float,
        lambda_init: float,
        lambda_visibility: float,
        pose_rot_sigma: float,
        pose_trans_sigma: float,
        plane_rot_sigma: float,
        plane_height_sigma: float,
        init_rot_sigma: float,
        init_trans_sigma: float,
        init_scale_sigma_log: float,
        init_gap_sigma: float,
        min_visible_fraction: float,
        camera_batch_size: int,
        adjacent_direction: torch.Tensor,
        init_prior: InitialGlobalPrior,
    ) -> None:
        self.dt_images = dt_images
        self.intrinsics = intrinsics
        self.world_to_cam_init = world_to_cam_init
        self.single_court_points = single_court_points
        self.single_point_weights = single_point_weights
        self.point_weights = torch.cat((single_point_weights, single_point_weights), dim=0)
        self.active_cameras = active_cameras
        self.dt_sigma = dt_sigma
        self.charbonnier_eps = charbonnier_eps
        self.z_min = z_min
        self.lambda_pose = lambda_pose
        self.lambda_plane = lambda_plane
        self.lambda_init = lambda_init
        self.lambda_visibility = lambda_visibility
        self.pose_rot_sigma = pose_rot_sigma
        self.pose_trans_sigma = pose_trans_sigma
        self.plane_rot_sigma = plane_rot_sigma
        self.plane_height_sigma = plane_height_sigma
        self.init_rot_sigma = init_rot_sigma
        self.init_trans_sigma = init_trans_sigma
        self.init_scale_sigma_log = init_scale_sigma_log
        self.init_gap_sigma = init_gap_sigma
        self.min_visible_fraction = min_visible_fraction
        self.camera_batch_size = camera_batch_size
        self.adjacent_direction = adjacent_direction
        self.base_center_offset = float(DOUBLES_WIDTH)
        self.num_cameras = int(dt_images.shape[0])
        self.height = int(dt_images.shape[-2])
        self.width = int(dt_images.shape[-1])
        self.init_prior_enabled = bool(init_prior.enabled)
        self.init_log_scale = float(math.log(max(init_prior.scale, 1.0e-8)))
        self.init_rotation = torch.from_numpy(init_prior.rotation).to(
            device=dt_images.device, dtype=dt_images.dtype
        )
        self.init_translation = torch.from_numpy(init_prior.translation).to(
            device=dt_images.device, dtype=dt_images.dtype
        )
        self.init_gap = float(init_prior.adjacent_gap)
        if init_prior.plane_normal is not None and init_prior.plane_d is not None:
            plane_normal = np.asarray(init_prior.plane_normal, dtype=np.float64)
            plane_norm = float(np.linalg.norm(plane_normal))
            if plane_norm > 1.0e-9:
                plane_normal = plane_normal / plane_norm
                plane_d = float(init_prior.plane_d) / plane_norm
                self.init_plane_normal = torch.from_numpy(plane_normal).to(
                    device=dt_images.device, dtype=dt_images.dtype
                )
                self.init_plane_d = torch.tensor(
                    plane_d, device=dt_images.device, dtype=dt_images.dtype
                )
            else:
                self.init_plane_normal = None
                self.init_plane_d = None
        else:
            self.init_plane_normal = None
            self.init_plane_d = None

    def global_components(
        self,
        global_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_scale = global_params[0]
        rotvec = global_params[1:4]
        translation = global_params[4:7]
        raw_gap = global_params[7]
        scale = torch.exp(log_scale)
        rotation = so3_exp(rotvec.unsqueeze(0))[0]
        gap = F.softplus(raw_gap)
        center_offset = self.base_center_offset + gap
        return scale, rotation, translation, gap, center_offset

    def transformed_court_points(self, global_params: torch.Tensor) -> torch.Tensor:
        scale, rotation, translation, _, center_offset = self.global_components(global_params)
        second_shift = self.adjacent_direction * center_offset
        local_points = torch.cat(
            (
                self.single_court_points,
                self.single_court_points + second_shift.unsqueeze(0),
            ),
            dim=0,
        )
        return scale * torch.matmul(local_points, rotation.transpose(0, 1)) + translation

    def refined_world_to_cam(self, pose_deltas: torch.Tensor) -> torch.Tensor:
        delta = se3_exp(pose_deltas)
        return torch.matmul(delta, self.world_to_cam_init)

    def dt_loss_and_visibility(
        self,
        global_params: torch.Tensor,
        pose_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_points = self.transformed_court_points(global_params)
        world_to_cam = self.refined_world_to_cam(pose_deltas)

        total_loss = world_points.sum() * 0.0
        visible_weight = world_points.sum() * 0.0
        possible_weight = world_points.sum() * 0.0

        for start in range(0, self.num_cameras, self.camera_batch_size):
            end = min(self.num_cameras, start + self.camera_batch_size)
            active = self.active_cameras[start:end]
            if not bool(torch.any(active).item()):
                continue

            dt = self.dt_images[start:end]
            K = self.intrinsics[start:end]
            Tcw = world_to_cam[start:end]

            Rcw = Tcw[:, :3, :3]
            tcw = Tcw[:, :3, 3]
            cam_points = torch.matmul(world_points.unsqueeze(0), Rcw.transpose(1, 2)) + tcw[:, None, :]
            z = cam_points[..., 2]

            x = cam_points[..., 0]
            y = cam_points[..., 1]
            fx = K[:, 0, 0][:, None]
            fy = K[:, 1, 1][:, None]
            cx = K[:, 0, 2][:, None]
            cy = K[:, 1, 2][:, None]
            u = fx * (x / z.clamp_min(self.z_min)) + cx
            v = fy * (y / z.clamp_min(self.z_min)) + cy

            visible = (
                active[:, None]
                & (z > self.z_min)
                & (u >= 0.0)
                & (u <= self.width - 1.0)
                & (v >= 0.0)
                & (v <= self.height - 1.0)
            )
            batch_possible = active[:, None].to(world_points.dtype) * self.point_weights[None, :]
            possible_weight = possible_weight + torch.sum(batch_possible)
            if not bool(torch.any(visible).item()):
                continue

            grid_x = 2.0 * (u / max(self.width - 1, 1)) - 1.0
            grid_y = 2.0 * (v / max(self.height - 1, 1)) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(-2)
            sampled = F.grid_sample(
                dt,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(1).squeeze(-1)

            residual = sampled / self.dt_sigma
            robust = torch.sqrt(residual * residual + self.charbonnier_eps * self.charbonnier_eps)
            weights = visible.to(world_points.dtype) * self.point_weights[None, :]
            total_loss = total_loss + torch.sum(weights * robust)
            visible_weight = visible_weight + torch.sum(weights)

        dt_loss = total_loss / visible_weight.clamp_min(1.0e-8)
        visible_frac = visible_weight / possible_weight.clamp_min(1.0e-8)
        return dt_loss, visible_frac

    def dt_loss_only(self, global_params: torch.Tensor, pose_deltas: torch.Tensor) -> torch.Tensor:
        return self.dt_loss_and_visibility(global_params, pose_deltas)[0]

    def pose_regularizer(self, pose_deltas: torch.Tensor) -> torch.Tensor:
        rot = pose_deltas[:, :3]
        trans = pose_deltas[:, 3:]
        reg = (
            rot.square().sum(dim=-1) / (self.pose_rot_sigma * self.pose_rot_sigma)
            + trans.square().sum(dim=-1) / (self.pose_trans_sigma * self.pose_trans_sigma)
        )
        active = self.active_cameras.to(reg.dtype)
        return torch.sum(reg * active) / active.sum().clamp_min(1.0)

    def global_regularizers(
        self,
        global_params: torch.Tensor,
        visible_frac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, rotation, translation, gap, _ = self.global_components(global_params)
        zero = global_params.sum() * 0.0

        plane_loss = zero
        if self.init_plane_normal is not None and self.init_plane_d is not None:
            current_normal = rotation[:, 2]
            plane_angle = vector_angle(current_normal, self.init_plane_normal)
            plane_height = torch.dot(self.init_plane_normal, translation) - self.init_plane_d
            plane_loss = (
                plane_angle.square() / (self.plane_rot_sigma * self.plane_rot_sigma)
                + plane_height.square() / (self.plane_height_sigma * self.plane_height_sigma)
            )

        init_loss = zero
        if self.init_prior_enabled:
            delta_log_scale = global_params[0] - self.init_log_scale
            rel_rot = torch.matmul(self.init_rotation.transpose(0, 1), rotation)
            init_angle = so3_relative_angle(rel_rot)
            delta_t = translation - self.init_translation
            if self.init_plane_normal is not None:
                delta_t = delta_t - torch.dot(delta_t, self.init_plane_normal) * self.init_plane_normal
            init_loss = (
                delta_log_scale.square() / (self.init_scale_sigma_log * self.init_scale_sigma_log)
                + init_angle.square() / (self.init_rot_sigma * self.init_rot_sigma)
                + delta_t.square().sum() / (self.init_trans_sigma * self.init_trans_sigma)
                + (gap - self.init_gap) * (gap - self.init_gap) / (self.init_gap_sigma * self.init_gap_sigma)
            )

        visibility_loss = torch.relu(self.min_visible_fraction - visible_frac).square()
        return plane_loss, init_loss, visibility_loss

    def total_loss(
        self,
        global_params: torch.Tensor,
        pose_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dt_loss, visible_frac = self.dt_loss_and_visibility(global_params, pose_deltas)
        pose_loss = self.pose_regularizer(pose_deltas)
        plane_loss, init_loss, visibility_loss = self.global_regularizers(global_params, visible_frac)
        total = (
            dt_loss
            + self.lambda_pose * pose_loss
            + self.lambda_plane * plane_loss
            + self.lambda_init * init_loss
            + self.lambda_visibility * visibility_loss
        )
        return total, dt_loss, pose_loss, plane_loss, init_loss, visibility_loss

    @torch.no_grad()
    def evaluate(self, global_params: torch.Tensor, pose_deltas: torch.Tensor) -> dict[str, float]:
        total, dt_loss, pose_loss, plane_loss, init_loss, visibility_loss = self.total_loss(global_params, pose_deltas)
        _, visible_frac = self.dt_loss_and_visibility(global_params, pose_deltas)
        _, _, _, gap, center_offset = self.global_components(global_params)
        return {
            "total_loss": float(total.item()),
            "dt_loss": float(dt_loss.item()),
            "pose_loss": float(pose_loss.item()),
            "plane_loss": float(plane_loss.item()),
            "init_loss": float(init_loss.item()),
            "visibility_loss": float(visibility_loss.item()),
            "visible_fraction": float(visible_frac.item()),
            "adjacent_gap": float(gap.item()),
            "adjacent_center_offset": float(center_offset.item()),
        }


def optimize_sim_only(
    problem: CourtAlignmentProblem,
    global_params: torch.nn.Parameter,
    pose_deltas: torch.Tensor,
    *,
    lr: float,
    iters: int,
    log_interval: int,
    label: str,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if iters <= 0:
        return history

    optimizer = torch.optim.Adam([global_params], lr=lr)
    pose_deltas_detached = pose_deltas.detach()
    for step in range(iters):
        optimizer.zero_grad(set_to_none=True)
        total, _, _, _, _, _ = problem.total_loss(global_params, pose_deltas_detached)
        total.backward()
        optimizer.step()
        if step == 0 or step == iters - 1 or (step + 1) % log_interval == 0:
            metrics = problem.evaluate(global_params.detach(), pose_deltas_detached)
            metrics["iter"] = step + 1
            history.append(metrics)
            print(
                f"[{label}] iter {step + 1:04d}/{iters:04d} "
                f"total={metrics['total_loss']:.6f} dt={metrics['dt_loss']:.6f} "
                f"pose={metrics['pose_loss']:.6f} plane={metrics['plane_loss']:.6f} "
                f"init={metrics['init_loss']:.6f} vis={metrics['visible_fraction']:.4f} "
                f"gap={metrics['adjacent_gap']:.6f}",
                flush=True,
            )
    return history


def optimize_pose_offsets(
    problem: CourtAlignmentProblem,
    global_params: torch.Tensor,
    pose_deltas: torch.nn.Parameter,
    *,
    lr: float,
    iters: int,
    log_interval: int,
    label: str,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if iters <= 0:
        return history

    optimizer = torch.optim.Adam([pose_deltas], lr=lr)
    global_params_detached = global_params.detach()
    for step in range(iters):
        optimizer.zero_grad(set_to_none=True)
        total, _, _, _, _, _ = problem.total_loss(global_params_detached, pose_deltas)
        total.backward()
        optimizer.step()
        if step == 0 or step == iters - 1 or (step + 1) % log_interval == 0:
            metrics = problem.evaluate(global_params_detached, pose_deltas.detach())
            metrics["iter"] = step + 1
            history.append(metrics)
            print(
                f"[{label}] iter {step + 1:04d}/{iters:04d} "
                f"total={metrics['total_loss']:.6f} dt={metrics['dt_loss']:.6f} "
                f"pose={metrics['pose_loss']:.6f} plane={metrics['plane_loss']:.6f} "
                f"init={metrics['init_loss']:.6f} vis={metrics['visible_fraction']:.4f} "
                f"gap={metrics['adjacent_gap']:.6f}",
                flush=True,
            )
    return history


def save_outputs(
    cfg: Config,
    train_stems: list[str],
    global_params: torch.Tensor,
    pose_deltas: torch.Tensor,
    world_to_cam_init: torch.Tensor,
    histories: dict[str, Any],
    active_cameras: np.ndarray,
    point_meta: list[dict[str, Any]],
    pose_trans_sigma: float,
    dt_shape: tuple[int, int],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    global_np = global_params.detach().cpu().numpy()
    pose_np = pose_deltas.detach().cpu().numpy()
    scale = float(np.exp(global_np[0]))
    rotation = rodrigues_np(global_np[1:4])
    translation = global_np[4:7].astype(np.float64)
    adjacent_gap = float(torch.nn.functional.softplus(torch.tensor(global_np[7], dtype=torch.float64)).item())
    adjacent_center_offset = float(DOUBLES_WIDTH + adjacent_gap)
    adjacent_dir = adjacent_direction_vector(cfg.adjacent_court_direction)
    sim_matrix = np.eye(4, dtype=np.float64)
    sim_matrix[:3, :3] = scale * rotation
    sim_matrix[:3, 3] = translation

    world_to_cam_refined = (
        se3_exp(pose_deltas.detach()).cpu().numpy() @ world_to_cam_init.detach().cpu().numpy()
    )
    camera_poses_refined = np.linalg.inv(world_to_cam_refined)

    np.save(cfg.output_dir / "camera_pose_offsets.npy", pose_np.astype(np.float32))
    np.save(cfg.output_dir / "world_to_camera_refined.npy", world_to_cam_refined.astype(np.float32))
    np.save(cfg.output_dir / "camera_poses_refined.npy", camera_poses_refined.astype(np.float32))
    np.savez(
        cfg.output_dir / "sim3_refined.npz",
        log_scale=np.array(global_np[0], dtype=np.float64),
        scale=np.array(scale, dtype=np.float64),
        rotation=rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        matrix=sim_matrix.astype(np.float64),
        adjacent_gap=np.array(adjacent_gap, dtype=np.float64),
        adjacent_center_offset=np.array(adjacent_center_offset, dtype=np.float64),
        adjacent_direction=adjacent_dir.astype(np.float64),
    )

    metrics = {
        "config": {
            **asdict(cfg),
            "scene_dir": str(cfg.scene_dir),
            "output_dir": str(cfg.output_dir),
            "init_sim3_path": str(cfg.init_sim3_path) if cfg.init_sim3_path is not None else None,
            "mask_path": str(cfg.mask_path) if cfg.mask_path is not None else None,
            "mast3r_dir": str(cfg.mast3r_dir) if cfg.mast3r_dir is not None else None,
        },
        "num_train_cameras": len(train_stems),
        "active_camera_count": int(active_cameras.sum()),
        "active_cameras": active_cameras.astype(bool).tolist(),
        "train_stems": train_stems,
        "dt_image_size": list(dt_shape),
        "pose_trans_sigma": float(pose_trans_sigma),
        "sim3": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "matrix": sim_matrix.tolist(),
        },
        "court_pair": {
            "adjacent_gap": adjacent_gap,
            "adjacent_center_offset": adjacent_center_offset,
            "adjacent_court_direction": cfg.adjacent_court_direction,
            "adjacent_direction_vector": adjacent_dir.tolist(),
            "base_doubles_width": float(DOUBLES_WIDTH),
            "court_centers_in_base_court_frame": [
                [0.0, 0.0, 0.0],
                (adjacent_dir * adjacent_center_offset).tolist(),
            ],
        },
        "histories": histories,
        "ground_segments": point_meta,
    }
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)
    mast3r_dir = cfg.mast3r_dir if cfg.mast3r_dir is not None else cfg.scene_dir / "mast3r"
    mask_path = cfg.mask_path if cfg.mask_path is not None else mast3r_dir / "court_line_masks.npy"

    train_stems = load_train_stems(cfg.scene_dir)
    intrinsics_path = mast3r_dir / "camera_intrinsics.npy"
    camera_poses_path = mast3r_dir / "camera_poses.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
    if not camera_poses_path.exists():
        raise FileNotFoundError(f"Missing camera poses: {camera_poses_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing court line masks: {mask_path}")

    intrinsics = np.load(intrinsics_path).astype(np.float64)
    camera_poses = np.load(camera_poses_path).astype(np.float64)
    masks = np.load(mask_path, mmap_mode="r")
    if intrinsics.shape[0] != len(train_stems):
        raise ValueError(f"Intrinsics count {intrinsics.shape[0]} does not match train split length {len(train_stems)}")
    if camera_poses.shape[0] != len(train_stems):
        raise ValueError(f"Pose count {camera_poses.shape[0]} does not match train split length {len(train_stems)}")
    if masks.shape[0] != len(train_stems):
        raise ValueError(f"Mask count {masks.shape[0]} does not match train split length {len(train_stems)}")

    first_image_path = find_first_image(cfg.scene_dir, train_stems[0])
    image_bgr = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {first_image_path}")
    full_h, full_w = image_bgr.shape[:2]
    if tuple(masks.shape[1:]) != (full_h, full_w):
        raise ValueError(
            f"Mask shape {masks.shape[1:]} does not match full image size {(full_h, full_w)}"
        )

    dt_images: list[np.ndarray] = []
    intrinsics_resized: list[np.ndarray] = []
    active_cameras = np.zeros((len(train_stems),), dtype=bool)

    print(f"Preparing {len(train_stems)} train masks...", flush=True)
    for cam_idx in range(len(train_stems)):
        K_full = scale_intrinsics_to_fullres(intrinsics[cam_idx], full_w, cfg.court_base_width)
        mask_uint8 = np.asarray(masks[cam_idx], dtype=np.uint8)
        resized_mask, resized_K = resize_mask_and_intrinsics(mask_uint8, K_full, cfg.max_image_size)
        dt_images.append(compute_distance_transform(resized_mask, cfg.dt_trunc))
        intrinsics_resized.append(resized_K.astype(np.float32))
        active_cameras[cam_idx] = bool(np.any(resized_mask > 0))

    dt_array = np.stack(dt_images, axis=0)
    K_array = np.stack(intrinsics_resized, axis=0)
    world_to_cam_init = np.stack([world_to_camera_from_c2w(c2w) for c2w in camera_poses], axis=0).astype(np.float32)

    court_points_np, point_weights_np, point_meta = build_ground_line_samples(cfg.line_sample_step)

    scene_centers = camera_poses[:, :3, 3]
    scene_centered = scene_centers - scene_centers.mean(axis=0, keepdims=True)
    scene_rms = float(np.sqrt(np.mean(np.sum(scene_centered * scene_centered, axis=1))))
    pose_trans_sigma = cfg.pose_trans_sigma
    if pose_trans_sigma is None:
        pose_trans_sigma = max(1.0e-4, 0.02 * scene_rms)

    init_prior = load_initial_global_prior(
        cfg.init_sim3_path,
        cfg.init_adjacent_gap,
    )
    global_init = np.concatenate(
        (
            np.array([math.log(init_prior.scale)], dtype=np.float64),
            rotvec_from_matrix_np(init_prior.rotation),
            np.asarray(init_prior.translation, dtype=np.float64),
            np.array([softplus_inverse(init_prior.adjacent_gap)], dtype=np.float64),
        ),
        axis=0,
    )

    dt_tensor = torch.from_numpy(dt_array).to(device=device, dtype=dtype).unsqueeze(1)
    K_tensor = torch.from_numpy(K_array).to(device=device, dtype=dtype)
    Tcw_init_tensor = torch.from_numpy(world_to_cam_init).to(device=device, dtype=dtype)
    court_points = torch.from_numpy(court_points_np).to(device=device, dtype=dtype)
    point_weights = torch.from_numpy(point_weights_np).to(device=device, dtype=dtype)
    active_tensor = torch.from_numpy(active_cameras).to(device=device)
    adjacent_dir = torch.from_numpy(adjacent_direction_vector(cfg.adjacent_court_direction)).to(device=device, dtype=dtype)
    global_params = torch.nn.Parameter(torch.from_numpy(global_init).to(device=device, dtype=dtype))
    pose_deltas = torch.nn.Parameter(torch.zeros((len(train_stems), 6), device=device, dtype=dtype))

    problem = CourtAlignmentProblem(
        dt_images=dt_tensor,
        intrinsics=K_tensor,
        world_to_cam_init=Tcw_init_tensor,
        single_court_points=court_points,
        single_point_weights=point_weights,
        active_cameras=active_tensor,
        dt_sigma=cfg.dt_sigma,
        charbonnier_eps=cfg.charbonnier_eps,
        z_min=cfg.z_min,
        lambda_pose=cfg.lambda_pose,
        lambda_plane=cfg.lambda_plane,
        lambda_init=cfg.lambda_init,
        lambda_visibility=cfg.lambda_visibility,
        pose_rot_sigma=math.radians(cfg.pose_rot_sigma_deg),
        pose_trans_sigma=pose_trans_sigma,
        plane_rot_sigma=math.radians(cfg.plane_rot_sigma_deg),
        plane_height_sigma=cfg.plane_height_sigma,
        init_rot_sigma=math.radians(cfg.init_rot_sigma_deg),
        init_trans_sigma=cfg.init_trans_sigma,
        init_scale_sigma_log=cfg.init_scale_sigma_log,
        init_gap_sigma=cfg.init_gap_sigma,
        min_visible_fraction=cfg.min_visible_fraction,
        camera_batch_size=cfg.camera_batch_size,
        adjacent_direction=adjacent_dir,
        init_prior=init_prior,
    )

    init_gap = float(F.softplus(global_params.detach()[7]).item())
    plane_prior_enabled = init_prior.plane_normal is not None and init_prior.plane_d is not None
    print(
        f"Device={device} dtype={dtype} dt_size={dt_array.shape[1]}x{dt_array.shape[2]} "
        f"active_cameras={int(active_cameras.sum())}/{len(active_cameras)} "
        f"scene_rms={scene_rms:.6f} pose_trans_sigma={pose_trans_sigma:.6f} "
        f"adjacent_dir={cfg.adjacent_court_direction} init_gap={init_gap:.6f} "
        f"init_prior={init_prior.enabled} plane_prior={plane_prior_enabled}",
        flush=True,
    )

    histories: dict[str, Any] = {
        "phase1_sim": optimize_sim_only(
            problem,
            global_params,
            pose_deltas.detach(),
            lr=cfg.sim_lr,
            iters=cfg.sim_phase_iters,
            log_interval=cfg.log_interval,
            label="phase1/sim",
        ),
        "alternations": [],
    }

    for alt_idx in range(cfg.num_alternations):
        print(f"[alt {alt_idx + 1}/{cfg.num_alternations}] local pose refinement", flush=True)
        local_history = optimize_pose_offsets(
            problem,
            global_params.detach(),
            pose_deltas,
            lr=cfg.pose_lr,
            iters=cfg.local_phase_iters,
            log_interval=cfg.log_interval,
            label=f"alt{alt_idx + 1}/local",
        )
        print(f"[alt {alt_idx + 1}/{cfg.num_alternations}] global Sim(3) refinement", flush=True)
        global_history = optimize_sim_only(
            problem,
            global_params,
            pose_deltas.detach(),
            lr=cfg.sim_lr,
            iters=cfg.global_phase_iters,
            log_interval=cfg.log_interval,
            label=f"alt{alt_idx + 1}/global",
        )
        histories["alternations"].append(
            {
                "index": alt_idx + 1,
                "local": local_history,
                "global": global_history,
                "final_metrics": problem.evaluate(global_params.detach(), pose_deltas.detach()),
            }
        )

    final_metrics = problem.evaluate(global_params.detach(), pose_deltas.detach())
    histories["final"] = final_metrics
    print(
        f"[final] total={final_metrics['total_loss']:.6f} "
        f"dt={final_metrics['dt_loss']:.6f} pose={final_metrics['pose_loss']:.6f} "
        f"plane={final_metrics['plane_loss']:.6f} init={final_metrics['init_loss']:.6f} "
        f"vis={final_metrics['visible_fraction']:.4f} "
        f"gap={final_metrics['adjacent_gap']:.6f}",
        flush=True,
    )

    save_outputs(
        cfg=cfg,
        train_stems=train_stems,
        global_params=global_params.detach(),
        pose_deltas=pose_deltas.detach(),
        world_to_cam_init=Tcw_init_tensor.detach(),
        histories=histories,
        active_cameras=active_cameras,
        point_meta=point_meta,
        pose_trans_sigma=pose_trans_sigma,
        dt_shape=tuple(dt_array.shape[1:]),
    )
    print(f"Outputs written to {cfg.output_dir}", flush=True)


if __name__ == "__main__":
    main()
