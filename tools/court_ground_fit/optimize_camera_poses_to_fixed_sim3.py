#!/usr/bin/env python3
"""Optimize camera poses against a fixed court Sim(3).

This stage keeps the court transform fixed and adjusts MASt3R camera poses to
better align image-space court masks and ground-plane court geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.court_scheme import COURT_SKELETON, DOUBLES_WIDTH, court_keypoints_3d


@dataclass
class PrecomputedData:
    train_stems: list[str]
    masks_down: np.ndarray
    image_dt: np.ndarray
    sample_pixels: np.ndarray
    sample_valid: np.ndarray
    K_down: np.ndarray
    base_poses: np.ndarray
    cluster_mask: np.ndarray
    other_mask: np.ndarray
    fixed_court_world_points: np.ndarray
    fixed_court_world_keypoints: np.ndarray
    fixed_court_plane_mask: np.ndarray
    plane_dt_m: np.ndarray
    plane_extent: tuple[float, float, float, float]
    plane_resolution: float
    plane_origin: np.ndarray
    plane_axis_x: np.ndarray
    plane_axis_y: np.ndarray
    plane_normal: np.ndarray
    plane_d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/tennis_court"))
    parser.add_argument("--mast3r-dir", type=Path, default=None)
    parser.add_argument("--ground-dir", type=Path, default=None)
    parser.add_argument("--transform-dir", type=Path, default=None)
    parser.add_argument(
        "--fixed-sim3-path",
        type=Path,
        default=None,
        help="Defaults to <transform-dir>/ground_heatmap_fit_sim3.json.",
    )
    parser.add_argument(
        "--cluster-source-path",
        type=Path,
        default=None,
        help="Defaults to <transform-dir>/ground_heatmap_fit.json for dominant_cameras.",
    )
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--pose-opt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tennis_court/court/pose_opt"))
    parser.add_argument("--court-base-width", type=float, default=512.0)
    parser.add_argument("--image-downsample", type=int, default=8)
    parser.add_argument("--cam-samples-per-camera", type=int, default=1536)
    parser.add_argument("--court-line-step-m", type=float, default=0.20)
    parser.add_argument("--court-line-thickness-px", type=int, default=2)
    parser.add_argument("--iters", type=int, default=220)
    parser.add_argument("--lr-other", type=float, default=2.0e-2)
    parser.add_argument("--lr-local", type=float, default=8.0e-3)
    parser.add_argument("--lambda-court-to-cam", type=float, default=1.0)
    parser.add_argument("--lambda-cam-to-court", type=float, default=1.0)
    parser.add_argument("--lambda-reg-other", type=float, default=0.02)
    parser.add_argument("--lambda-reg-local", type=float, default=0.10)
    parser.add_argument("--lambda-reg-cluster", type=float, default=0.20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def adjacent_direction_vector(direction: str) -> np.ndarray:
    if direction == "+x":
        return np.array([1.0, 0.0], dtype=np.float64)
    if direction == "-x":
        return np.array([-1.0, 0.0], dtype=np.float64)
    if direction == "+y":
        return np.array([0.0, 1.0], dtype=np.float64)
    if direction == "-y":
        return np.array([0.0, -1.0], dtype=np.float64)
    raise ValueError(f"Unsupported direction: {direction}")


def load_train_stems(scene_dir: Path) -> list[str]:
    path = scene_dir / "images_train.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing train image list: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scale_intrinsics_to_fullres(K: np.ndarray, image_width: int, base_width: float) -> np.ndarray:
    scale = float(image_width) / float(base_width)
    K_full = np.asarray(K, dtype=np.float64).copy()
    K_full[:2, :] *= scale
    K_full[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return K_full


def build_ground_segments_3d() -> tuple[np.ndarray, list[tuple[int, int]]]:
    keypoints = court_keypoints_3d().cpu().numpy().astype(np.float64)
    segments: list[tuple[int, int]] = []
    for i0, i1 in COURT_SKELETON:
        if abs(float(keypoints[i0, 2])) > 1.0e-9 or abs(float(keypoints[i1, 2])) > 1.0e-9:
            continue
        segments.append((int(i0), int(i1)))
    return keypoints, segments


def build_two_court_keypoints_local(adjacent_direction: str, adjacent_gap: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    keypoints, segments = build_ground_segments_3d()
    shift_xy = adjacent_direction_vector(adjacent_direction) * (DOUBLES_WIDTH + adjacent_gap)
    shift = np.array([shift_xy[0], shift_xy[1], 0.0], dtype=np.float64)
    second = keypoints + shift[None, :]
    combined = np.concatenate((keypoints, second), axis=0)
    offset = keypoints.shape[0]
    combined_segments = segments + [(i0 + offset, i1 + offset) for i0, i1 in segments]
    return combined, combined_segments


def sample_segment_points(p0: np.ndarray, p1: np.ndarray, step_m: float) -> np.ndarray:
    length = float(np.linalg.norm(p1 - p0))
    num = max(2, int(math.ceil(length / max(step_m, 1.0e-6))) + 1)
    alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
    return (1.0 - alpha[:, None]) * p0[None, :] + alpha[:, None] * p1[None, :]


def build_fixed_court_world(
    sim3_payload: dict[str, Any],
    step_m: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    local_keypoints, segments = build_two_court_keypoints_local(
        str(sim3_payload["adjacent_direction"]),
        float(sim3_payload["adjacent_gap"]),
    )
    rotation = np.asarray(sim3_payload["rotation"], dtype=np.float64)
    translation = np.asarray(sim3_payload["translation"], dtype=np.float64)
    scale = float(sim3_payload["scale"])
    world_keypoints = scale * (local_keypoints @ rotation.T) + translation[None, :]
    samples = [sample_segment_points(world_keypoints[i0], world_keypoints[i1], step_m) for i0, i1 in segments]
    world_points = np.concatenate(samples, axis=0) if samples else np.empty((0, 3), dtype=np.float64)
    return world_points, world_keypoints, segments


def load_plane_frame(ground_dir: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    plane_frame_path = ground_dir / "plane_frame.json"
    if not plane_frame_path.exists():
        raise FileNotFoundError(f"Missing plane frame: {plane_frame_path}")
    payload = json.loads(plane_frame_path.read_text(encoding="utf-8"))
    plane = payload["plane"]
    return (
        np.asarray(plane["normal"], dtype=np.float64),
        float(plane["d"]),
        np.asarray(plane["origin"], dtype=np.float64),
        np.asarray(plane["axis_x"], dtype=np.float64),
        np.asarray(plane["axis_y"], dtype=np.float64),
    )


def load_raster_grid(ground_dir: Path) -> tuple[tuple[float, float, float, float], float, tuple[int, int]]:
    raster_grid_path = ground_dir / "raster_grid.json"
    if not raster_grid_path.exists():
        raise FileNotFoundError(f"Missing raster grid: {raster_grid_path}")
    payload = json.loads(raster_grid_path.read_text(encoding="utf-8"))
    extent = tuple(float(v) for v in payload["extent_xy"])
    resolution = float(payload["resolution"])
    shape = (int(payload["shape_hw"][0]), int(payload["shape_hw"][1]))
    return extent, resolution, shape


def world_to_plane_uv(world_points: np.ndarray, origin: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    rel = world_points - origin[None, :]
    return np.column_stack((rel @ axis_x, rel @ axis_y)).astype(np.float64)


def plane_to_grid(uv: np.ndarray, extent: tuple[float, float, float, float], resolution: float) -> np.ndarray:
    x_min, _, _, y_max = extent
    cols = (uv[:, 0] - x_min) / resolution
    rows = (y_max - uv[:, 1]) / resolution
    return np.column_stack((cols, rows)).astype(np.float32)


def render_court_plane_mask(
    world_keypoints: np.ndarray,
    segments: list[tuple[int, int]],
    extent: tuple[float, float, float, float],
    resolution: float,
    shape_hw: tuple[int, int],
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    thickness_px: int,
) -> np.ndarray:
    uv = world_to_plane_uv(world_keypoints, origin, axis_x, axis_y)
    grid = plane_to_grid(uv, extent, resolution)
    image = np.zeros(shape_hw, dtype=np.uint8)
    for i0, i1 in segments:
        p0 = grid[i0]
        p1 = grid[i1]
        cv2.line(
            image,
            (int(round(float(p0[0]))), int(round(float(p0[1])))),
            (int(round(float(p1[0]))), int(round(float(p1[1])))),
            255,
            thickness=max(1, thickness_px),
            lineType=cv2.LINE_AA,
        )
    return image


def load_cluster_split(cluster_source_path: Path, n_cams: int) -> tuple[np.ndarray, str]:
    if not cluster_source_path.exists():
        raise FileNotFoundError(f"Missing cluster source: {cluster_source_path}")
    payload = json.loads(cluster_source_path.read_text(encoding="utf-8"))
    if "dominant_cameras" not in payload:
        raise ValueError(f"{cluster_source_path} does not contain dominant_cameras")
    cluster_indices = sorted(int(idx) for idx in payload["dominant_cameras"])
    cluster_mask = np.zeros((n_cams,), dtype=bool)
    cluster_mask[np.asarray(cluster_indices, dtype=np.int64)] = True
    return cluster_mask, str(cluster_source_path)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def se3_exp(rotvec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    batch_shape = rotvec.shape[:-1]
    omega = rotvec.reshape(-1, 3)
    trans_flat = trans.reshape(-1, 3)
    zeros = torch.zeros((omega.shape[0],), dtype=omega.dtype, device=omega.device)
    skew = torch.stack(
        (
            zeros,
            -omega[:, 2],
            omega[:, 1],
            omega[:, 2],
            zeros,
            -omega[:, 0],
            -omega[:, 1],
            omega[:, 0],
            zeros,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation = torch.matrix_exp(skew)
    transform = torch.eye(4, dtype=omega.dtype, device=omega.device).unsqueeze(0).repeat(omega.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = trans_flat
    return transform.reshape(*batch_shape, 4, 4)


def compose_c2w(
    base_poses: torch.Tensor,
    local_rot: torch.Tensor,
    local_trans: torch.Tensor,
    other_rot: torch.Tensor,
    other_trans: torch.Tensor,
    other_mask: torch.Tensor,
) -> torch.Tensor:
    local_tf = se3_exp(local_rot, local_trans)
    composed = torch.matmul(base_poses, local_tf)
    if bool(other_mask.any()):
        other_tf = se3_exp(other_rot.unsqueeze(0), other_trans.unsqueeze(0))[0]
        composed_other = torch.matmul(other_tf.unsqueeze(0), composed[other_mask])
        composed = composed.clone()
        composed[other_mask] = composed_other
    return composed


def sample_dt_image(
    dt_images: torch.Tensor,
    grid_xy: torch.Tensor,
) -> torch.Tensor:
    sampled = F.grid_sample(
        dt_images,
        grid_xy.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0, :, 0]


def project_court_to_image_loss(
    poses: torch.Tensor,
    K_down: torch.Tensor,
    court_world_points: torch.Tensor,
    dt_images: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_hw
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    world = court_world_points.unsqueeze(0).expand(poses.shape[0], -1, -1)
    cam = torch.matmul(world - t[:, None, :], R)
    z = cam[..., 2]
    fx = K_down[:, 0, 0].unsqueeze(1)
    fy = K_down[:, 1, 1].unsqueeze(1)
    cx = K_down[:, 0, 2].unsqueeze(1)
    cy = K_down[:, 1, 2].unsqueeze(1)
    eps = 1.0e-6
    u = fx * (cam[..., 0] / torch.clamp(z, min=eps)) + cx
    v = fy * (cam[..., 1] / torch.clamp(z, min=eps)) + cy
    valid = (z > eps) & (u >= 0.0) & (u <= float(width - 1)) & (v >= 0.0) & (v <= float(height - 1))
    x_norm = 2.0 * (u / max(width - 1, 1)) - 1.0
    y_norm = 2.0 * (v / max(height - 1, 1)) - 1.0
    sampled = sample_dt_image(dt_images, torch.stack((x_norm, y_norm), dim=-1))
    valid_f = valid.float()
    denom = valid_f.sum(dim=1) + 1.0e-8
    per_camera = (sampled * valid_f).sum(dim=1) / denom
    used = valid.sum(dim=1) > 0
    return per_camera.mean(), per_camera, used


def project_mask_to_plane_loss(
    poses: torch.Tensor,
    K_down: torch.Tensor,
    sampled_pixels: torch.Tensor,
    sampled_valid: torch.Tensor,
    plane_dt: torch.Tensor,
    plane_extent: tuple[float, float, float, float],
    plane_resolution: float,
    plane_origin: torch.Tensor,
    plane_axis_x: torch.Tensor,
    plane_axis_y: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height = plane_dt.shape[-2]
    width = plane_dt.shape[-1]
    pixels_h = torch.cat((sampled_pixels, torch.ones_like(sampled_pixels[..., :1])), dim=-1)
    K_inv = torch.linalg.inv(K_down)
    dirs_cam = torch.matmul(pixels_h, K_inv.transpose(1, 2))
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    dirs_world = torch.matmul(dirs_cam, R.transpose(1, 2))
    numer = plane_d - torch.sum(t * plane_normal.unsqueeze(0), dim=1, keepdim=True)
    denom = torch.sum(dirs_world * plane_normal.view(1, 1, 3), dim=-1)
    eps = 1.0e-6
    ray_t = numer / torch.where(torch.abs(denom) > eps, denom, torch.full_like(denom, eps))
    valid = sampled_valid & (torch.abs(denom) > eps) & (ray_t > 0.0)
    world = t[:, None, :] + ray_t[..., None] * dirs_world
    rel = world - plane_origin.view(1, 1, 3)
    u = torch.sum(rel * plane_axis_x.view(1, 1, 3), dim=-1)
    v = torch.sum(rel * plane_axis_y.view(1, 1, 3), dim=-1)
    x_min, _, _, y_max = plane_extent
    cols = (u - float(x_min)) / plane_resolution
    rows = (float(y_max) - v) / plane_resolution
    inside = (cols >= 0.0) & (cols <= float(width - 1)) & (rows >= 0.0) & (rows <= float(height - 1))
    valid = valid & inside
    x_norm = 2.0 * (cols / max(width - 1, 1)) - 1.0
    y_norm = 2.0 * (rows / max(height - 1, 1)) - 1.0
    plane_dt_batch = plane_dt.expand(poses.shape[0], -1, -1, -1)
    sampled = sample_dt_image(plane_dt_batch, torch.stack((x_norm, y_norm), dim=-1))
    valid_f = valid.float()
    denom = valid_f.sum(dim=1) + 1.0e-8
    per_camera = (sampled * valid_f).sum(dim=1) / denom
    used = valid.sum(dim=1) > 0
    return per_camera.mean(), per_camera, used


def evaluate_losses(
    poses: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    court_to_cam_mean, court_to_cam_per_camera, court_to_cam_used = project_court_to_image_loss(
        poses,
        tensors["K_down"],
        tensors["court_world_points"],
        tensors["image_dt"],
        tensors["image_hw"],
    )
    cam_to_court_mean, cam_to_court_per_camera, cam_to_court_used = project_mask_to_plane_loss(
        poses,
        tensors["K_down"],
        tensors["sample_pixels"],
        tensors["sample_valid"],
        tensors["plane_dt"],
        tensors["plane_extent"],
        tensors["plane_resolution"],
        tensors["plane_origin"],
        tensors["plane_axis_x"],
        tensors["plane_axis_y"],
        tensors["plane_normal"],
        tensors["plane_d"],
    )
    return {
        "court_to_cam_mean": court_to_cam_mean,
        "cam_to_court_mean": cam_to_court_mean,
        "court_to_cam_per_camera": court_to_cam_per_camera,
        "cam_to_court_per_camera": cam_to_court_per_camera,
        "court_to_cam_used": court_to_cam_used,
        "cam_to_court_used": cam_to_court_used,
        "total_per_camera": args.lambda_court_to_cam * court_to_cam_per_camera
        + args.lambda_cam_to_court * cam_to_court_per_camera,
    }


def save_loss_curve(trace: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(trace["iter"], trace["total"], label="total")
    plt.plot(trace["iter"], trace["court_to_cam"], label="court_to_cam")
    plt.plot(trace["iter"], trace["cam_to_court"], label="cam_to_court")
    plt.plot(trace["iter"], trace["reg_other"], label="reg_other")
    plt.plot(trace["iter"], trace["reg_local"], label="reg_local")
    plt.plot(trace["iter"], trace["reg_cluster"], label="reg_cluster")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def project_points_to_image_np(
    c2w: np.ndarray,
    K: np.ndarray,
    world_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    cam = (world_points - translation[None, :]) @ rotation
    z = cam[:, 2]
    valid = z > 1.0e-6
    uv = np.zeros((world_points.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = K[0, 0] * (cam[valid, 0] / z[valid]) + K[0, 2]
    uv[valid, 1] = K[1, 1] * (cam[valid, 1] / z[valid]) + K[1, 2]
    return uv, valid


def render_image_overlay(
    mask_down: np.ndarray,
    court_keypoints_world: np.ndarray,
    segments: list[tuple[int, int]],
    K_down: np.ndarray,
    pose_before: np.ndarray,
    pose_after: np.ndarray,
) -> np.ndarray:
    base = np.repeat((mask_down > 0).astype(np.uint8)[..., None] * 80, 3, axis=2)
    before_uv, before_valid = project_points_to_image_np(pose_before, K_down, court_keypoints_world)
    after_uv, after_valid = project_points_to_image_np(pose_after, K_down, court_keypoints_world)
    overlay = base.copy()
    for i0, i1 in segments:
        if before_valid[i0] and before_valid[i1]:
            cv2.line(
                overlay,
                (int(round(float(before_uv[i0, 0]))), int(round(float(before_uv[i0, 1])))),
                (int(round(float(before_uv[i1, 0]))), int(round(float(before_uv[i1, 1])))),
                (32, 64, 255),
                1,
                cv2.LINE_AA,
            )
        if after_valid[i0] and after_valid[i1]:
            cv2.line(
                overlay,
                (int(round(float(after_uv[i0, 0]))), int(round(float(after_uv[i0, 1])))),
                (int(round(float(after_uv[i1, 0]))), int(round(float(after_uv[i1, 1])))),
                (32, 255, 64),
                1,
                cv2.LINE_AA,
            )
    return overlay


def project_pixels_to_plane_np(
    c2w: np.ndarray,
    K: np.ndarray,
    sampled_pixels: np.ndarray,
    valid_mask: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> np.ndarray:
    pixels = sampled_pixels[valid_mask]
    if pixels.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pixels_h = np.column_stack((pixels, np.ones((pixels.shape[0],), dtype=np.float32)))
    K_inv = np.linalg.inv(K)
    dirs_cam = pixels_h @ K_inv.T
    dirs_world = dirs_cam @ c2w[:3, :3].T
    cam_center = c2w[:3, 3]
    denom = dirs_world @ plane_normal
    numer = plane_d - float(np.dot(cam_center, plane_normal))
    valid = np.abs(denom) > 1.0e-9
    ray_t = np.full((dirs_world.shape[0],), np.nan, dtype=np.float64)
    ray_t[valid] = numer / denom[valid]
    valid &= ray_t > 0.0
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    world = cam_center[None, :] + ray_t[valid, None] * dirs_world[valid]
    rel = world - origin[None, :]
    uv = np.column_stack((rel @ axis_x, rel @ axis_y))
    return uv.astype(np.float32)


def render_plane_overlay(
    plane_mask: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float,
    before_uv: np.ndarray,
    after_uv: np.ndarray,
) -> np.ndarray:
    base = np.repeat((plane_mask > 0).astype(np.uint8)[..., None] * 80, 3, axis=2)
    overlay = base.copy()
    x_min, _, _, y_max = extent
    for uv, color in ((before_uv, (32, 64, 255)), (after_uv, (32, 255, 64))):
        if uv.size == 0:
            continue
        cols = np.round((uv[:, 0] - x_min) / resolution).astype(np.int32)
        rows = np.round((y_max - uv[:, 1]) / resolution).astype(np.int32)
        valid = (cols >= 0) & (cols < overlay.shape[1]) & (rows >= 0) & (rows < overlay.shape[0])
        overlay[rows[valid], cols[valid]] = color
    return overlay


def save_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 256, cols: int = 4) -> None:
    if not image_paths:
        return
    rows = int(math.ceil(len(image_paths) / cols))
    canvas = np.full((rows * thumb_size, cols * thumb_size, 3), 20, dtype=np.uint8)
    for idx, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_size
        x0 = col * thumb_size
        canvas[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = thumb
        cv2.putText(
            canvas,
            path.stem[:28],
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)


def prepare_data(args: argparse.Namespace) -> tuple[PrecomputedData, list[tuple[int, int]]]:
    scene_dir = args.scene_dir.expanduser().resolve()
    mast3r_dir = args.mast3r_dir.expanduser().resolve() if args.mast3r_dir is not None else scene_dir / "mast3r"
    ground_dir = args.ground_dir.expanduser().resolve() if args.ground_dir is not None else scene_dir / "court" / "ground"
    transform_dir = args.transform_dir.expanduser().resolve() if args.transform_dir is not None else scene_dir / "court" / "transform"
    fixed_sim3_path = args.fixed_sim3_path.expanduser().resolve() if args.fixed_sim3_path is not None else transform_dir / "ground_heatmap_fit_sim3.json"
    cluster_source_path = args.cluster_source_path.expanduser().resolve() if args.cluster_source_path is not None else transform_dir / "ground_heatmap_fit.json"
    mask_path = args.mask_path.expanduser().resolve() if args.mask_path is not None else scene_dir / "court" / "line" / "court_line_masks.npy"

    train_stems = load_train_stems(scene_dir)
    masks = np.load(mask_path, mmap_mode="r")
    intrinsics = np.load(mast3r_dir / "camera_intrinsics.npy").astype(np.float64)
    poses = np.load(mast3r_dir / "camera_poses.npy").astype(np.float64)
    if masks.shape[0] != poses.shape[0] or intrinsics.shape[0] != poses.shape[0]:
        raise ValueError("Camera counts do not match between masks, intrinsics, and poses.")
    if len(train_stems) != poses.shape[0]:
        raise ValueError("images_train.txt count does not match camera count.")

    cluster_mask, _ = load_cluster_split(cluster_source_path, poses.shape[0])
    other_mask = ~cluster_mask

    fixed_sim3 = json.loads(fixed_sim3_path.read_text(encoding="utf-8"))
    fixed_court_world_points, fixed_court_world_keypoints, fixed_segments = build_fixed_court_world(
        fixed_sim3,
        args.court_line_step_m,
    )
    plane_normal, plane_d, plane_origin, plane_axis_x, plane_axis_y = load_plane_frame(ground_dir)
    plane_extent, plane_resolution, plane_shape_hw = load_raster_grid(ground_dir)
    fixed_plane_mask = render_court_plane_mask(
        fixed_court_world_keypoints,
        fixed_segments,
        plane_extent,
        plane_resolution,
        plane_shape_hw,
        plane_origin,
        plane_axis_x,
        plane_axis_y,
        args.court_line_thickness_px,
    )
    plane_dt_m = cv2.distanceTransform((fixed_plane_mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    plane_dt_m *= float(plane_resolution)

    full_height, full_width = int(masks.shape[1]), int(masks.shape[2])
    image_downsample = max(1, int(args.image_downsample))
    down_width = int(round(full_width / image_downsample))
    down_height = int(round(full_height / image_downsample))
    rng = np.random.default_rng(args.seed)

    masks_down = np.zeros((poses.shape[0], down_height, down_width), dtype=np.uint8)
    image_dt = np.zeros((poses.shape[0], down_height, down_width), dtype=np.float32)
    sample_pixels = np.zeros((poses.shape[0], args.cam_samples_per_camera, 2), dtype=np.float32)
    sample_valid = np.zeros((poses.shape[0], args.cam_samples_per_camera), dtype=bool)
    K_down = np.zeros((poses.shape[0], 3, 3), dtype=np.float32)

    for cam_idx in range(poses.shape[0]):
        mask_down = cv2.resize(
            np.asarray(masks[cam_idx], dtype=np.uint8),
            (down_width, down_height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_down = (mask_down > 0).astype(np.uint8)
        masks_down[cam_idx] = mask_down
        dt = cv2.distanceTransform((mask_down == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        image_dt[cam_idx] = dt / float(max(down_height, down_width, 1))
        ys, xs = np.nonzero(mask_down > 0)
        coords = np.column_stack((xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5)).astype(np.float32)
        if coords.shape[0] > args.cam_samples_per_camera:
            chosen = rng.choice(coords.shape[0], size=args.cam_samples_per_camera, replace=False)
            coords = coords[chosen]
        count = coords.shape[0]
        sample_pixels[cam_idx, :count] = coords
        sample_valid[cam_idx, :count] = True
        k_full = scale_intrinsics_to_fullres(intrinsics[cam_idx], full_width, args.court_base_width)
        k_down = k_full.copy()
        k_down[:2, :] /= float(image_downsample)
        k_down[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        K_down[cam_idx] = k_down.astype(np.float32)

    precomputed = PrecomputedData(
        train_stems=train_stems,
        masks_down=masks_down,
        image_dt=image_dt,
        sample_pixels=sample_pixels,
        sample_valid=sample_valid,
        K_down=K_down,
        base_poses=poses.astype(np.float32),
        cluster_mask=cluster_mask,
        other_mask=other_mask,
        fixed_court_world_points=fixed_court_world_points.astype(np.float32),
        fixed_court_world_keypoints=fixed_court_world_keypoints.astype(np.float32),
        fixed_court_plane_mask=fixed_plane_mask,
        plane_dt_m=plane_dt_m,
        plane_extent=plane_extent,
        plane_resolution=plane_resolution,
        plane_origin=plane_origin.astype(np.float32),
        plane_axis_x=plane_axis_x.astype(np.float32),
        plane_axis_y=plane_axis_y.astype(np.float32),
        plane_normal=plane_normal.astype(np.float32),
        plane_d=float(plane_d),
    )
    return precomputed, fixed_segments


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_dir = args.scene_dir.expanduser().resolve()
    transform_dir = args.transform_dir.expanduser().resolve() if args.transform_dir is not None else scene_dir / "court" / "transform"
    fixed_sim3_path = args.fixed_sim3_path.expanduser().resolve() if args.fixed_sim3_path is not None else transform_dir / "ground_heatmap_fit_sim3.json"
    cluster_source_path = args.cluster_source_path.expanduser().resolve() if args.cluster_source_path is not None else transform_dir / "ground_heatmap_fit.json"
    pose_opt_dir = args.pose_opt_dir.expanduser().resolve() if args.pose_opt_dir is not None else scene_dir / "court" / "pose_opt"
    output_dir = args.output_dir.expanduser().resolve()
    image_overlay_dir = output_dir / "per_camera_image_overlays"
    plane_overlay_dir = output_dir / "per_camera_plane_overlays"
    pose_opt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_overlay_dir.mkdir(parents=True, exist_ok=True)
    plane_overlay_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing fixed-court pose optimization inputs...", flush=True)
    precomputed, fixed_segments = prepare_data(args)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    tensors: dict[str, torch.Tensor | tuple[int, int] | tuple[float, float, float, float] | float] = {
        "K_down": torch.from_numpy(precomputed.K_down).to(device=device, dtype=torch.float32),
        "court_world_points": torch.from_numpy(precomputed.fixed_court_world_points).to(device=device, dtype=torch.float32),
        "image_dt": torch.from_numpy(precomputed.image_dt[:, None, :, :]).to(device=device, dtype=torch.float32),
        "sample_pixels": torch.from_numpy(precomputed.sample_pixels).to(device=device, dtype=torch.float32),
        "sample_valid": torch.from_numpy(precomputed.sample_valid).to(device=device),
        "plane_dt": torch.from_numpy(precomputed.plane_dt_m[None, None, :, :]).to(device=device, dtype=torch.float32),
        "plane_origin": torch.from_numpy(precomputed.plane_origin).to(device=device, dtype=torch.float32),
        "plane_axis_x": torch.from_numpy(precomputed.plane_axis_x).to(device=device, dtype=torch.float32),
        "plane_axis_y": torch.from_numpy(precomputed.plane_axis_y).to(device=device, dtype=torch.float32),
        "plane_normal": torch.from_numpy(precomputed.plane_normal).to(device=device, dtype=torch.float32),
        "plane_d": torch.tensor(precomputed.plane_d, device=device, dtype=torch.float32),
        "image_hw": tuple(precomputed.masks_down.shape[1:]),
        "plane_extent": precomputed.plane_extent,
        "plane_resolution": precomputed.plane_resolution,
    }
    base_poses = torch.from_numpy(precomputed.base_poses).to(device=device, dtype=torch.float32)
    cluster_mask = torch.from_numpy(precomputed.cluster_mask).to(device=device)
    other_mask = torch.from_numpy(precomputed.other_mask).to(device=device)

    other_rot = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
    other_trans = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
    local_rot = torch.nn.Parameter(torch.zeros((precomputed.base_poses.shape[0], 3), device=device, dtype=torch.float32))
    local_trans = torch.nn.Parameter(torch.zeros((precomputed.base_poses.shape[0], 3), device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [
            {"params": [other_rot, other_trans], "lr": args.lr_other},
            {"params": [local_rot, local_trans], "lr": args.lr_local},
        ]
    )

    trace: dict[str, list[float]] = {
        "iter": [],
        "total": [],
        "court_to_cam": [],
        "cam_to_court": [],
        "reg_other": [],
        "reg_local": [],
        "reg_cluster": [],
    }

    with torch.no_grad():
        poses_initial = compose_c2w(base_poses, local_rot, local_trans, other_rot, other_trans, other_mask)
        initial_loss = evaluate_losses(poses_initial, tensors, args)

    print("Optimizing camera poses against fixed Sim(3)...", flush=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_total = float("inf")
    for step in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        poses_current = compose_c2w(base_poses, local_rot, local_trans, other_rot, other_trans, other_mask)
        losses = evaluate_losses(poses_current, tensors, args)
        reg_other = other_rot.pow(2).mean() + other_trans.pow(2).mean()
        reg_local = local_rot.pow(2).mean() + local_trans.pow(2).mean()
        if bool(cluster_mask.any()):
            reg_cluster = local_rot[cluster_mask].pow(2).mean() + local_trans[cluster_mask].pow(2).mean()
        else:
            reg_cluster = torch.zeros((), device=device, dtype=torch.float32)
        total = (
            args.lambda_court_to_cam * losses["court_to_cam_mean"]
            + args.lambda_cam_to_court * losses["cam_to_court_mean"]
            + args.lambda_reg_other * reg_other
            + args.lambda_reg_local * reg_local
            + args.lambda_reg_cluster * reg_cluster
        )
        total.backward()
        optimizer.step()

        total_value = float(total.detach().cpu())
        if total_value < best_total:
            best_total = total_value
            best_state = {
                "other_rot": other_rot.detach().clone(),
                "other_trans": other_trans.detach().clone(),
                "local_rot": local_rot.detach().clone(),
                "local_trans": local_trans.detach().clone(),
            }
        trace["iter"].append(step)
        trace["total"].append(total_value)
        trace["court_to_cam"].append(float(losses["court_to_cam_mean"].detach().cpu()))
        trace["cam_to_court"].append(float(losses["cam_to_court_mean"].detach().cpu()))
        trace["reg_other"].append(float(reg_other.detach().cpu()))
        trace["reg_local"].append(float(reg_local.detach().cpu()))
        trace["reg_cluster"].append(float(reg_cluster.detach().cpu()))
        if step % max(1, args.log_every) == 0 or step + 1 == args.iters:
            print(
                f"[iter {step:04d}] total={total_value:.6f} "
                f"court->cam={trace['court_to_cam'][-1]:.6f} "
                f"cam->court={trace['cam_to_court'][-1]:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Optimization did not produce a best state.")

    with torch.no_grad():
        other_rot.copy_(best_state["other_rot"])
        other_trans.copy_(best_state["other_trans"])
        local_rot.copy_(best_state["local_rot"])
        local_trans.copy_(best_state["local_trans"])
        poses_final = compose_c2w(base_poses, local_rot, local_trans, other_rot, other_trans, other_mask)
        final_loss = evaluate_losses(poses_final, tensors, args)

    optimized_poses = poses_final.detach().cpu().numpy().astype(np.float32)
    initial_poses = poses_initial.detach().cpu().numpy().astype(np.float32)
    other_matrix = se3_exp(other_rot.detach().unsqueeze(0), other_trans.detach().unsqueeze(0))[0].detach().cpu().numpy().astype(np.float32)
    local_matrices = se3_exp(local_rot.detach(), local_trans.detach()).detach().cpu().numpy().astype(np.float32)

    optimized_pose_path = pose_opt_dir / "optimized_camera_poses.npy"
    np.save(optimized_pose_path, optimized_poses)
    delta_path = pose_opt_dir / "camera_pose_opt_deltas.npz"
    np.savez_compressed(
        delta_path,
        train_stems=np.asarray(precomputed.train_stems, dtype=str),
        split_labels=np.where(precomputed.other_mask, 1, 0).astype(np.int32),
        c2w_init=precomputed.base_poses.astype(np.float32),
        c2w_opt=optimized_poses,
        delta_other_matrix=other_matrix,
        delta_other_se3=np.concatenate(
            (other_rot.detach().cpu().numpy().astype(np.float32), other_trans.detach().cpu().numpy().astype(np.float32))
        ),
        delta_local_matrix=local_matrices,
        delta_local_se3=np.concatenate(
            (
                local_rot.detach().cpu().numpy().astype(np.float32),
                local_trans.detach().cpu().numpy().astype(np.float32),
            ),
            axis=1,
        ),
        loss_before=initial_loss["total_per_camera"].detach().cpu().numpy().astype(np.float32),
        loss_after=final_loss["total_per_camera"].detach().cpu().numpy().astype(np.float32),
    )

    cluster_indices = np.flatnonzero(precomputed.cluster_mask).tolist()
    other_indices = np.flatnonzero(precomputed.other_mask).tolist()
    per_camera_records: list[dict[str, Any]] = []
    delta_rot_norm = np.linalg.norm(local_rot.detach().cpu().numpy(), axis=1)
    delta_trans_norm = np.linalg.norm(local_trans.detach().cpu().numpy(), axis=1)
    for idx, stem in enumerate(precomputed.train_stems):
        per_camera_records.append(
            {
                "index": int(idx),
                "stem": stem,
                "split": "other" if bool(precomputed.other_mask[idx]) else "cluster",
                "used_in_optimization": bool(np.count_nonzero(precomputed.masks_down[idx]) > 0),
                "has_valid_mask": bool(np.count_nonzero(precomputed.masks_down[idx]) > 0),
                "loss_before": float(initial_loss["total_per_camera"][idx].detach().cpu()),
                "loss_after": float(final_loss["total_per_camera"][idx].detach().cpu()),
                "court_to_cam_before": float(initial_loss["court_to_cam_per_camera"][idx].detach().cpu()),
                "court_to_cam_after": float(final_loss["court_to_cam_per_camera"][idx].detach().cpu()),
                "cam_to_court_before": float(initial_loss["cam_to_court_per_camera"][idx].detach().cpu()),
                "cam_to_court_after": float(final_loss["cam_to_court_per_camera"][idx].detach().cpu()),
                "delta_rotation_deg": float(np.rad2deg(delta_rot_norm[idx])),
                "delta_translation_m": float(delta_trans_norm[idx]),
            }
        )

    summary = {
        "scene_dir": str(scene_dir),
        "mast3r_dir": str((scene_dir / "mast3r").resolve()),
        "fixed_sim3_path": str(fixed_sim3_path),
        "cluster_source_path": str(cluster_source_path),
        "input_camera_poses_path": str((scene_dir / "mast3r" / "camera_poses.npy").resolve()),
        "optimized_camera_poses_path": str(optimized_pose_path),
        "cluster_cameras": cluster_indices,
        "other_cameras": other_indices,
        "parameterization": {
            "other_correction": "SE3_left",
            "per_camera_correction": "SE3_right",
        },
        "loss_definition": {
            "court_to_cam": "uniform_image_distance_transform",
            "cam_to_court": "uniform_plane_distance_transform",
            "regularizers": {
                "lambda_reg_other": float(args.lambda_reg_other),
                "lambda_reg_local": float(args.lambda_reg_local),
                "lambda_reg_cluster": float(args.lambda_reg_cluster),
            },
        },
        "loss_initial": {
            "total": float(
                args.lambda_court_to_cam * initial_loss["court_to_cam_mean"].detach().cpu()
                + args.lambda_cam_to_court * initial_loss["cam_to_court_mean"].detach().cpu()
            ),
            "court_to_cam": float(initial_loss["court_to_cam_mean"].detach().cpu()),
            "cam_to_court": float(initial_loss["cam_to_court_mean"].detach().cpu()),
            "reg_other": 0.0,
            "reg_local": 0.0,
            "reg_cluster": 0.0,
        },
        "loss_final": {
            "total": float(best_total),
            "court_to_cam": float(final_loss["court_to_cam_mean"].detach().cpu()),
            "cam_to_court": float(final_loss["cam_to_court_mean"].detach().cpu()),
            "reg_other": float((other_rot.pow(2).mean() + other_trans.pow(2).mean()).detach().cpu()),
            "reg_local": float((local_rot.pow(2).mean() + local_trans.pow(2).mean()).detach().cpu()),
            "reg_cluster": float(
                (
                    local_rot[cluster_mask].pow(2).mean() + local_trans[cluster_mask].pow(2).mean()
                    if bool(cluster_mask.any())
                    else torch.zeros((), device=device)
                ).detach().cpu()
            ),
        },
        "pose_delta_summary": {
            "other_rotation_deg": float(np.rad2deg(np.linalg.norm(other_rot.detach().cpu().numpy()))),
            "other_translation_m": float(np.linalg.norm(other_trans.detach().cpu().numpy())),
            "per_camera_rotation_deg_mean": float(np.rad2deg(delta_rot_norm.mean())),
            "per_camera_translation_m_mean": float(delta_trans_norm.mean()),
        },
    }
    summary_path = pose_opt_dir / "camera_pose_opt.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_camera_payload = {
        "scene_dir": str(scene_dir),
        "fixed_sim3_path": str(fixed_sim3_path),
        "optimized_camera_poses_path": str(optimized_pose_path),
        "records": per_camera_records,
    }
    per_camera_path = pose_opt_dir / "camera_pose_opt_per_camera.json"
    per_camera_path.write_text(json.dumps(per_camera_payload, indent=2), encoding="utf-8")

    manifest = {
        "scene_dir": str(scene_dir),
        "pose_opt_dir": str(pose_opt_dir),
        "debug_output_dir": str(output_dir),
        "fixed_sim3_path": str(fixed_sim3_path),
        "cluster_source_path": str(cluster_source_path),
        "optimized_camera_poses_path": str(optimized_pose_path),
        "summary_path": str(summary_path),
        "per_camera_path": str(per_camera_path),
        "delta_path": str(delta_path),
    }
    manifest_path = pose_opt_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_summary_path = output_dir / "split_summary.json"
    split_summary_path.write_text(
        json.dumps(
            {
                "cluster_indices": cluster_indices,
                "other_indices": other_indices,
                "cluster_stems": [precomputed.train_stems[idx] for idx in cluster_indices],
                "other_stems": [precomputed.train_stems[idx] for idx in other_indices],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "scene_dir": str(scene_dir),
        "fixed_sim3_path": str(fixed_sim3_path),
        "cluster_source_path": str(cluster_source_path),
        "pose_opt_dir": str(pose_opt_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "image_downsample": int(args.image_downsample),
        "cam_samples_per_camera": int(args.cam_samples_per_camera),
        "iters": int(args.iters),
        "learning_rates": {"other": float(args.lr_other), "local": float(args.lr_local)},
        "loss_weights": {
            "lambda_court_to_cam": float(args.lambda_court_to_cam),
            "lambda_cam_to_court": float(args.lambda_cam_to_court),
            "lambda_reg_other": float(args.lambda_reg_other),
            "lambda_reg_local": float(args.lambda_reg_local),
            "lambda_reg_cluster": float(args.lambda_reg_cluster),
        },
        "loss_initial": summary["loss_initial"],
        "loss_final": summary["loss_final"],
        "pose_delta_summary": summary["pose_delta_summary"],
        "optimized_camera_poses_path": str(optimized_pose_path),
        "train_stems": precomputed.train_stems,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    trace_path = output_dir / "optimization_trace.npz"
    np.savez_compressed(
        trace_path,
        iter=np.asarray(trace["iter"], dtype=np.int32),
        total=np.asarray(trace["total"], dtype=np.float32),
        court_to_cam=np.asarray(trace["court_to_cam"], dtype=np.float32),
        cam_to_court=np.asarray(trace["cam_to_court"], dtype=np.float32),
        reg_other=np.asarray(trace["reg_other"], dtype=np.float32),
        reg_local=np.asarray(trace["reg_local"], dtype=np.float32),
        reg_cluster=np.asarray(trace["reg_cluster"], dtype=np.float32),
    )
    save_loss_curve(trace, output_dir / "loss_curve.png")

    image_overlay_paths: list[Path] = []
    plane_overlay_paths: list[Path] = []
    for idx, stem in enumerate(precomputed.train_stems):
        image_overlay = render_image_overlay(
            precomputed.masks_down[idx],
            precomputed.fixed_court_world_keypoints,
            fixed_segments,
            precomputed.K_down[idx],
            initial_poses[idx],
            optimized_poses[idx],
        )
        image_path = image_overlay_dir / f"{idx:03d}_{stem}.png"
        cv2.imwrite(str(image_path), image_overlay)
        image_overlay_paths.append(image_path)

        before_uv = project_pixels_to_plane_np(
            initial_poses[idx],
            precomputed.K_down[idx],
            precomputed.sample_pixels[idx],
            precomputed.sample_valid[idx],
            precomputed.plane_normal,
            precomputed.plane_d,
            precomputed.plane_origin,
            precomputed.plane_axis_x,
            precomputed.plane_axis_y,
        )
        after_uv = project_pixels_to_plane_np(
            optimized_poses[idx],
            precomputed.K_down[idx],
            precomputed.sample_pixels[idx],
            precomputed.sample_valid[idx],
            precomputed.plane_normal,
            precomputed.plane_d,
            precomputed.plane_origin,
            precomputed.plane_axis_x,
            precomputed.plane_axis_y,
        )
        plane_overlay = render_plane_overlay(
            precomputed.fixed_court_plane_mask,
            precomputed.plane_extent,
            precomputed.plane_resolution,
            before_uv,
            after_uv,
        )
        plane_path = plane_overlay_dir / f"{idx:03d}_{stem}.png"
        cv2.imwrite(str(plane_path), plane_overlay)
        plane_overlay_paths.append(plane_path)

    save_contact_sheet(image_overlay_paths, output_dir / "contact_sheet_image_overlays.png")
    save_contact_sheet(plane_overlay_paths, output_dir / "contact_sheet_plane_overlays.png")

    print(f"Saved optimized poses to {optimized_pose_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    print(f"Saved debug metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
