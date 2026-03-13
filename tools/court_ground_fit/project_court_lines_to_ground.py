#!/usr/bin/env python3
"""Project court-line masks from each camera onto the estimated ground plane.

This script treats each non-zero pixel in ``court_line_masks.npy`` as a projector
pixel, casts a ray from the corresponding camera through that pixel, intersects
the ray with the ground plane estimated from the MASt3R point cloud, and writes
minimal downstream artifacts under ``data/.../court/ground`` while saving debug
visualizations under ``results/.../court/ground``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.scene_viewer.utils.court_init_estimator import CourtInitEstimator


@dataclass
class CameraProjectionSummary:
    index: int
    stem: str
    mask_pixels: int
    projected_pixels: int
    dropped_pixels: int
    x_min: float | None
    x_max: float | None
    y_min: float | None
    y_max: float | None
    footprint_pixels: int = 0
    footprint_area_m2: float = 0.0
    sigma_m: float | None = None
    center_u: float | None = None
    center_v: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/tennis_court"),
        help="Scene root containing images_train.txt and mast3r/.",
    )
    parser.add_argument(
        "--mast3r-dir",
        type=Path,
        default=None,
        help="Override MASt3R directory. Defaults to <scene-dir>/mast3r.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Override court_line_masks.npy path. Defaults to <scene-dir>/court/line/court_line_masks.npy.",
    )
    parser.add_argument(
        "--ground-dir",
        type=Path,
        default=None,
        help="Directory for downstream ground-plane artifacts. Defaults to <scene-dir>/court/ground.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tennis_court/court/ground"),
        help="Directory for projected artifacts.",
    )
    parser.add_argument(
        "--court-base-width",
        type=float,
        default=512.0,
        help="Base width assumed by MASt3R intrinsics before scaling to full resolution.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.04,
        help="Top-down raster resolution in meters per pixel.",
    )
    parser.add_argument(
        "--extent-percentile",
        type=float,
        default=0.5,
        help="Lower/upper percentile used to suppress outliers when choosing output bounds.",
    )
    parser.add_argument(
        "--extent-margin",
        type=float,
        default=2.0,
        help="Extra margin in meters added around the robust projected extent.",
    )
    parser.add_argument("--n-sample", type=int, default=50_000)
    parser.add_argument("--ransac-iters", type=int, default=1_000)
    parser.add_argument("--ransac-thr", type=float, default=0.05)
    parser.add_argument(
        "--max-saved-points",
        type=int,
        default=200_000,
        help="Maximum number of projected 3D points to store in the merged sample NPZ.",
    )
    parser.add_argument(
        "--confidence-sigma-scale",
        type=float,
        default=0.45,
        help="Sigma scale relative to the equivalent radius of the visible footprint area.",
    )
    parser.add_argument(
        "--confidence-sigma-min",
        type=float,
        default=2.0,
        help="Minimum Gaussian sigma for per-camera confidence maps, in meters.",
    )
    parser.add_argument(
        "--confidence-sigma-max",
        type=float,
        default=20.0,
        help="Maximum Gaussian sigma for per-camera confidence maps, in meters.",
    )
    return parser.parse_args()


def load_train_stems(scene_dir: Path) -> list[str]:
    names_path = scene_dir / "images_train.txt"
    if not names_path.exists():
        raise FileNotFoundError(f"Missing train image list: {names_path}")
    return [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scale_intrinsics_to_fullres(K: np.ndarray, image_width: int, base_width: float) -> np.ndarray:
    scale = float(image_width) / float(base_width)
    K_full = np.asarray(K, dtype=np.float64).copy()
    K_full[:2, :] *= scale
    K_full[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return K_full


def estimate_ground_frame(
    mast3r_dir: Path,
    camera_poses: np.ndarray,
    *,
    n_sample: int,
    ransac_iters: int,
    ransac_thr: float,
) -> dict[str, np.ndarray | float]:
    estimator = CourtInitEstimator(mast3r_dir)
    pts_xyz, _ = estimator._load_point_cloud(n_sample)
    cam_centers = camera_poses[:, :3, 3]

    normal, plane_d = estimator._fit_plane_ransac(pts_xyz, ransac_iters, ransac_thr)
    cam_centroid = cam_centers.mean(axis=0)
    if float(np.dot(cam_centroid, normal) - plane_d) < 0.0:
        normal = -normal
        plane_d = -plane_d

    cam_proj = cam_centers - (cam_centers @ normal - plane_d)[:, None] * normal
    e_y, e_x_cand = estimator._pca_axes_from_points(cam_proj, normal)

    cam_proj_y = cam_proj @ e_y
    cam_proj_x = cam_proj @ e_x_cand
    ext_y = float(np.percentile(cam_proj_y, 95) - np.percentile(cam_proj_y, 5))
    ext_x = float(np.percentile(cam_proj_x, 95) - np.percentile(cam_proj_x, 5))
    if ext_y < ext_x:
        e_y, e_x_cand = e_x_cand, e_y

    e_z = normal
    e_y = e_y - np.dot(e_y, e_z) * e_z
    e_y /= np.linalg.norm(e_y)
    e_x = np.cross(e_y, e_z)
    e_x /= np.linalg.norm(e_x)

    plane_dists = np.abs(pts_xyz @ normal - plane_d)
    near_plane_mask = plane_dists < float(np.clip(2.0, 0.05, 5.0))
    inlier_pts = pts_xyz[near_plane_mask] if int(near_plane_mask.sum()) >= 10 else cam_proj
    cx = float(np.median(inlier_pts @ e_x))
    cy = float(np.median(inlier_pts @ e_y))
    origin = cx * e_x + cy * e_y + plane_d * e_z

    return {
        "plane_normal": normal.astype(np.float64),
        "plane_d": float(plane_d),
        "origin": origin.astype(np.float64),
        "axis_x": e_x.astype(np.float64),
        "axis_y": e_y.astype(np.float64),
        "plane_points_sample": pts_xyz.astype(np.float32),
    }


def project_mask_to_plane(
    mask: np.ndarray,
    c2w: np.ndarray,
    K_full: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        empty_uv = np.empty((0, 2), dtype=np.float32)
        empty_xyz = np.empty((0, 3), dtype=np.float32)
        return empty_uv, empty_xyz

    pixels = np.stack((xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)), axis=0)
    K_inv = np.linalg.inv(K_full)
    dirs_cam = (K_inv @ pixels).T
    dirs_world = (c2w[:3, :3] @ dirs_cam.T).T

    cam_center = c2w[:3, 3].astype(np.float64)
    denom = dirs_world @ plane_normal
    numer = float(plane_d) - float(np.dot(cam_center, plane_normal))
    valid = np.abs(denom) > 1.0e-9
    t = np.full((dirs_world.shape[0],), np.nan, dtype=np.float64)
    t[valid] = numer / denom[valid]
    valid &= t > 0.0
    if not np.any(valid):
        empty_uv = np.empty((0, 2), dtype=np.float32)
        empty_xyz = np.empty((0, 3), dtype=np.float32)
        return empty_uv, empty_xyz

    xyz = cam_center[None, :] + t[valid, None] * dirs_world[valid]
    rel = xyz - origin[None, :]
    uv = np.column_stack((rel @ axis_x, rel @ axis_y))
    return uv.astype(np.float32), xyz.astype(np.float32)


def choose_extent(all_uv: np.ndarray, percentile: float, margin: float) -> tuple[float, float, float, float]:
    low = float(percentile)
    high = float(100.0 - percentile)
    x0, y0 = np.percentile(all_uv, low, axis=0)
    x1, y1 = np.percentile(all_uv, high, axis=0)
    x0 = float(x0 - margin)
    x1 = float(x1 + margin)
    y0 = float(y0 - margin)
    y1 = float(y1 + margin)
    if not x1 > x0:
        x0 -= 1.0
        x1 += 1.0
    if not y1 > y0:
        y0 -= 1.0
        y1 += 1.0
    return x0, x1, y0, y1


def rasterize_uv(uv: np.ndarray, extent: tuple[float, float, float, float], resolution: float) -> np.ndarray:
    x_min, x_max, y_min, y_max = extent
    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    counts = np.zeros((height, width), dtype=np.uint16)
    if uv.size == 0:
        return counts

    cols = np.floor((uv[:, 0] - x_min) / resolution).astype(np.int64)
    rows = np.floor((y_max - uv[:, 1]) / resolution).astype(np.int64)
    valid = (
        (cols >= 0)
        & (cols < width)
        & (rows >= 0)
        & (rows < height)
    )
    if not np.any(valid):
        return counts

    flat = rows[valid] * width + cols[valid]
    bincount = np.bincount(flat, minlength=height * width)
    counts += bincount.reshape(height, width).astype(np.uint16)
    return counts


def grid_shape_from_extent(extent: tuple[float, float, float, float], resolution: float) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = extent
    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    return height, width


def build_plane_grid(
    extent: tuple[float, float, float, float],
    resolution: float,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    height, width = grid_shape_from_extent(extent, resolution)
    x_min, _, _, y_max = extent
    cols = x_min + (np.arange(width, dtype=np.float64) + 0.5) * resolution
    rows = y_max - (np.arange(height, dtype=np.float64) + 0.5) * resolution
    grid_u, grid_v = np.meshgrid(cols, rows)
    uv = np.column_stack((grid_u.reshape(-1), grid_v.reshape(-1))).astype(np.float32)
    xyz = (
        origin[None, :]
        + uv[:, 0:1].astype(np.float64) * axis_x[None, :]
        + uv[:, 1:2].astype(np.float64) * axis_y[None, :]
    ).astype(np.float32)
    return uv, xyz, (height, width)


def invert_c2w(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = c2w[:3, :3].astype(np.float64)
    translation = c2w[:3, 3].astype(np.float64)
    world_to_camera_rotation = rotation.T
    world_to_camera_translation = -world_to_camera_rotation @ translation
    return world_to_camera_rotation, world_to_camera_translation


def project_camera_center_to_plane_uv(
    cam_center: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    origin: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> np.ndarray:
    projected = cam_center - (float(np.dot(cam_center, plane_normal)) - plane_d) * plane_normal
    rel = projected - origin
    return np.array([float(np.dot(rel, axis_x)), float(np.dot(rel, axis_y))], dtype=np.float32)


def compute_camera_confidence_map(
    world_points: np.ndarray,
    plane_uv: np.ndarray,
    grid_shape: tuple[int, int],
    c2w: np.ndarray,
    K_full: np.ndarray,
    image_width: int,
    image_height: int,
    center_uv: np.ndarray,
    resolution: float,
    sigma_scale: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[np.ndarray, int, float]:
    world_to_camera_rotation, world_to_camera_translation = invert_c2w(c2w)
    cam_points = world_points.astype(np.float64) @ world_to_camera_rotation.T + world_to_camera_translation[None, :]
    z = cam_points[:, 2]
    valid = z > 1.0e-6
    if np.any(valid):
        u = np.empty_like(z)
        v = np.empty_like(z)
        u[valid] = K_full[0, 0] * (cam_points[valid, 0] / z[valid]) + K_full[0, 2]
        v[valid] = K_full[1, 1] * (cam_points[valid, 1] / z[valid]) + K_full[1, 2]
        valid &= (u >= 0.0) & (u <= float(image_width - 1)) & (v >= 0.0) & (v <= float(image_height - 1))

    footprint_pixels = int(valid.sum())
    if footprint_pixels <= 0:
        return np.zeros(grid_shape, dtype=np.float16), 0, float(sigma_min)

    footprint_area = footprint_pixels * (resolution ** 2)
    equivalent_radius = math.sqrt(max(footprint_area, resolution ** 2) / math.pi)
    sigma_m = float(np.clip(equivalent_radius * sigma_scale, sigma_min, sigma_max))

    delta_u = plane_uv[:, 0].astype(np.float64) - float(center_uv[0])
    delta_v = plane_uv[:, 1].astype(np.float64) - float(center_uv[1])
    dist2 = delta_u * delta_u + delta_v * delta_v
    confidence = np.zeros((plane_uv.shape[0],), dtype=np.float32)
    confidence[valid] = np.exp((-0.5 * dist2[valid]) / max(sigma_m * sigma_m, 1.0e-8)).astype(np.float32)
    return confidence.reshape(grid_shape).astype(np.float16), footprint_pixels, sigma_m


def write_binary_image(path: Path, counts: np.ndarray) -> None:
    binary = np.where(counts > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(str(path), binary)


def write_heatmap_image(path: Path, counts: np.ndarray) -> None:
    if counts.size == 0 or int(counts.max()) == 0:
        image = np.zeros((max(1, counts.shape[0]), max(1, counts.shape[1]), 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        return
    scaled = np.log1p(counts.astype(np.float32))
    scaled /= float(scaled.max())
    heat_uint8 = np.clip(np.round(255.0 * scaled), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), colored)


def write_confidence_image(path: Path, confidence: np.ndarray) -> None:
    scaled = np.clip(np.round(255.0 * np.asarray(confidence, dtype=np.float32)), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), colored)


def build_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 256, cols: int = 4) -> None:
    if not image_paths:
        return
    rows = int(math.ceil(len(image_paths) / cols))
    canvas = np.full((rows * thumb_size, cols * thumb_size, 3), 24, dtype=np.uint8)
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_NEAREST)
        r = idx // cols
        c = idx % cols
        y0 = r * thumb_size
        x0 = c * thumb_size
        canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb
        label = image_path.stem[:28]
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)


def maybe_save_point_sample(output_path: Path, points_xyz: np.ndarray, points_uv: np.ndarray, max_saved_points: int) -> int:
    total = int(points_xyz.shape[0])
    if total == 0:
        np.savez_compressed(
            output_path,
            xyz=np.empty((0, 3), dtype=np.float32),
            uv=np.empty((0, 2), dtype=np.float32),
        )
        return 0

    if total > max_saved_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(total, size=max_saved_points, replace=False)
        sample_xyz = points_xyz[indices]
        sample_uv = points_uv[indices]
    else:
        sample_xyz = points_xyz
        sample_uv = points_uv
    np.savez_compressed(output_path, xyz=sample_xyz.astype(np.float32), uv=sample_uv.astype(np.float32))
    return int(sample_xyz.shape[0])


def pack_projected_uvs(projected_uvs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(projected_uvs) + 1,), dtype=np.int64)
    if not projected_uvs:
        return np.empty((0, 2), dtype=np.float32), offsets

    lengths = np.array([uv.shape[0] for uv in projected_uvs], dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    if total == 0:
        return np.empty((0, 2), dtype=np.float32), offsets
    packed = np.concatenate(projected_uvs, axis=0).astype(np.float32, copy=False)
    return packed, offsets


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    mast3r_dir = args.mast3r_dir.expanduser().resolve() if args.mast3r_dir is not None else scene_dir / "mast3r"
    mask_path = args.mask_path.expanduser().resolve() if args.mask_path is not None else scene_dir / "court" / "line" / "court_line_masks.npy"
    ground_dir = args.ground_dir.expanduser().resolve() if args.ground_dir is not None else scene_dir / "court" / "ground"
    output_dir = args.output_dir.expanduser().resolve()
    per_camera_dir = output_dir / "per_camera"
    reliability_dir = output_dir / "reliability"

    if not mast3r_dir.exists():
        raise FileNotFoundError(f"Missing MASt3R directory: {mast3r_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing court-line masks: {mask_path}")

    train_stems = load_train_stems(scene_dir)
    intrinsics_path = mast3r_dir / "camera_intrinsics.npy"
    camera_poses_path = mast3r_dir / "camera_poses.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
    if not camera_poses_path.exists():
        raise FileNotFoundError(f"Missing camera poses: {camera_poses_path}")

    masks = np.load(mask_path, mmap_mode="r")
    intrinsics = np.load(intrinsics_path).astype(np.float64)
    camera_poses = np.load(camera_poses_path).astype(np.float64)

    if masks.ndim != 3:
        raise ValueError(f"Expected masks to have shape (N, H, W), got {masks.shape}")
    if intrinsics.shape[0] != masks.shape[0] or camera_poses.shape[0] != masks.shape[0]:
        raise ValueError(
            "Camera, intrinsics, and mask counts do not match: "
            f"poses={camera_poses.shape[0]} intrinsics={intrinsics.shape[0]} masks={masks.shape[0]}"
        )
    if len(train_stems) != masks.shape[0]:
        raise ValueError(f"Train stem count {len(train_stems)} does not match mask count {masks.shape[0]}")

    full_height, full_width = int(masks.shape[1]), int(masks.shape[2])
    ground_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_camera_dir.mkdir(parents=True, exist_ok=True)
    reliability_dir.mkdir(parents=True, exist_ok=True)

    print("Estimating ground plane and in-plane basis...", flush=True)
    ground = estimate_ground_frame(
        mast3r_dir,
        camera_poses,
        n_sample=args.n_sample,
        ransac_iters=args.ransac_iters,
        ransac_thr=args.ransac_thr,
    )
    plane_normal = np.asarray(ground["plane_normal"], dtype=np.float64)
    plane_d = float(ground["plane_d"])
    origin = np.asarray(ground["origin"], dtype=np.float64)
    axis_x = np.asarray(ground["axis_x"], dtype=np.float64)
    axis_y = np.asarray(ground["axis_y"], dtype=np.float64)

    projected_uvs: list[np.ndarray] = []
    projected_xyzs: list[np.ndarray] = []
    summaries: list[CameraProjectionSummary] = []

    print(f"Projecting {masks.shape[0]} camera masks onto the plane...", flush=True)
    for cam_idx in range(masks.shape[0]):
        K_full = scale_intrinsics_to_fullres(intrinsics[cam_idx], full_width, args.court_base_width)
        mask = np.asarray(masks[cam_idx], dtype=np.uint8)
        uv, xyz = project_mask_to_plane(
            mask=mask,
            c2w=camera_poses[cam_idx],
            K_full=K_full,
            plane_normal=plane_normal,
            plane_d=plane_d,
            origin=origin,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        projected_uvs.append(uv)
        projected_xyzs.append(xyz)
        mask_pixels = int(np.count_nonzero(mask))
        projected_pixels = int(uv.shape[0])
        dropped_pixels = int(mask_pixels - projected_pixels)
        if projected_pixels > 0:
            x_min, y_min = uv.min(axis=0)
            x_max, y_max = uv.max(axis=0)
            summary = CameraProjectionSummary(
                index=cam_idx,
                stem=train_stems[cam_idx],
                mask_pixels=mask_pixels,
                projected_pixels=projected_pixels,
                dropped_pixels=dropped_pixels,
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
            )
        else:
            summary = CameraProjectionSummary(
                index=cam_idx,
                stem=train_stems[cam_idx],
                mask_pixels=mask_pixels,
                projected_pixels=0,
                dropped_pixels=dropped_pixels,
                x_min=None,
                x_max=None,
                y_min=None,
                y_max=None,
            )
        summaries.append(summary)

    non_empty_uvs = [uv for uv in projected_uvs if uv.size > 0]
    if not non_empty_uvs:
        raise RuntimeError("No court-line pixels produced valid ground-plane intersections.")

    all_uv = np.concatenate(non_empty_uvs, axis=0)
    all_xyz = np.concatenate([xyz for xyz in projected_xyzs if xyz.size > 0], axis=0)
    extent = choose_extent(all_uv, args.extent_percentile, args.extent_margin)
    packed_uv, packed_offsets = pack_projected_uvs(projected_uvs)
    grid_uv, grid_world, grid_shape = build_plane_grid(extent, args.grid_resolution, origin, axis_x, axis_y)
    reliability_maps: list[np.ndarray] = []
    confidence_image_paths: list[Path] = []

    plane_frame_path = ground_dir / "plane_frame.json"
    plane_frame: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "mast3r_dir": str(mast3r_dir),
        "mask_path": str(mask_path),
        "image_size": [full_height, full_width],
        "plane": {
            "normal": plane_normal.tolist(),
            "d": plane_d,
            "origin": origin.tolist(),
            "axis_x": axis_x.tolist(),
            "axis_y": axis_y.tolist(),
        },
    }
    plane_frame_path.write_text(json.dumps(plane_frame, indent=2), encoding="utf-8")

    projected_train_path = ground_dir / "projected_train.npz"
    np.savez_compressed(
        projected_train_path,
        uv_points=packed_uv,
        uv_offsets=packed_offsets,
        train_stems=np.asarray(train_stems, dtype=str),
        image_size=np.asarray([full_height, full_width], dtype=np.int32),
    )

    raster_grid_path = ground_dir / "raster_grid.json"
    raster_grid = {
        "extent_xy": [float(v) for v in extent],
        "resolution": float(args.grid_resolution),
        "shape_hw": [int(grid_shape[0]), int(grid_shape[1])],
    }
    raster_grid_path.write_text(json.dumps(raster_grid, indent=2), encoding="utf-8")

    merged_counts = rasterize_uv(all_uv, extent, args.grid_resolution)
    np.savez_compressed(
        output_dir / "merged_projection_counts.npz",
        counts=merged_counts,
        extent=np.asarray(extent, dtype=np.float64),
        resolution=np.float64(args.grid_resolution),
    )
    write_binary_image(output_dir / "merged_projection_binary.png", merged_counts)
    write_heatmap_image(output_dir / "merged_projection_heatmap.png", merged_counts)

    saved_camera_images: list[Path] = []
    for cam_idx, (summary, uv) in enumerate(zip(summaries, projected_uvs)):
        counts = rasterize_uv(uv, extent, args.grid_resolution)
        binary_path = per_camera_dir / f"{summary.index:03d}_{summary.stem}.png"
        write_binary_image(binary_path, counts)
        saved_camera_images.append(binary_path)

        K_full = scale_intrinsics_to_fullres(intrinsics[cam_idx], full_width, args.court_base_width)
        center_uv = project_camera_center_to_plane_uv(
            camera_poses[cam_idx, :3, 3].astype(np.float64),
            plane_normal,
            plane_d,
            origin,
            axis_x,
            axis_y,
        )
        confidence_map, footprint_pixels, sigma_m = compute_camera_confidence_map(
            grid_world,
            grid_uv,
            grid_shape,
            camera_poses[cam_idx],
            K_full,
            full_width,
            full_height,
            center_uv,
            args.grid_resolution,
            sigma_scale=float(args.confidence_sigma_scale),
            sigma_min=float(args.confidence_sigma_min),
            sigma_max=float(args.confidence_sigma_max),
        )
        reliability_maps.append(confidence_map)
        summary.footprint_pixels = footprint_pixels
        summary.footprint_area_m2 = float(footprint_pixels * (args.grid_resolution ** 2))
        summary.sigma_m = float(sigma_m)
        summary.center_u = float(center_uv[0])
        summary.center_v = float(center_uv[1])
        confidence_path = reliability_dir / f"{summary.index:03d}_{summary.stem}.png"
        write_confidence_image(confidence_path, confidence_map)
        confidence_image_paths.append(confidence_path)

    build_contact_sheet(saved_camera_images, output_dir / "per_camera_contact_sheet.png")
    build_contact_sheet(confidence_image_paths, output_dir / "reliability_contact_sheet.png")
    saved_points = maybe_save_point_sample(
        output_dir / "merged_projected_points_sample.npz",
        all_xyz,
        all_uv,
        args.max_saved_points,
    )

    visibility_train_path = ground_dir / "visibility_train.npz"
    np.savez_compressed(
        visibility_train_path,
        confidence_maps=np.stack(reliability_maps, axis=0).astype(np.float16),
        camera_center_uv=np.asarray([[item.center_u, item.center_v] for item in summaries], dtype=np.float32),
        sigma_m=np.asarray(
            [item.sigma_m if item.sigma_m is not None else float(args.confidence_sigma_min) for item in summaries],
            dtype=np.float32,
        ),
        train_stems=np.asarray(train_stems, dtype=str),
    )

    manifest: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "mast3r_dir": str(mast3r_dir),
        "mask_path": str(mask_path),
        "ground_dir": str(ground_dir),
        "debug_output_dir": str(output_dir),
        "image_size": [full_height, full_width],
        "raster_grid_path": str(raster_grid_path),
        "visibility_train_path": str(visibility_train_path),
        "plane": {
            "normal": plane_normal.tolist(),
            "d": plane_d,
            "origin": origin.tolist(),
            "axis_x": axis_x.tolist(),
            "axis_y": axis_y.tolist(),
        },
        "projection": {
            "camera_count": int(len(summaries)),
            "total_mask_pixels": int(sum(item.mask_pixels for item in summaries)),
            "total_projected_pixels": int(sum(item.projected_pixels for item in summaries)),
            "packed_uv_points": int(packed_uv.shape[0]),
            "total_footprint_pixels": int(sum(item.footprint_pixels for item in summaries)),
        },
        "cameras": [asdict(item) for item in summaries],
    }
    (ground_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    debug_metadata = {
        "grid_resolution": float(args.grid_resolution),
        "extent_xy": [float(v) for v in extent],
        "merged_point_sample_saved": int(saved_points),
        "merged_heatmap_max_count": int(merged_counts.max()),
        "mean_confidence_max": float(np.mean([float(np.max(item)) for item in reliability_maps])) if reliability_maps else 0.0,
    }
    (output_dir / "metadata.json").write_text(json.dumps(debug_metadata, indent=2), encoding="utf-8")

    print(f"Saved merged heatmap to {output_dir / 'merged_projection_heatmap.png'}", flush=True)
    print(f"Saved {len(saved_camera_images)} per-camera top-down masks to {per_camera_dir}", flush=True)
    print(f"Saved plane frame to {plane_frame_path}", flush=True)
    print(f"Saved raster grid to {raster_grid_path}", flush=True)
    print(f"Saved projected UVs to {projected_train_path}", flush=True)
    print(f"Saved confidence maps to {visibility_train_path}", flush=True)
    print(f"Saved manifest to {ground_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
