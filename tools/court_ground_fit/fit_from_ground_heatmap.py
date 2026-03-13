#!/usr/bin/env python3
"""Fit a two-court template to ground-plane heatmaps derived from camera masks.

Pipeline:
1. Project each camera's court-line mask onto the estimated ground plane.
2. Fit the model on contiguous camera chunks.
3. Cluster the chunk poses, refit on the dominant cluster, and export an
   init_sim3.json-style result for later refinement.
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

from project_court_lines_to_ground import (
    estimate_ground_frame,
    load_train_stems,
    project_mask_to_plane,
    rasterize_uv,
    scale_intrinsics_to_fullres,
)
from utils.court_scheme import COURT_SKELETON, DOUBLES_WIDTH, court_keypoints_3d


@dataclass
class FitResult:
    stage: str
    params: dict[str, float]
    total_loss: float
    forward_loss: float
    backward_loss: float
    model_pixels: int
    data_pixels: int


@dataclass
class ChunkFitRecord:
    chunk_index: int
    start: int
    end: int
    camera_indices: list[int]
    projected_pixels: int
    fit: FitResult
    cluster: int = -1


@dataclass
class ClusterSummary:
    cluster: int
    chunk_count: int
    chunk_indices: list[int]
    camera_indices: list[int]
    total_camera_votes: float
    mean_params: dict[str, float]


@dataclass
class HeatmapData:
    counts: np.ndarray
    weights: np.ndarray
    clean_mask: np.ndarray
    dt_to_data: np.ndarray
    data_rows: np.ndarray
    data_cols: np.ndarray
    data_weights: np.ndarray
    extent: tuple[float, float, float, float]
    resolution: float
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/tennis_court"))
    parser.add_argument("--mast3r-dir", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tennis_court/court_ground_heatmap_fit"),
    )
    parser.add_argument("--adjacent-court-direction", choices=("+x", "-x", "+y", "-y"), default="+x")
    parser.add_argument("--court-base-width", type=float, default=512.0)
    parser.add_argument("--grid-resolution", type=float, default=0.04)
    parser.add_argument("--extent-margin", type=float, default=2.0)
    parser.add_argument("--extent-percentile", type=float, default=0.5)
    parser.add_argument("--n-sample", type=int, default=50_000)
    parser.add_argument("--ransac-iters", type=int, default=1_000)
    parser.add_argument("--ransac-thr", type=float, default=0.05)
    parser.add_argument("--line-sample-step", type=float, default=0.08)
    parser.add_argument("--line-thickness-px", type=int, default=2)
    parser.add_argument("--weight-threshold-quantile", type=float, default=0.68)
    parser.add_argument("--morph-open", type=int, default=3)
    parser.add_argument("--morph-close", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=32)
    parser.add_argument("--lambda-backward", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--window-stride", type=int, default=6)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--min-silhouette", type=float, default=0.16)
    parser.add_argument("--init-gap", type=float, default=3.0)
    parser.add_argument("--search-theta-step-deg", type=float, default=8.0)
    parser.add_argument("--search-theta-min-deg", type=float, default=0.25)
    parser.add_argument("--search-trans-step", type=float, default=1.25)
    parser.add_argument("--search-trans-min", type=float, default=0.05)
    parser.add_argument("--search-logscale-step", type=float, default=0.08)
    parser.add_argument("--search-logscale-min", type=float, default=0.002)
    parser.add_argument("--search-gap-step", type=float, default=1.0)
    parser.add_argument("--search-gap-min", type=float, default=0.05)
    parser.add_argument("--search-iters", type=int, default=80)
    parser.add_argument("--max-gap", type=float, default=12.0)
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


def wrap_angle(theta: float) -> float:
    return float(math.atan2(math.sin(theta), math.cos(theta)))


def build_ground_segments() -> tuple[np.ndarray, list[tuple[int, int]]]:
    keypoints = court_keypoints_3d().cpu().numpy().astype(np.float64)
    keypoints_2d = keypoints[:, :2].copy()
    segments: list[tuple[int, int]] = []
    for i0, i1 in COURT_SKELETON:
        if abs(float(keypoints[i0, 2])) > 1.0e-9 or abs(float(keypoints[i1, 2])) > 1.0e-9:
            continue
        segments.append((int(i0), int(i1)))
    return keypoints_2d, segments


def build_weighted_heatmap(
    counts: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float,
    threshold_quantile: float,
    morph_open: int,
    morph_close: int,
    min_component_area: int,
) -> HeatmapData:
    if counts.size == 0:
        raise ValueError("Heatmap counts are empty.")

    weights = np.log1p(counts.astype(np.float32))
    max_weight = float(weights.max())
    if max_weight > 0.0:
        weights /= max_weight
    nonzero = weights[counts > 0]
    def cleanup(binary_mask: np.ndarray) -> np.ndarray:
        clean = binary_mask.astype(np.uint8)
        if morph_open > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
        if morph_close > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        if min_component_area > 1 and int(clean.max()) > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
            filtered = np.zeros_like(clean)
            for label in range(1, n_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area >= min_component_area:
                    filtered[labels == label] = 1
            clean = filtered
        return clean

    threshold_candidates = [float(np.clip(threshold_quantile, 0.0, 1.0)), 0.55, 0.40, 0.25, 0.0]
    threshold = 0.0
    clean_mask = np.zeros_like(counts, dtype=np.uint8)
    for q in threshold_candidates:
        if nonzero.size > 0:
            threshold = float(np.quantile(nonzero, q))
            raw_mask = (weights >= threshold) & (counts > 0)
        else:
            threshold = 0.0
            raw_mask = counts > 0
        clean_mask = cleanup(raw_mask)
        if int(clean_mask.sum()) >= max(8, min_component_area):
            break
    if int(clean_mask.sum()) == 0:
        clean_mask = cleanup(counts > 0)
    if int(clean_mask.sum()) == 0:
        clean_mask = (counts > 0).astype(np.uint8)

    dt_to_data = cv2.distanceTransform((clean_mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    data_rows, data_cols = np.nonzero(clean_mask > 0)
    data_weights = weights[data_rows, data_cols].astype(np.float32)
    return HeatmapData(
        counts=counts.astype(np.uint16, copy=False),
        weights=weights,
        clean_mask=clean_mask.astype(bool),
        dt_to_data=dt_to_data,
        data_rows=data_rows.astype(np.int32),
        data_cols=data_cols.astype(np.int32),
        data_weights=data_weights,
        extent=extent,
        resolution=float(resolution),
        threshold=float(threshold),
    )


def plane_to_grid(uv: np.ndarray, extent: tuple[float, float, float, float], resolution: float) -> np.ndarray:
    x_min, _, _, y_max = extent
    cols = (uv[:, 0] - x_min) / resolution
    rows = (y_max - uv[:, 1]) / resolution
    return np.column_stack((cols, rows)).astype(np.float32)


def grid_to_plane(rows: np.ndarray, cols: np.ndarray, extent: tuple[float, float, float, float], resolution: float) -> np.ndarray:
    x_min, _, _, y_max = extent
    x = x_min + (cols.astype(np.float64) + 0.5) * resolution
    y = y_max - (rows.astype(np.float64) + 0.5) * resolution
    return np.column_stack((x, y)).astype(np.float64)


def sample_ground_points(step: float) -> np.ndarray:
    keypoints, segments = build_ground_segments()
    samples: list[np.ndarray] = []
    for i0, i1 in segments:
        p0 = keypoints[i0]
        p1 = keypoints[i1]
        length = float(np.linalg.norm(p1 - p0))
        num = max(2, int(math.ceil(length / step)) + 1)
        alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
        pts = (1.0 - alpha[:, None]) * p0[None, :] + alpha[:, None] * p1[None, :]
        samples.append(pts)
    return np.concatenate(samples, axis=0) if samples else np.empty((0, 2), dtype=np.float64)


def transform_local_points(local_points: np.ndarray, scale: float, theta: float, translation: np.ndarray) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return scale * (local_points @ rotation.T) + translation[None, :]


def make_two_court_local_points(keypoints_2d: np.ndarray, direction: str, gap: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    shift = adjacent_direction_vector(direction) * (DOUBLES_WIDTH + gap)
    second = keypoints_2d + shift[None, :]
    combined = np.concatenate((keypoints_2d, second), axis=0)
    base_segments = []
    for i0, i1 in build_ground_segments()[1]:
        base_segments.append((i0, i1))
    offset = keypoints_2d.shape[0]
    segments = base_segments + [(i0 + offset, i1 + offset) for i0, i1 in base_segments]
    return combined, segments


def render_model_mask(
    params: dict[str, float],
    extent: tuple[float, float, float, float],
    resolution: float,
    image_shape: tuple[int, int],
    direction: str,
    line_thickness_px: int,
) -> np.ndarray:
    keypoints_2d, _ = build_ground_segments()
    local_keypoints, segments = make_two_court_local_points(keypoints_2d, direction, params["gap"])
    transformed = transform_local_points(
        local_keypoints,
        params["scale"],
        params["theta"],
        np.array([params["tx"], params["ty"]], dtype=np.float64),
    )
    grid_points = plane_to_grid(transformed, extent, resolution)
    height, width = image_shape
    image = np.zeros((height, width), dtype=np.uint8)
    for i0, i1 in segments:
        p0 = grid_points[i0]
        p1 = grid_points[i1]
        x0 = int(round(float(p0[0])))
        y0 = int(round(float(p0[1])))
        x1 = int(round(float(p1[0])))
        y1 = int(round(float(p1[1])))
        cv2.line(image, (x0, y0), (x1, y1), color=255, thickness=max(1, line_thickness_px), lineType=cv2.LINE_AA)
    return image


def evaluate_params(
    params: dict[str, float],
    heatmap: HeatmapData,
    direction: str,
    line_thickness_px: int,
    lambda_backward: float,
) -> FitResult:
    model_mask = render_model_mask(
        params,
        heatmap.extent,
        heatmap.resolution,
        heatmap.counts.shape,
        direction,
        line_thickness_px,
    )
    model_pixels = model_mask > 0
    model_count = int(model_pixels.sum())
    if model_count == 0:
        return FitResult(
            stage="eval",
            params=params,
            total_loss=float("inf"),
            forward_loss=float("inf"),
            backward_loss=float("inf"),
            model_pixels=0,
            data_pixels=int(heatmap.data_rows.size),
        )

    forward_loss = float(heatmap.dt_to_data[model_pixels].mean())
    if heatmap.data_rows.size > 0:
        dt_to_model = cv2.distanceTransform((model_mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        values = dt_to_model[heatmap.data_rows, heatmap.data_cols]
        denom = float(heatmap.data_weights.sum()) + 1.0e-8
        backward_loss = float(np.sum(values * heatmap.data_weights) / denom)
    else:
        backward_loss = float("inf")

    total_loss = forward_loss + lambda_backward * backward_loss
    return FitResult(
        stage="eval",
        params={k: float(v) for k, v in params.items()},
        total_loss=float(total_loss),
        forward_loss=float(forward_loss),
        backward_loss=float(backward_loss),
        model_pixels=model_count,
        data_pixels=int(heatmap.data_rows.size),
    )


def template_centroid(gap: float, direction: str) -> np.ndarray:
    base = sample_ground_points(0.20)
    shift = adjacent_direction_vector(direction) * (DOUBLES_WIDTH + gap)
    points = np.concatenate((base, base + shift[None, :]), axis=0)
    return points.mean(axis=0)


def weighted_data_centroid(heatmap: HeatmapData) -> np.ndarray:
    if heatmap.data_rows.size == 0:
        return np.zeros((2,), dtype=np.float64)
    plane_points = grid_to_plane(heatmap.data_rows, heatmap.data_cols, heatmap.extent, heatmap.resolution)
    weights = heatmap.data_weights.astype(np.float64)
    denom = float(weights.sum()) + 1.0e-8
    return (plane_points * weights[:, None]).sum(axis=0) / denom


def make_seed_params(
    base_scale: float,
    base_theta: float,
    base_gap: float,
    heatmap: HeatmapData,
    direction: str,
) -> list[dict[str, float]]:
    centroid = weighted_data_centroid(heatmap)
    seeds: list[dict[str, float]] = []
    theta_candidates = [base_theta, base_theta + math.pi]
    scale_candidates = [base_scale, base_scale * 1.1, base_scale * 0.9]
    gap_candidates = [max(0.0, base_gap), max(0.0, base_gap - 1.0), base_gap + 1.0]
    seen: set[tuple[int, int, int]] = set()
    for theta in theta_candidates:
        for scale in scale_candidates:
            for gap in gap_candidates:
                key = (int(round(theta * 1000)), int(round(scale * 1000)), int(round(gap * 1000)))
                if key in seen:
                    continue
                seen.add(key)
                local_centroid = template_centroid(gap, direction)
                transformed_centroid = transform_local_points(
                    local_centroid[None, :],
                    scale,
                    theta,
                    np.zeros((2,), dtype=np.float64),
                )[0]
                translation = centroid - transformed_centroid
                seeds.append(
                    {
                        "scale": float(scale),
                        "theta": wrap_angle(theta),
                        "tx": float(translation[0]),
                        "ty": float(translation[1]),
                        "gap": float(max(0.0, gap)),
                    }
                )
    return seeds


def clamp_params(params: dict[str, float], max_gap: float) -> dict[str, float]:
    return {
        "scale": float(max(1.0e-5, params["scale"])),
        "theta": wrap_angle(params["theta"]),
        "tx": float(params["tx"]),
        "ty": float(params["ty"]),
        "gap": float(np.clip(params["gap"], 0.0, max_gap)),
    }


def coordinate_descent_fit(
    heatmap: HeatmapData,
    init_params: dict[str, float],
    direction: str,
    line_thickness_px: int,
    lambda_backward: float,
    search_iters: int,
    theta_step_deg: float,
    theta_min_deg: float,
    trans_step: float,
    trans_min: float,
    logscale_step: float,
    logscale_min: float,
    gap_step: float,
    gap_min: float,
    max_gap: float,
    extra_seeds: list[dict[str, float]] | None = None,
) -> FitResult:
    seeds = [clamp_params(init_params, max_gap)]
    if extra_seeds:
        seeds.extend(clamp_params(seed, max_gap) for seed in extra_seeds)

    best_result: FitResult | None = None
    for seed in seeds:
        current = dict(seed)
        current_result = evaluate_params(current, heatmap, direction, line_thickness_px, lambda_backward)
        steps = {
            "log_scale": float(logscale_step),
            "theta": float(math.radians(theta_step_deg)),
            "tx": float(trans_step),
            "ty": float(trans_step),
            "gap": float(gap_step),
        }
        min_steps = {
            "log_scale": float(logscale_min),
            "theta": float(math.radians(theta_min_deg)),
            "tx": float(trans_min),
            "ty": float(trans_min),
            "gap": float(gap_min),
        }

        for _ in range(max(1, search_iters)):
            improved = False
            dims = ("log_scale", "theta", "tx", "ty", "gap")
            for dim in dims:
                for sign in (-1.0, 1.0):
                    candidate = dict(current)
                    if dim == "log_scale":
                        candidate["scale"] = float(candidate["scale"] * math.exp(sign * steps["log_scale"]))
                    else:
                        candidate[dim] = float(candidate[dim] + sign * steps[dim])
                    candidate = clamp_params(candidate, max_gap)
                    result = evaluate_params(candidate, heatmap, direction, line_thickness_px, lambda_backward)
                    if result.total_loss + 1.0e-6 < current_result.total_loss:
                        current = candidate
                        current_result = result
                        improved = True
            if not improved:
                for dim in dims:
                    steps[dim] *= 0.5
                if all(steps[dim] <= min_steps[dim] for dim in dims):
                    break

        if best_result is None or current_result.total_loss < best_result.total_loss:
            best_result = current_result

    if best_result is None:
        raise RuntimeError("Coordinate descent produced no result.")
    return best_result


def rotation_matrix_from_plane(theta: float, axis_x: np.ndarray, axis_y: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    world_x = c * axis_x + s * axis_y
    world_y = -s * axis_x + c * axis_y
    world_x /= np.linalg.norm(world_x)
    world_y /= np.linalg.norm(world_y)
    world_z = plane_normal / np.linalg.norm(plane_normal)
    return np.column_stack((world_x, world_y, world_z)).astype(np.float64)


def fit_to_sim3_payload(
    fit: FitResult,
    ground: dict[str, np.ndarray | float],
    adjacent_direction: str,
) -> dict[str, Any]:
    plane_normal = np.asarray(ground["plane_normal"], dtype=np.float64)
    axis_x = np.asarray(ground["axis_x"], dtype=np.float64)
    axis_y = np.asarray(ground["axis_y"], dtype=np.float64)
    origin = np.asarray(ground["origin"], dtype=np.float64)
    rotation = rotation_matrix_from_plane(fit.params["theta"], axis_x, axis_y, plane_normal)
    translation = origin + fit.params["tx"] * axis_x + fit.params["ty"] * axis_y
    return {
        "scale": float(fit.params["scale"]),
        "rotation": rotation.tolist(),
        "translation": translation.astype(np.float64).tolist(),
        "adjacent_gap": float(fit.params["gap"]),
        "adjacent_direction": adjacent_direction,
        "plane_normal": plane_normal.astype(np.float64).tolist(),
        "plane_d": float(ground["plane_d"]),
    }


def render_overlay(heatmap: HeatmapData, fit: FitResult, direction: str, line_thickness_px: int) -> np.ndarray:
    heat_uint8 = np.clip(np.round(255.0 * heatmap.weights), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    model = render_model_mask(fit.params, heatmap.extent, heatmap.resolution, heatmap.counts.shape, direction, line_thickness_px)
    clean_edges = np.where(heatmap.clean_mask, 255, 0).astype(np.uint8)
    overlay = color.copy()
    overlay[clean_edges > 0] = (255, 255, 255)
    overlay[model > 0] = (0, 64, 255)
    return overlay


def build_camera_uvs(
    scene_dir: Path,
    mast3r_dir: Path,
    mask_path: Path,
    court_base_width: float,
    n_sample: int,
    ransac_iters: int,
    ransac_thr: float,
) -> tuple[list[str], list[np.ndarray], dict[str, np.ndarray | float]]:
    train_stems = load_train_stems(scene_dir)
    intrinsics = np.load(mast3r_dir / "camera_intrinsics.npy").astype(np.float64)
    camera_poses = np.load(mast3r_dir / "camera_poses.npy").astype(np.float64)
    masks = np.load(mask_path, mmap_mode="r")
    if masks.shape[0] != len(train_stems):
        raise ValueError(f"Mask count {masks.shape[0]} does not match train stem count {len(train_stems)}")
    ground = estimate_ground_frame(
        mast3r_dir,
        camera_poses,
        n_sample=n_sample,
        ransac_iters=ransac_iters,
        ransac_thr=ransac_thr,
    )

    full_width = int(masks.shape[2])
    plane_normal = np.asarray(ground["plane_normal"], dtype=np.float64)
    plane_d = float(ground["plane_d"])
    origin = np.asarray(ground["origin"], dtype=np.float64)
    axis_x = np.asarray(ground["axis_x"], dtype=np.float64)
    axis_y = np.asarray(ground["axis_y"], dtype=np.float64)

    projected_uvs: list[np.ndarray] = []
    for cam_idx in range(masks.shape[0]):
        K_full = scale_intrinsics_to_fullres(intrinsics[cam_idx], full_width, court_base_width)
        mask = np.asarray(masks[cam_idx], dtype=np.uint8)
        uv, _ = project_mask_to_plane(
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
    return train_stems, projected_uvs, ground


def choose_extent(all_uv: np.ndarray, percentile: float, margin: float) -> tuple[float, float, float, float]:
    low = float(percentile)
    high = float(100.0 - percentile)
    x0, y0 = np.percentile(all_uv, low, axis=0)
    x1, y1 = np.percentile(all_uv, high, axis=0)
    x0 = float(x0 - margin)
    x1 = float(x1 + margin)
    y0 = float(y0 - margin)
    y1 = float(y1 + margin)
    return x0, x1, y0, y1


def build_counts_for_cameras(
    projected_uvs: list[np.ndarray],
    camera_indices: list[int],
    extent: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    uv_list = [projected_uvs[idx] for idx in camera_indices if projected_uvs[idx].size > 0]
    if not uv_list:
        x_min, x_max, y_min, y_max = extent
        width = int(math.ceil((x_max - x_min) / resolution)) + 1
        height = int(math.ceil((y_max - y_min) / resolution)) + 1
        return np.zeros((height, width), dtype=np.uint16)
    merged = np.concatenate(uv_list, axis=0)
    return rasterize_uv(merged, extent, resolution)


def fit_single_heatmap(
    counts: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float,
    init_params: dict[str, float],
    args: argparse.Namespace,
    stage: str,
    extra_seeds: list[dict[str, float]] | None = None,
) -> tuple[HeatmapData, FitResult]:
    heatmap = build_weighted_heatmap(
        counts,
        extent,
        resolution,
        threshold_quantile=args.weight_threshold_quantile,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_component_area=args.min_component_area,
    )
    seeds = make_seed_params(
        init_params["scale"],
        init_params["theta"],
        init_params["gap"],
        heatmap,
        args.adjacent_court_direction,
    )
    if extra_seeds:
        seeds.extend(extra_seeds)
    fit = coordinate_descent_fit(
        heatmap,
        init_params=init_params,
        direction=args.adjacent_court_direction,
        line_thickness_px=args.line_thickness_px,
        lambda_backward=args.lambda_backward,
        search_iters=args.search_iters,
        theta_step_deg=args.search_theta_step_deg,
        theta_min_deg=args.search_theta_min_deg,
        trans_step=args.search_trans_step,
        trans_min=args.search_trans_min,
        logscale_step=args.search_logscale_step,
        logscale_min=args.search_logscale_min,
        gap_step=args.search_gap_step,
        gap_min=args.search_gap_min,
        max_gap=args.max_gap,
        extra_seeds=seeds,
    )
    fit.stage = stage
    return heatmap, fit


def params_to_feature_vector(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            math.cos(params["theta"]),
            math.sin(params["theta"]),
            math.log(max(params["scale"], 1.0e-8)),
            params["tx"],
            params["ty"],
            params["gap"],
        ],
        dtype=np.float64,
    )


def run_kmeans(features: np.ndarray, k: int, n_init: int = 16, max_iter: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(0)
    n = features.shape[0]
    best_inertia = float("inf")
    best_labels = np.zeros((n,), dtype=np.int32)
    best_centers = features[:k].copy()

    for _ in range(n_init):
        centers = np.empty((k, features.shape[1]), dtype=np.float64)
        first = int(rng.integers(n))
        centers[0] = features[first]
        for center_idx in range(1, k):
            d2 = np.min(np.sum((features[:, None, :] - centers[None, :center_idx, :]) ** 2, axis=2), axis=1)
            total = float(d2.sum())
            if total <= 1.0e-12:
                centers[center_idx] = features[int(rng.integers(n))]
            else:
                probs = d2 / total
                chosen = int(rng.choice(n, p=probs))
                centers[center_idx] = features[chosen]

        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(max_iter):
            d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d2, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_idx in range(k):
                members = features[labels == cluster_idx]
                if members.size == 0:
                    centers[cluster_idx] = features[int(rng.integers(n))]
                else:
                    centers[cluster_idx] = members.mean(axis=0)

        inertia = float(np.sum((features - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    n = features.shape[0]
    unique = np.unique(labels)
    if unique.size <= 1 or unique.size >= n:
        return -1.0
    dist = np.sqrt(np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2))
    scores = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same_count = int(np.sum(same))
        if same_count <= 1:
            scores[i] = 0.0
            continue
        a = float(dist[i, same].sum() / max(1, same_count - 1))
        b = float("inf")
        for other in unique:
            if other == labels[i]:
                continue
            other_mask = labels == other
            if not np.any(other_mask):
                continue
            b = min(b, float(dist[i, other_mask].mean()))
        denom = max(a, b)
        scores[i] = 0.0 if denom <= 1.0e-12 else (b - a) / denom
    return float(scores.mean())


def cluster_chunk_fits(records: list[ChunkFitRecord], args: argparse.Namespace) -> tuple[np.ndarray, float]:
    labels = np.full((len(records),), -1, dtype=np.int32)
    valid_indices = [
        idx
        for idx, record in enumerate(records)
        if np.isfinite(record.fit.total_loss) and record.fit.data_pixels > 0
    ]
    if len(valid_indices) < 3:
        labels[valid_indices] = 0
        return labels, -1.0

    features = np.stack([params_to_feature_vector(records[idx].fit.params) for idx in valid_indices], axis=0)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    normalized = (features - mean) / std

    best_labels = np.zeros((len(valid_indices),), dtype=np.int32)
    best_score = -1.0
    max_clusters = min(int(args.max_clusters), len(valid_indices) - 1)
    for k in range(2, max_clusters + 1):
        labels, _, _ = run_kmeans(normalized, k)
        score = silhouette_score(normalized, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_score < float(args.min_silhouette):
        labels[valid_indices] = 0
        return labels, best_score
    labels[valid_indices] = best_labels.astype(np.int32)
    return labels, float(best_score)


def weighted_mean_params(records: list[ChunkFitRecord]) -> dict[str, float]:
    weights = np.array(
        [
            record.projected_pixels / max(record.fit.total_loss, 1.0e-4)
            for record in records
        ],
        dtype=np.float64,
    )
    weights = np.clip(weights, 1.0e-8, None)
    weights /= weights.sum()
    theta = math.atan2(
        float(np.sum(weights * np.sin([record.fit.params["theta"] for record in records]))),
        float(np.sum(weights * np.cos([record.fit.params["theta"] for record in records]))),
    )
    log_scale = float(np.sum(weights * np.log([max(record.fit.params["scale"], 1.0e-8) for record in records])))
    tx = float(np.sum(weights * np.array([record.fit.params["tx"] for record in records], dtype=np.float64)))
    ty = float(np.sum(weights * np.array([record.fit.params["ty"] for record in records], dtype=np.float64)))
    gap = float(np.sum(weights * np.array([record.fit.params["gap"] for record in records], dtype=np.float64)))
    return {
        "scale": float(math.exp(log_scale)),
        "theta": wrap_angle(theta),
        "tx": tx,
        "ty": ty,
        "gap": gap,
    }


def dominant_cluster_camera_indices(records: list[ChunkFitRecord], labels: np.ndarray) -> tuple[int, list[int], list[ClusterSummary]]:
    unique = np.unique(labels[labels >= 0])
    if unique.size == 0:
        return 0, [], []

    max_camera = max((max(record.camera_indices) for record in records if record.camera_indices), default=-1) + 1
    camera_votes = np.zeros((max_camera, unique.size), dtype=np.float64)
    cluster_summaries: list[ClusterSummary] = []

    label_to_pos = {int(label): idx for idx, label in enumerate(unique.tolist())}
    for record, label in zip(records, labels.tolist()):
        if int(label) < 0:
            continue
        weight = record.projected_pixels / max(record.fit.total_loss, 1.0e-4)
        pos = label_to_pos[int(label)]
        for camera_idx in record.camera_indices:
            camera_votes[camera_idx, pos] += weight

    cluster_scores = camera_votes.sum(axis=0)
    dominant_pos = int(np.argmax(cluster_scores))
    dominant_label = int(unique[dominant_pos])

    for label in unique.tolist():
        pos = label_to_pos[int(label)]
        member_records = [record for record, current in zip(records, labels.tolist()) if int(current) == int(label)]
        assigned_cameras = [idx for idx in range(max_camera) if int(np.argmax(camera_votes[idx])) == pos and camera_votes[idx, pos] > 0.0]
        cluster_summaries.append(
            ClusterSummary(
                cluster=int(label),
                chunk_count=len(member_records),
                chunk_indices=[record.chunk_index for record in member_records],
                camera_indices=assigned_cameras,
                total_camera_votes=float(cluster_scores[pos]),
                mean_params=weighted_mean_params(member_records) if member_records else {},
            )
        )

    dominant_cameras = [idx for idx in range(max_camera) if int(np.argmax(camera_votes[idx])) == dominant_pos and camera_votes[idx, dominant_pos] > 0.0]
    if not dominant_cameras:
        dominant_cameras = sorted({camera_idx for record, label in zip(records, labels.tolist()) if int(label) == dominant_label for camera_idx in record.camera_indices})
    return dominant_label, dominant_cameras, cluster_summaries


def initial_params_from_ground(ground: dict[str, np.ndarray | float], adjacent_direction: str, init_gap: float) -> dict[str, float]:
    estimator_scale = float(0.25)
    estimator_theta = 0.0
    estimator_tx = 0.0
    estimator_ty = 0.0
    axis_x = np.asarray(ground["axis_x"], dtype=np.float64)
    axis_y = np.asarray(ground["axis_y"], dtype=np.float64)
    plane_normal = np.asarray(ground["plane_normal"], dtype=np.float64)
    origin = np.asarray(ground["origin"], dtype=np.float64)

    from tools.scene_viewer.utils.court_init_estimator import CourtInitEstimator

    estimator = CourtInitEstimator(Path(ground["mast3r_dir"]) if "mast3r_dir" in ground else Path("data/tennis_court/mast3r"))
    res = estimator.estimate()
    estimator_scale = float(res["scale"])
    rotation = np.asarray(res["rotation"], dtype=np.float64)
    translation = np.asarray(res["translation"], dtype=np.float64)
    world_x = rotation[:, 0]
    theta = math.atan2(float(np.dot(world_x, axis_y)), float(np.dot(world_x, axis_x)))
    estimator_theta = wrap_angle(theta)
    estimator_tx = float(np.dot(translation - origin, axis_x))
    estimator_ty = float(np.dot(translation - origin, axis_y))
    if float(np.dot(rotation[:, 2], plane_normal)) < 0.0:
        estimator_theta = wrap_angle(estimator_theta + math.pi)

    return {
        "scale": estimator_scale,
        "theta": estimator_theta,
        "tx": estimator_tx,
        "ty": estimator_ty,
        "gap": float(init_gap),
    }


def save_chunk_contact_sheet(chunk_images: list[Path], output_path: Path, thumb_size: int = 256, cols: int = 4) -> None:
    if not chunk_images:
        return
    rows = int(math.ceil(len(chunk_images) / cols))
    canvas = np.full((rows * thumb_size, cols * thumb_size, 3), 16, dtype=np.uint8)
    for idx, image_path in enumerate(chunk_images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        r = idx // cols
        c = idx % cols
        y0 = r * thumb_size
        x0 = c * thumb_size
        canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb
        cv2.putText(canvas, image_path.stem[:32], (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    mast3r_dir = args.mast3r_dir.expanduser().resolve() if args.mast3r_dir is not None else scene_dir / "mast3r"
    mask_path = args.mask_path.expanduser().resolve() if args.mask_path is not None else mast3r_dir / "court_line_masks.npy"
    output_dir = args.output_dir.expanduser().resolve()
    chunk_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("Projecting per-camera court-line masks to the ground plane...", flush=True)
    train_stems, projected_uvs, ground = build_camera_uvs(
        scene_dir=scene_dir,
        mast3r_dir=mast3r_dir,
        mask_path=mask_path,
        court_base_width=args.court_base_width,
        n_sample=args.n_sample,
        ransac_iters=args.ransac_iters,
        ransac_thr=args.ransac_thr,
    )
    ground["mast3r_dir"] = str(mast3r_dir)
    all_uv = np.concatenate([uv for uv in projected_uvs if uv.size > 0], axis=0)
    extent = choose_extent(all_uv, args.extent_percentile, args.extent_margin)
    init_params = initial_params_from_ground(ground, args.adjacent_court_direction, args.init_gap)

    print("Running chunk-wise heatmap fits...", flush=True)
    chunk_records: list[ChunkFitRecord] = []
    chunk_images: list[Path] = []
    previous_fit: FitResult | None = None
    chunk_index = 0
    for start in range(0, len(projected_uvs), max(1, args.window_stride)):
        end = min(len(projected_uvs), start + max(1, args.window_size))
        if end - start < 2:
            continue
        camera_indices = list(range(start, end))
        counts = build_counts_for_cameras(projected_uvs, camera_indices, extent, args.grid_resolution)
        projected_pixels = int(np.sum(counts, dtype=np.float64))
        if projected_pixels <= 0:
            continue
        chunk_init = previous_fit.params if previous_fit is not None and np.isfinite(previous_fit.total_loss) else init_params
        extra_seeds = [init_params]
        if previous_fit is not None and np.isfinite(previous_fit.total_loss):
            extra_seeds.append(previous_fit.params)
        chunk_heatmap, chunk_fit = fit_single_heatmap(
            counts,
            extent,
            args.grid_resolution,
            init_params=chunk_init,
            args=args,
            stage=f"chunk_{chunk_index:03d}",
            extra_seeds=extra_seeds,
        )
        record = ChunkFitRecord(
            chunk_index=chunk_index,
            start=start,
            end=end,
            camera_indices=camera_indices,
            projected_pixels=projected_pixels,
            fit=chunk_fit,
        )
        chunk_records.append(record)
        image_path = chunk_dir / f"chunk_{chunk_index:03d}_{start:03d}_{end:03d}.png"
        cv2.imwrite(str(image_path), render_overlay(chunk_heatmap, chunk_fit, args.adjacent_court_direction, args.line_thickness_px))
        chunk_images.append(image_path)
        if np.isfinite(chunk_fit.total_loss):
            previous_fit = chunk_fit
        chunk_index += 1

    if not chunk_records:
        raise RuntimeError("No valid chunk fits were produced from the projected court-line masks.")

    labels, silhouette = cluster_chunk_fits(chunk_records, args)
    for record, label in zip(chunk_records, labels.tolist()):
        record.cluster = int(label)

    dominant_label, dominant_cameras, cluster_summaries = dominant_cluster_camera_indices(chunk_records, labels)
    print(
        f"Chunk clustering chose {len(np.unique(labels))} cluster(s), silhouette={silhouette:.4f}, dominant={dominant_label}",
        flush=True,
    )

    if not dominant_cameras:
        best_chunk = min(chunk_records, key=lambda record: record.fit.total_loss)
        dominant_label = int(best_chunk.cluster)
        dominant_cameras = list(best_chunk.camera_indices)

    dominant_counts = build_counts_for_cameras(projected_uvs, dominant_cameras, extent, args.grid_resolution)
    best_chunk = min(chunk_records, key=lambda record: record.fit.total_loss)
    dominant_init = best_chunk.fit.params
    dominant_member_records = [record for record in chunk_records if record.cluster == dominant_label]
    extra_seeds = [init_params, best_chunk.fit.params]
    if dominant_member_records:
        dominant_init = weighted_mean_params(dominant_member_records)
        extra_seeds.append(dominant_member_records[np.argmin([record.fit.total_loss for record in dominant_member_records])].fit.params)

    print(f"Refitting on dominant-cluster cameras ({len(dominant_cameras)} cameras)...", flush=True)
    dominant_heatmap, final_fit = fit_single_heatmap(
        dominant_counts,
        extent,
        args.grid_resolution,
        init_params=dominant_init,
        args=args,
        stage="dominant_cluster_final",
        extra_seeds=extra_seeds,
    )
    cv2.imwrite(str(output_dir / "dominant_cluster_overlay.png"), render_overlay(dominant_heatmap, final_fit, args.adjacent_court_direction, args.line_thickness_px))
    cv2.imwrite(str(output_dir / "dominant_cluster_heatmap.png"), cv2.applyColorMap(np.clip(np.round(255.0 * dominant_heatmap.weights), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    save_chunk_contact_sheet(chunk_images, output_dir / "chunk_overlays_contact_sheet.png")

    selected_fit = final_fit
    selected_heatmap = dominant_heatmap
    selected_fit_source = "dominant_cluster"
    if not np.isfinite(final_fit.total_loss):
        selected_fit = best_chunk.fit
        selected_heatmap, _ = fit_single_heatmap(
            build_counts_for_cameras(projected_uvs, best_chunk.camera_indices, extent, args.grid_resolution),
            extent,
            args.grid_resolution,
            init_params=best_chunk.fit.params,
            args=args,
            stage="best_chunk_fallback",
            extra_seeds=[init_params],
        )
        selected_fit_source = "best_chunk"

    cv2.imwrite(
        str(output_dir / "selected_fit_overlay.png"),
        render_overlay(selected_heatmap, selected_fit, args.adjacent_court_direction, args.line_thickness_px),
    )
    sim3_payload = fit_to_sim3_payload(selected_fit, ground, args.adjacent_court_direction)
    init_sim3_path = output_dir / "init_sim3_from_ground_heatmap.json"
    init_sim3_path.write_text(json.dumps(sim3_payload, indent=2), encoding="utf-8")

    metadata = {
        "scene_dir": str(scene_dir),
        "mast3r_dir": str(mast3r_dir),
        "mask_path": str(mask_path),
        "output_dir": str(output_dir),
        "adjacent_court_direction": args.adjacent_court_direction,
        "extent_xy": [float(v) for v in extent],
        "grid_resolution": float(args.grid_resolution),
        "ground_plane": {
            "normal": np.asarray(ground["plane_normal"], dtype=np.float64).tolist(),
            "d": float(ground["plane_d"]),
            "origin": np.asarray(ground["origin"], dtype=np.float64).tolist(),
            "axis_x": np.asarray(ground["axis_x"], dtype=np.float64).tolist(),
            "axis_y": np.asarray(ground["axis_y"], dtype=np.float64).tolist(),
        },
        "dominant_cluster_fit": asdict(final_fit),
        "selected_fit_source": selected_fit_source,
        "final_fit": asdict(selected_fit),
        "cluster_silhouette": float(silhouette),
        "dominant_cluster": int(dominant_label),
        "dominant_cameras": dominant_cameras,
        "train_stems": train_stems,
        "chunk_fits": [
            {
                "chunk_index": record.chunk_index,
                "start": record.start,
                "end": record.end,
                "camera_indices": record.camera_indices,
                "camera_stems": [train_stems[idx] for idx in record.camera_indices],
                "projected_pixels": record.projected_pixels,
                "cluster": record.cluster,
                "fit": asdict(record.fit),
            }
            for record in chunk_records
        ],
        "clusters": [asdict(summary) for summary in cluster_summaries],
        "exported_init_sim3_path": str(init_sim3_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    np.savez_compressed(
        output_dir / "heatmaps.npz",
        dominant_counts=dominant_counts,
        extent=np.asarray(extent, dtype=np.float64),
        resolution=np.asarray(args.grid_resolution, dtype=np.float64),
        dominant_cameras=np.asarray(dominant_cameras, dtype=np.int32),
    )

    print(f"Saved final init Sim(3) to {init_sim3_path}", flush=True)
    print(f"Saved metadata to {output_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
