#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pycolmap
from PIL import Image


INVALID_POINT3D_ID = 2**64 - 1


def load_train_names(scene_dir: Path) -> list[str]:
    train_file = scene_dir / "images_train.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"{train_file} is missing. Run scripts/prepare_colmap_scene.py first."
        )
    return [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_xy(x: float, y: float, actual_width: int) -> int:
    scale = 512.0 / float(actual_width)
    x_512 = int(np.clip(np.rint(x * scale), 0, 511))
    y_512 = int(np.clip(np.rint(y * scale), 0, 511))
    return y_512 * 512 + x_512


def collect_image_points(image) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        point3d_id = int(point2d.point3D_id)
        if point3d_id == INVALID_POINT3D_ID:
            continue
        xy = np.asarray(point2d.xy, dtype=np.float32)
        points[point3d_id] = (float(xy[0]), float(xy[1]))
    return points


def ensure_resized_images(scene_dir: Path, factor: int, image_names: list[str]) -> Path:
    image_dir = scene_dir / ("images" if factor == 1 else f"images_{factor}")
    if image_dir.exists():
        return image_dir

    source_dir = scene_dir / "images"
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} is missing.")

    image_dir.mkdir(parents=True, exist_ok=True)
    for name in image_names:
        src = source_dir / name
        dst = image_dir / name
        with Image.open(src) as image:
            width, height = image.size
            resized = image.resize((width // factor, height // factor), Image.Resampling.LANCZOS)
            resized.save(dst)
    return image_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3RGS epipolar correspondence tensors from COLMAP tracks."
    )
    parser.add_argument("--scene_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--pair_window", type=int, default=2)
    parser.add_argument("--min_shared_points", type=int, default=64)
    parser.add_argument("--num_correspondences", type=int, default=256)
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found under {scene_dir}")

    output_dir = (args.output_dir or scene_dir / "mast3r").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    train_stems = load_train_names(scene_dir)
    images_by_stem = {Path(image.name).stem: image for image in reconstruction.images.values()}
    missing = [stem for stem in train_stems if stem not in images_by_stem]
    if missing:
        raise RuntimeError(f"Train images are missing from COLMAP reconstruction: {missing[:5]}")

    train_images = [images_by_stem[stem] for stem in train_stems]
    train_image_points = [collect_image_points(image) for image in train_images]

    image_dir = ensure_resized_images(scene_dir, args.data_factor, [image.name for image in train_images])
    actual_sample = imageio.imread(image_dir / train_images[0].name)
    actual_height, actual_width = actual_sample.shape[:2]
    full_width = int(train_images[0].camera.width)
    full_height = int(train_images[0].camera.height)
    scale_x = actual_width / float(full_width)
    scale_y = actual_height / float(full_height)
    num_corr = args.num_correspondences

    corr_i_all: list[np.ndarray] = []
    corr_j_all: list[np.ndarray] = []
    corr_mask_all: list[np.ndarray] = []
    corr_weight_all: list[np.ndarray] = []
    corr_batch_idx_all: list[np.ndarray] = []
    ei_all: list[int] = []
    ej_all: list[int] = []

    for i in range(len(train_images)):
        for offset in range(1, args.pair_window + 1):
            j = i + offset
            if j >= len(train_images):
                break

            shared_ids = list(set(train_image_points[i].keys()) & set(train_image_points[j].keys()))
            if len(shared_ids) < args.min_shared_points:
                continue

            weighted_matches = []
            for point3d_id in shared_ids:
                point3d = reconstruction.points3D[point3d_id]
                xi, yi = train_image_points[i][point3d_id]
                xj, yj = train_image_points[j][point3d_id]
                weight = 1.0 / (1.0 + float(point3d.error))
                weighted_matches.append((weight, xi, yi, xj, yj))

            weighted_matches.sort(key=lambda item: item[0], reverse=True)
            selected = weighted_matches[:num_corr]

            corr_i = np.zeros((num_corr,), dtype=np.int64)
            corr_j = np.zeros((num_corr,), dtype=np.int64)
            corr_mask = np.zeros((num_corr,), dtype=np.float32)
            corr_weight = np.zeros((num_corr,), dtype=np.float32)
            corr_batch_idx = np.zeros((num_corr,), dtype=np.int64)

            for idx, (weight, xi, yi, xj, yj) in enumerate(selected):
                corr_i[idx] = encode_xy(xi * scale_x, yi * scale_y, actual_width)
                corr_j[idx] = encode_xy(xj * scale_x, yj * scale_y, actual_width)
                corr_mask[idx] = 1.0
                corr_weight[idx] = weight

            corr_i_all.append(corr_i)
            corr_j_all.append(corr_j)
            corr_mask_all.append(corr_mask)
            corr_weight_all.append(corr_weight)
            corr_batch_idx_all.append(corr_batch_idx)
            ei_all.append(i)
            ej_all.append(j)

    if not corr_i_all:
        raise RuntimeError("No train pairs satisfied the correspondence threshold.")

    num_train = len(train_images)
    depthmaps = np.zeros((num_train, 1, 1), dtype=np.float32)

    np.save(output_dir / "corr_i.npy", np.stack(corr_i_all))
    np.save(output_dir / "corr_j.npy", np.stack(corr_j_all))
    np.save(output_dir / "corr_batch_idx.npy", np.stack(corr_batch_idx_all))
    np.save(output_dir / "corr_mask.npy", np.stack(corr_mask_all))
    np.save(output_dir / "corr_weight.npy", np.stack(corr_weight_all))
    np.save(output_dir / "ei.npy", np.asarray(ei_all, dtype=np.int64))
    np.save(output_dir / "ej.npy", np.asarray(ej_all, dtype=np.int64))
    np.save(output_dir / "depthmaps.npy", depthmaps)

    print(f"Prepared epipolar data: {output_dir}")
    print(f"Train images: {num_train}")
    print(f"Pairs: {len(corr_i_all)}")
    print(f"Correspondences per pair: {num_corr}")


if __name__ == "__main__":
    main()
