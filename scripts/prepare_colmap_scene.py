#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pycolmap


def camera_to_k(camera) -> np.ndarray:
    model = str(camera.model).split(".")[-1]
    params = np.asarray(camera.params, dtype=np.float32)
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model == "RADIAL":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def image_to_c2w(image) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :] = image.cam_from_world().inverse().matrix().astype(np.float32)
    return c2w


def build_split(names: list[str], test_every: int) -> tuple[list[str], list[str]]:
    test = [name for idx, name in enumerate(names) if idx % test_every == 0]
    train = [name for idx, name in enumerate(names) if idx % test_every != 0]
    if not test and names:
        test = [names[0]]
        train = names[1:]
    if not train and names:
        train = names[:-1]
        test = [names[-1]]
    if not train or not test:
        raise ValueError("Need at least 2 registered images to build train/test splits.")
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the minimum 3RGS MASt3R-style metadata from a COLMAP scene."
    )
    parser.add_argument("--scene_dir", required=True, type=Path)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--mast3r_dir", type=Path, default=None)
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found under {scene_dir}")

    images_dir = scene_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found under {scene_dir}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    registered = sorted(reconstruction.images.values(), key=lambda img: img.name)
    stems = [Path(img.name).stem for img in registered]
    train_stems, test_stems = build_split(stems, args.test_every)
    train_set = set(train_stems)

    train_intrinsics = []
    train_poses = []
    pose_gt_train = []
    pose_gt_test = []

    for image in registered:
        stem = Path(image.name).stem
        camera = reconstruction.cameras[image.camera_id]
        K = camera_to_k(camera)
        c2w = image_to_c2w(image)
        if stem in train_set:
            train_intrinsics.append(K)
            train_poses.append(c2w)
            pose_gt_train.append(c2w)
        else:
            pose_gt_test.append(c2w)

    mast3r_dir = (args.mast3r_dir or scene_dir / "mast3r").expanduser().resolve()
    mast3r_dir.mkdir(parents=True, exist_ok=True)

    np.save(mast3r_dir / "camera_intrinsics.npy", np.stack(train_intrinsics).astype(np.float32))
    np.save(mast3r_dir / "camera_poses.npy", np.stack(train_poses).astype(np.float32))
    np.save(scene_dir / "pose_gt_train.npy", np.stack(pose_gt_train).astype(np.float32))
    np.save(scene_dir / "pose_gt_test.npy", np.stack(pose_gt_test).astype(np.float32))

    (scene_dir / "images_train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (scene_dir / "images_test.txt").write_text("\n".join(test_stems) + "\n", encoding="utf-8")

    src_ply = sparse_dir / "points3D.ply"
    if not src_ply.exists():
        raise FileNotFoundError(f"{src_ply} is missing; generate COLMAP PLY first.")
    shutil.copy2(src_ply, mast3r_dir / "pointcloud.ply")

    print(f"Prepared scene: {scene_dir}")
    print(f"Registered images: {len(registered)}")
    print(f"Train images: {len(train_stems)}")
    print(f"Test images: {len(test_stems)}")
    print(f"Metadata dir: {mast3r_dir}")


if __name__ == "__main__":
    main()
