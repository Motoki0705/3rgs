#!/usr/bin/env python3
"""
mast3r_loader.py
----------------
Load scene data from MASt3R preprocessing outputs.

Sources:
  - point cloud: data_dir/mast3r/pointcloud.ply  (random sample)
  - cameras:     data_dir/mast3r/camera_poses.npy       [78, 4, 4] c2w
                 data_dir/mast3r/camera_intrinsics.npy  [78, 3, 3] K
  - image names: data_dir/images_train.txt  (78 stems, same order as npy)

NOTE: Poses are in the raw MASt3R coordinate space (unnormalized).

Usage:
    python tools/scene_viewer/utils/mast3r_loader.py \
        --data-dir data/tennis_court \
        --n-points 50000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print("[mast3r_loader] plyfile not found, installing…", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile", "-q"])
    from plyfile import PlyData


def load(data_dir: str, n_points: int = 50000) -> dict:
    data_dir = Path(data_dir).resolve()
    mast3r_dir = data_dir / "mast3r"
    images_dir = data_dir / "images"

    print(f"[mast3r_loader] data_dir = {data_dir}", file=sys.stderr)

    # ── Train image names ────────────────────────────────────────────────────
    with open(data_dir / "images_train.txt") as f:
        train_stems = [l.strip() for l in f if l.strip()]

    # Find image extension
    img_files = sorted(os.listdir(images_dir))
    ext = os.path.splitext(img_files[0])[1] if img_files else ".jpg"

    # ── Camera poses & intrinsics ─────────────────────────────────────────────
    camera_poses = np.load(mast3r_dir / "camera_poses.npy")       # [78, 4, 4]
    camera_intrinsics = np.load(mast3r_dir / "camera_intrinsics.npy")  # [78, 3, 3]

    assert camera_poses.shape[0] == len(train_stems), (
        f"Mismatch: {camera_poses.shape[0]} poses vs {len(train_stems)} train images"
    )

    cameras = []
    for i, stem in enumerate(train_stems):
        K = camera_intrinsics[i]
        c2w = camera_poses[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        # Approximate image size from principal point
        # (mast3r uses downscaled images; actual width ≈ 2*cx, height ≈ 2*cy)
        width  = int(round(cx * 2))
        height = int(round(cy * 2))

        cameras.append({
            "image_name": f"{stem}{ext}",
            "global_idx": i,
            "is_train": True,
            "c2w": c2w.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        })

    print(f"[mast3r_loader] {len(cameras)} train cameras", file=sys.stderr)

    # ── Point cloud ───────────────────────────────────────────────────────────
    ply_path = mast3r_dir / "pointcloud.ply"
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    n_total = len(v)
    n_sample = min(n_points, n_total)
    idx = np.random.choice(n_total, size=n_sample, replace=False)

    xs = np.asarray(v["x"])[idx]
    ys = np.asarray(v["y"])[idx]
    zs = np.asarray(v["z"])[idx]
    rs = np.asarray(v["red"])[idx].astype(np.float32) / 255.0
    gs = np.asarray(v["green"])[idx].astype(np.float32) / 255.0
    bs = np.asarray(v["blue"])[idx].astype(np.float32) / 255.0

    pts_xyz = np.stack([xs, ys, zs], axis=1)  # [n_sample, 3]
    pts_rgb = np.stack([rs, gs, bs], axis=1)  # [n_sample, 3]

    print(f"[mast3r_loader] {n_sample}/{n_total} points sampled", file=sys.stderr)

    return {
        "mode": "mast3r",
        "points": {
            "xyz": pts_xyz.tolist(),
            "rgb": pts_rgb.tolist(),
        },
        "cameras": cameras,
        "image_dir": str(images_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-points", type=int, default=50000)
    args = parser.parse_args()

    data = load(args.data_dir, args.n_points)
    json.dump(data, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
