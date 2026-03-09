#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAST3R_REPO = REPO_ROOT / "third_party" / "mast3r"


def hash_md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


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
        raise ValueError("Need at least 2 images to build train/test splits.")
    return train, test


def resolve_mast3r_python(mast3r_repo: Path, explicit_python: Path | None) -> Path:
    if explicit_python is not None:
        python_path = explicit_python.expanduser().resolve()
    else:
        python_path = mast3r_repo / ".venv" / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(
            f"MASt3R Python executable not found: {python_path}. "
            "Create third_party/mast3r/.venv before running preprocess.py."
        )
    return python_path


def resolve_pair_dir(result_dir: Path) -> Path:
    preferred = result_dir / "cache" / "corres_conf=desc_conf_subsample=8"
    if preferred.exists():
        return preferred
    candidates = sorted(path for path in (result_dir / "cache").iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No cached correspondence directory found under {result_dir / 'cache'}")
    return candidates[0]


def run_mast3r(args: argparse.Namespace, mast3r_repo: Path, mast3r_python: Path, scene_dir: Path, result_dir: Path) -> None:
    if result_dir.exists() and args.force:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(mast3r_python),
        str(mast3r_repo / "scripts" / "run_mast3r_sfm.py"),
        "--data-dir",
        str(scene_dir),
        "--output-dir",
        str(result_dir),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--scene-graph",
        args.scene_graph,
        "--winsize",
        str(args.winsize),
        "--lr1",
        str(args.lr1),
        "--niter1",
        str(args.niter1),
        "--lr2",
        str(args.lr2),
        "--niter2",
        str(args.niter2),
        "--matching-conf-thr",
        str(args.matching_conf_thr),
        "--min-conf-thr",
        str(args.min_conf_thr),
        "--cam-size",
        str(args.cam_size),
        "--max-points",
        str(args.max_points),
        "--max-match-viz",
        str(args.max_match_viz),
    ]
    if args.shared_intrinsics:
        cmd.append("--shared-intrinsics")

    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(mast3r_repo))


def package_scene(args: argparse.Namespace, scene_dir: Path, result_dir: Path) -> None:
    pose_manifest = json.loads((result_dir / "camera_poses.json").read_text(encoding="utf-8"))
    pose_by_name = {entry["image_name"]: entry for entry in pose_manifest}
    image_names = sorted(pose_by_name)
    stems = [Path(name).stem for name in image_names]
    train_stems, test_stems = build_split(stems, args.test_every)
    train_names = [name for name in image_names if Path(name).stem in set(train_stems)]
    test_names = [name for name in image_names if Path(name).stem in set(test_stems)]

    mast3r_dir = scene_dir / "mast3r"
    mast3r_dir.mkdir(parents=True, exist_ok=True)

    train_intrinsics = [np.asarray(pose_by_name[name]["intrinsics"], dtype=np.float32) for name in train_names]
    train_poses = [np.asarray(pose_by_name[name]["cam2w"], dtype=np.float32) for name in train_names]
    test_poses = [np.asarray(pose_by_name[name]["cam2w"], dtype=np.float32) for name in test_names]

    np.save(mast3r_dir / "camera_intrinsics.npy", np.stack(train_intrinsics).astype(np.float32))
    np.save(mast3r_dir / "camera_poses.npy", np.stack(train_poses).astype(np.float32))
    np.save(scene_dir / "pose_gt_train.npy", np.stack(train_poses).astype(np.float32))
    np.save(scene_dir / "pose_gt_test.npy", np.stack(test_poses).astype(np.float32))
    shutil.copy2(result_dir / "sparse_points.ply", mast3r_dir / "pointcloud.ply")

    (scene_dir / "images_train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (scene_dir / "images_test.txt").write_text("\n".join(test_stems) + "\n", encoding="utf-8")

    run_config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
    data_dir = Path(run_config["data_dir"])
    pair_dir = resolve_pair_dir(result_dir)
    train_idx_by_name = {name: idx for idx, name in enumerate(train_names)}

    total_corr = args.max_correspondences + args.max_manual_correspondences
    corr_i_all: list[np.ndarray] = []
    corr_j_all: list[np.ndarray] = []
    corr_mask_all: list[np.ndarray] = []
    corr_weight_all: list[np.ndarray] = []
    corr_batch_idx_all: list[np.ndarray] = []
    corr_is_manual_all: list[np.ndarray] = []
    ei_all: list[int] = []
    ej_all: list[int] = []

    for left_idx, left_name in enumerate(train_names):
        left_path = data_dir / "images" / left_name
        for right_name in train_names[left_idx + 1:]:
            right_path = data_dir / "images" / right_name
            h1 = hash_md5(str(left_path))
            h2 = hash_md5(str(right_path))
            pair_path = pair_dir / f"{h1}-{h2}.pth"
            reverse = False
            if not pair_path.exists():
                pair_path = pair_dir / f"{h2}-{h1}.pth"
                reverse = True
            if not pair_path.exists():
                continue

            _, payload = torch.load(pair_path, map_location="cpu")
            points_i, points_j, conf = payload
            if reverse:
                points_i, points_j = points_j, points_i

            points_i = points_i[: args.max_correspondences].numpy()
            points_j = points_j[: args.max_correspondences].numpy()
            conf = conf[: args.max_correspondences].numpy()
            if len(conf) == 0:
                continue

            corr_i = np.zeros((total_corr,), dtype=np.int64)
            corr_j = np.zeros((total_corr,), dtype=np.int64)
            corr_mask = np.zeros((total_corr,), dtype=np.float32)
            corr_weight = np.zeros((total_corr,), dtype=np.float32)
            corr_batch_idx = np.zeros((total_corr,), dtype=np.int64)
            corr_is_manual = np.zeros((total_corr,), dtype=np.float32)

            for idx, (xy_i, xy_j, weight) in enumerate(zip(points_i, points_j, conf)):
                corr_i[idx] = int(np.clip(np.rint(xy_i[1]), 0, 511)) * 512 + int(np.clip(np.rint(xy_i[0]), 0, 511))
                corr_j[idx] = int(np.clip(np.rint(xy_j[1]), 0, 511)) * 512 + int(np.clip(np.rint(xy_j[0]), 0, 511))
                corr_mask[idx] = 1.0
                corr_weight[idx] = float(weight)

            corr_i_all.append(corr_i)
            corr_j_all.append(corr_j)
            corr_mask_all.append(corr_mask)
            corr_weight_all.append(corr_weight)
            corr_batch_idx_all.append(corr_batch_idx)
            corr_is_manual_all.append(corr_is_manual)
            ei_all.append(train_idx_by_name[left_name])
            ej_all.append(train_idx_by_name[right_name])

    if not corr_i_all:
        raise RuntimeError("No MASt3R pair correspondences were packaged.")

    depthmaps = np.zeros((len(train_names), 1, 1), dtype=np.float32)
    np.save(mast3r_dir / "corr_i.npy", np.stack(corr_i_all))
    np.save(mast3r_dir / "corr_j.npy", np.stack(corr_j_all))
    np.save(mast3r_dir / "corr_mask.npy", np.stack(corr_mask_all))
    np.save(mast3r_dir / "corr_weight.npy", np.stack(corr_weight_all))
    np.save(mast3r_dir / "corr_batch_idx.npy", np.stack(corr_batch_idx_all))
    np.save(mast3r_dir / "corr_is_manual.npy", np.stack(corr_is_manual_all))
    np.save(mast3r_dir / "ei.npy", np.asarray(ei_all, dtype=np.int64))
    np.save(mast3r_dir / "ej.npy", np.asarray(ej_all, dtype=np.int64))
    np.save(mast3r_dir / "depthmaps.npy", depthmaps)

    summary = {
        "scene_dir": str(scene_dir),
        "mast3r_result_dir": str(result_dir),
        "train_images": len(train_names),
        "test_images": len(test_names),
        "pairs": len(corr_i_all),
        "max_correspondences": args.max_correspondences,
        "max_manual_correspondences": args.max_manual_correspondences,
        "test_every": args.test_every,
    }
    (mast3r_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared scene: {scene_dir}")
    print(f"MASt3R results: {result_dir}")
    print(f"Train images: {len(train_names)}")
    print(f"Test images: {len(test_names)}")
    print(f"Pairs: {len(corr_i_all)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MASt3R-SfM for a 3RGS scene and package the outputs into data/{scene}/mast3r."
    )
    parser.add_argument("--scene_dir", required=True, type=Path, help="Scene root containing images/.")
    parser.add_argument("--mast3r_repo", type=Path, default=DEFAULT_MAST3R_REPO)
    parser.add_argument("--mast3r_python", type=Path, default=None)
    parser.add_argument("--mast3r_result_dir", type=Path, default=None)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--max_correspondences", type=int, default=256)
    parser.add_argument("--max_manual_correspondences", type=int, default=40)
    parser.add_argument("--skip_mast3r_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--scene_graph", default="swin", choices=["swin", "logwin", "complete"])
    parser.add_argument("--winsize", type=int, default=2)
    parser.add_argument("--shared_intrinsics", action="store_true")
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=150)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--niter2", type=int, default=150)
    parser.add_argument("--matching_conf_thr", type=float, default=2.0)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--cam_size", type=float, default=0.03)
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--max_match_viz", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    if not (scene_dir / "images").exists():
        raise FileNotFoundError(f"{scene_dir / 'images'} is missing.")

    mast3r_repo = args.mast3r_repo.expanduser().resolve()
    mast3r_python = resolve_mast3r_python(mast3r_repo, args.mast3r_python)
    result_dir = (args.mast3r_result_dir or scene_dir / "mast3r_sfm").expanduser().resolve()

    if not args.skip_mast3r_run:
        run_mast3r(args, mast3r_repo, mast3r_python, scene_dir, result_dir)
    package_scene(args, scene_dir, result_dir)


if __name__ == "__main__":
    main()
