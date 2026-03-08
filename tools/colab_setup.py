#!/usr/bin/env python3
"""Prepare a cloned 3rgs checkout for Colab training, including epipolar data."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/content/data")
DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_RESULT_ROOT = Path("/content/drive/MyDrive/3rgs_runs")
DEFAULT_PYTHON_PACKAGES = [
    "pip",
    "setuptools",
    "wheel",
    "ninja",
]
PHASES = [
    "mount_drive",
    "python_deps",
    "copy_tar",
    "extract_tar",
    "prepare_scene",
    "prepare_epipolar",
    "verify_runtime",
    "train",
]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    location = f" (cwd={cwd})" if cwd else ""
    print(f"+ {' '.join(cmd)}{location}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def print_phase(current: int, total: int, name: str, detail: str) -> None:
    print(f"\n[{current}/{total}] {name}: {detail}", flush=True)


def mount_drive(mount_point: Path, force_remount: bool) -> None:
    if (mount_point / "MyDrive").exists() and not force_remount:
        print(f"Drive already mounted at {mount_point}", flush=True)
        return

    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google.colab is unavailable. Run this script inside Colab.") from exc

    mount_point.mkdir(parents=True, exist_ok=True)
    drive.mount(str(mount_point), force_remount=force_remount)


def install_python_deps(repo_dir: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", *DEFAULT_PYTHON_PACKAGES])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.10.0",
            "torchvision==0.25.0",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")])


def copy_tar_to_local(src_tar: Path, local_tar: Path, overwrite: bool) -> None:
    if not src_tar.exists():
        raise FileNotFoundError(f"Tar file not found: {src_tar}")
    local_tar.parent.mkdir(parents=True, exist_ok=True)
    if local_tar.exists():
        if overwrite:
            local_tar.unlink()
        else:
            print(f"Local tar already exists: {local_tar}", flush=True)
            return
    print(f"Copying tar to local storage: {src_tar} -> {local_tar}", flush=True)
    shutil.copy2(src_tar, local_tar)


def extract_scene_tar(local_tar: Path, data_root: Path, scene_name: str, overwrite: bool) -> Path:
    target_dir = data_root / scene_name
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Scene already extracted at {target_dir}", flush=True)
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xf", str(local_tar), "-C", str(data_root)])
    return target_dir


def resolve_scene_root(scene_dir: Path) -> Path:
    if (scene_dir / "images").exists() and (scene_dir / "sparse" / "0").exists():
        return scene_dir
    children = [path for path in scene_dir.iterdir() if path.is_dir()]
    if len(children) == 1:
        child = children[0]
        if (child / "images").exists() and (child / "sparse" / "0").exists():
            print(f"Using nested scene root: {child}", flush=True)
            return child
    return scene_dir


def validate_training_layout(scene_dir: Path) -> Path:
    scene_root = resolve_scene_root(scene_dir)
    required = ["images", "sparse/0"]
    missing = [rel for rel in required if not (scene_root / rel).exists()]
    if missing:
        raise RuntimeError(f"Extracted scene is missing required paths: {missing}")
    return scene_root


def ensure_repo_link(repo_dir: Path, scene_root: Path, link_name: str) -> Path:
    link_path = repo_dir / link_name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(scene_root)
    return link_path


def verify_runtime(repo_dir: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    code = """
import torch
import pycolmap
import gsplat
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('torch_cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
"""
    run([sys.executable, "-c", code], cwd=repo_dir, env=env)


def train(repo_dir: Path, data_link: Path, args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir).expanduser()
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "src/trainer.py",
        args.train_mode,
        "--data_dir",
        str(data_link),
        "--data_factor",
        str(args.data_factor),
        "--result_dir",
        str(result_dir),
        "--pose_opt_type",
        args.pose_opt_type,
        "--max_steps",
        str(args.max_steps),
        "--save_steps",
        *[str(step) for step in args.save_steps],
        "--eval_steps",
        *[str(step) for step in args.eval_steps],
        "--tb_every",
        str(args.tb_every),
    ]
    if args.use_epipolar_loss:
        cmd.append("--use-corres-epipolar-loss")
    else:
        cmd.append("--no-use-corres-epipolar-loss")
    for extra in args.extra_train_arg:
        cmd.extend(extra.split())
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("MPLBACKEND", "Agg")
    run(cmd, cwd=repo_dir, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-tar-path", required=True, help="Path to the dataset tar on mounted Drive.")
    parser.add_argument("--scene-name", required=True, help="Scene name under /content/data/<scene_name>.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--drive-mount-point", default=str(DEFAULT_DRIVE_MOUNT))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_ROOT / "tennis_court_colab"))
    parser.add_argument("--repo-data-link-name", default="data_tennis_court")
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--pair-window", type=int, default=2)
    parser.add_argument("--min-shared-points", type=int, default=64)
    parser.add_argument("--num-correspondences", type=int, default=256)
    parser.add_argument("--train-mode", choices=["default", "mcmc"], default="default")
    parser.add_argument("--pose-opt-type", choices=["sfm", "mlp"], default="sfm")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--save-steps", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--tb-every", type=int, default=100)
    parser.add_argument("--use-epipolar-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-train-arg", action="append", default=[], help="Extra trainer CLI fragment.")
    parser.add_argument("--skip-drive-mount", action="store_true")
    parser.add_argument("--force-remount-drive", action="store_true")
    parser.add_argument("--overwrite-tar", action="store_true")
    parser.add_argument("--overwrite-scene", action="store_true")
    parser.add_argument("--start-phase", choices=PHASES, default=PHASES[0])
    parser.add_argument("--end-phase", choices=PHASES, default=PHASES[-1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve()
    drive_mount_point = Path(args.drive_mount_point).resolve()
    drive_tar_path = Path(args.drive_tar_path)
    if not drive_tar_path.is_absolute():
        drive_tar_path = (drive_mount_point / drive_tar_path).resolve()
    local_tar = Path("/content") / drive_tar_path.name

    start_idx = PHASES.index(args.start_phase)
    end_idx = PHASES.index(args.end_phase)
    if start_idx > end_idx:
        raise ValueError("--start-phase must be earlier than or equal to --end-phase.")

    def should_run(phase: str) -> bool:
        idx = PHASES.index(phase)
        return start_idx <= idx <= end_idx

    selected_phases = [phase for phase in PHASES if should_run(phase)]
    total_phases = len(selected_phases)
    phase_counter = 0

    def enter_phase(name: str, detail: str) -> None:
        nonlocal phase_counter
        phase_counter += 1
        print_phase(phase_counter, total_phases, name, detail)

    if should_run("mount_drive") and not args.skip_drive_mount:
        enter_phase("mount_drive", f"mounting Drive at {drive_mount_point}")
        mount_drive(drive_mount_point, args.force_remount_drive)

    if should_run("python_deps"):
        enter_phase("python_deps", "installing Python dependencies")
        install_python_deps(repo_dir)

    if should_run("copy_tar"):
        enter_phase("copy_tar", f"copying archive from {drive_tar_path}")
        copy_tar_to_local(drive_tar_path, local_tar, args.overwrite_tar)

    scene_root: Path | None = None
    if should_run("extract_tar"):
        enter_phase("extract_tar", f"extracting archive into {data_root}")
        extract_scene_tar(local_tar, data_root, args.scene_name, args.overwrite_scene)
    if end_idx >= PHASES.index("extract_tar"):
        scene_root = validate_training_layout(data_root / args.scene_name)

    data_link: Path | None = None
    if should_run("prepare_scene"):
        if scene_root is None:
            raise RuntimeError("Scene root is unavailable. Run extract_tar first.")
        enter_phase("prepare_scene", "creating MASt3R-style metadata and repo data link")
        data_link = ensure_repo_link(repo_dir, scene_root, args.repo_data_link_name)
        run(
            [
                sys.executable,
                "scripts/prepare_colmap_scene.py",
                "--scene_dir",
                str(data_link),
                "--test_every",
                "8",
            ],
            cwd=repo_dir,
        )
    elif end_idx >= PHASES.index("prepare_scene") and scene_root is not None:
        data_link = ensure_repo_link(repo_dir, scene_root, args.repo_data_link_name)

    if should_run("prepare_epipolar"):
        if data_link is None:
            raise RuntimeError("Data link is unavailable. Run prepare_scene first.")
        enter_phase("prepare_epipolar", "building epipolar correspondence tensors")
        run(
            [
                sys.executable,
                "scripts/prepare_epipolar_data.py",
                "--scene_dir",
                str(data_link),
                "--data_factor",
                str(args.data_factor),
                "--pair_window",
                str(args.pair_window),
                "--min_shared_points",
                str(args.min_shared_points),
                "--num_correspondences",
                str(args.num_correspondences),
            ],
            cwd=repo_dir,
        )

    if should_run("verify_runtime"):
        enter_phase("verify_runtime", "verifying torch, CUDA, and package imports")
        verify_runtime(repo_dir)

    if should_run("train"):
        if data_link is None:
            raise RuntimeError("Data link is unavailable. Run prepare_scene first.")
        enter_phase("train", "running trainer.py")
        train(repo_dir, data_link, args)

    print("\nSetup complete.", flush=True)
    print(f"Repository: {repo_dir}", flush=True)
    if scene_root is not None:
        print(f"Scene path: {scene_root}", flush=True)
    if data_link is not None:
        print(f"Repo data link: {data_link}", flush=True)
    print(f"Completed phases: {args.start_phase} -> {args.end_phase}", flush=True)


if __name__ == "__main__":
    main()
