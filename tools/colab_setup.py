#!/usr/bin/env python3
"""Prepare a cloned 3rgs checkout for Colab training with court pose refinement."""

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
    "preprocess_mast3r",
    "fit_court_ground",
    "optimize_camera_pose",
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
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(repo_dir / "third_party" / "mast3r" / "dust3r" / "requirements.txt"),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(repo_dir / "third_party" / "mast3r" / "requirements.txt"),
        ]
    )
    ensure_mast3r_python_shim(repo_dir)


def ensure_mast3r_python_shim(repo_dir: Path) -> Path:
    mast3r_repo = repo_dir / "third_party" / "mast3r"
    bin_dir = mast3r_repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target_python = Path(sys.executable).resolve()
    for name in ("python", "python3"):
        link_path = bin_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_python)
    print(f"Created MASt3R python shim: {bin_dir / 'python'} -> {target_python}", flush=True)
    return bin_dir / "python"


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
    if (scene_dir / "images").exists():
        return scene_dir
    children = [path for path in scene_dir.iterdir() if path.is_dir()]
    if len(children) == 1:
        child = children[0]
        if (child / "images").exists():
            print(f"Using nested scene root: {child}", flush=True)
            return child
    return scene_dir


def validate_training_layout(scene_dir: Path) -> Path:
    scene_root = resolve_scene_root(scene_dir)
    required = [
        "images",
        "court",
        "court/line",
        "court/line/court_line_masks.npy",
    ]
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


def has_preprocessed_mast3r(scene_root: Path) -> bool:
    required = [
        scene_root / "images_train.txt",
        scene_root / "images_test.txt",
        scene_root / "pose_gt_train.npy",
        scene_root / "pose_gt_test.npy",
        scene_root / "mast3r" / "camera_intrinsics.npy",
        scene_root / "mast3r" / "camera_poses.npy",
        scene_root / "mast3r" / "pointcloud.ply",
        scene_root / "mast3r" / "corr_i.npy",
        scene_root / "mast3r" / "corr_j.npy",
        scene_root / "mast3r" / "corr_mask.npy",
        scene_root / "mast3r" / "corr_weight.npy",
        scene_root / "mast3r" / "corr_batch_idx.npy",
        scene_root / "mast3r" / "ei.npy",
        scene_root / "mast3r" / "ej.npy",
        scene_root / "mast3r" / "depthmaps.npy",
    ]
    return all(path.exists() for path in required)


def has_ground_fit_outputs(scene_root: Path) -> bool:
    required = [
        scene_root / "court" / "ground" / "plane_frame.json",
        scene_root / "court" / "ground" / "projected_train.npz",
        scene_root / "court" / "ground" / "visibility_train.npz",
        scene_root / "court" / "transform" / "ground_heatmap_fit.json",
        scene_root / "court" / "transform" / "ground_heatmap_fit_sim3.json",
    ]
    return all(path.exists() for path in required)


def has_pose_opt_outputs(scene_root: Path) -> bool:
    required = [
        scene_root / "court" / "pose_opt" / "optimized_camera_poses.npy",
        scene_root / "court" / "pose_opt" / "camera_pose_opt.json",
    ]
    return all(path.exists() for path in required)


def run_mast3r_preprocess(repo_dir: Path, data_link: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/preprocess.py",
            "--scene_dir",
            str(data_link),
            "--test_every",
            "8",
            "--shared_intrinsics",
        ],
        cwd=repo_dir,
    )


def run_court_ground_fit(repo_dir: Path, data_link: Path) -> None:
    run(
        [
            sys.executable,
            "tools/court_ground_fit/project_court_lines_to_ground.py",
            "--scene-dir",
            str(data_link),
        ],
        cwd=repo_dir,
    )
    run(
        [
            sys.executable,
            "tools/court_ground_fit/fit_from_ground_heatmap.py",
            "--scene-dir",
            str(data_link),
        ],
        cwd=repo_dir,
    )


def run_pose_optimization(repo_dir: Path, data_link: Path) -> None:
    run(
        [
            sys.executable,
            "tools/court_ground_fit/optimize_camera_poses_to_fixed_sim3.py",
            "--scene-dir",
            str(data_link),
        ],
        cwd=repo_dir,
    )


def install_optimized_training_poses(scene_root: Path) -> Path:
    optimized_path = scene_root / "court" / "pose_opt" / "optimized_camera_poses.npy"
    target_path = scene_root / "mast3r" / "camera_poses.npy"
    if not optimized_path.exists():
        raise FileNotFoundError(f"Optimized camera poses not found: {optimized_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Replacing training poses: {optimized_path} -> {target_path}", flush=True)
    shutil.copy2(optimized_path, target_path)
    return target_path


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
    parser.add_argument("--repo-data-link-name", default="data_scene")
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
    if scene_root is not None and end_idx >= PHASES.index("preprocess_mast3r"):
        data_link = ensure_repo_link(repo_dir, scene_root, args.repo_data_link_name)

    if should_run("preprocess_mast3r"):
        if scene_root is None or data_link is None:
            raise RuntimeError("Scene root is unavailable. Run extract_tar first.")
        enter_phase("preprocess_mast3r", "running MASt3R preprocessing and epipolar preparation")
        if has_preprocessed_mast3r(scene_root):
            print(f"Found MASt3R preprocessing outputs under {scene_root}; skipping preprocess.py", flush=True)
        else:
            run_mast3r_preprocess(repo_dir, data_link)

    if should_run("fit_court_ground"):
        if scene_root is None or data_link is None:
            raise RuntimeError("Scene root is unavailable. Run extract_tar first.")
        enter_phase("fit_court_ground", "running court ground-fit pipeline")
        if has_ground_fit_outputs(scene_root):
            print(f"Found court ground-fit outputs under {scene_root}; skipping court_ground_fit pipeline", flush=True)
        else:
            run_court_ground_fit(repo_dir, data_link)

    if should_run("optimize_camera_pose"):
        if scene_root is None or data_link is None:
            raise RuntimeError("Scene root is unavailable. Run extract_tar first.")
        enter_phase("optimize_camera_pose", "optimizing camera poses to fixed court Sim(3)")
        if has_pose_opt_outputs(scene_root):
            print(f"Found pose optimization outputs under {scene_root}; skipping optimize_camera_poses_to_fixed_sim3.py", flush=True)
        else:
            run_pose_optimization(repo_dir, data_link)

    if should_run("verify_runtime"):
        enter_phase("verify_runtime", "verifying torch, CUDA, and package imports")
        verify_runtime(repo_dir)

    if should_run("train"):
        if data_link is None:
            raise RuntimeError("Data link is unavailable. Run extract_tar first.")
        if scene_root is None:
            raise RuntimeError("Scene root is unavailable. Run extract_tar first.")
        enter_phase("train", "running trainer.py")
        installed_pose_path = install_optimized_training_poses(scene_root)
        print(f"Training will use poses from {installed_pose_path}", flush=True)
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
