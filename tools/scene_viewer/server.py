#!/usr/bin/env python3
"""
Scene Viewer Server
------------------
Supports two modes:

  --mode scene (default)
      Serves a web UI for visualizing:
      - 3DGS point cloud (sampled from checkpoint)
      - Original camera positions (blue)
      - Pose-refined camera positions (orange, MLP-based refinement)
      - Click on camera to see training image

      Usage:
          python tools/scene_viewer/server.py \
              --ckpt results/tennis_court_30k/ckpts/ckpt_29999_rank0.pt \
              --data-dir data/tennis_court \
              --port 8080

  --mode court-init
      Interactive Sim(3) initializer for fit_court_sim3.py.
      Displays MASt3R point cloud + cameras + two-court wireframe overlay.
      Allows manual adjustment of scale/rotation/translation/gap and saves
      the result as init_sim3.json.

      Usage:
          python tools/scene_viewer/server.py \
              --mode court-init \
              --scene-dir data/tennis_court \
              --port 8090

  --mode court-result
      Visualize the output of fit_court_sim3.py.
      Shows the MASt3R point cloud, original cameras (blue), refined
      cameras (orange), and two-court wireframes from sim3_refined.npz.

      Usage:
          python tools/scene_viewer/server.py \
              --mode court-result \
              --scene-dir data/tennis_court \
              --port 8092
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file, send_from_directory

# ─── Repo root paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(Path(__file__).parent))

# ─── Global cache ────────────────────────────────────────────────────────────
SCENE_DATA = None
COURT_DATA: dict | None = None          # court-init mode
COURT_INIT_OUTPUT_PATH: Path | None = None  # where to save init_sim3.json
COURT_RESULT_DATA: dict | None = None   # court-result mode


def current_view_data() -> dict | None:
    """Return whichever dataset is active for the current viewer mode."""
    for d in (SCENE_DATA, COURT_DATA, COURT_RESULT_DATA):
        if d is not None:
            return d
    return None


# ─── Math helpers ────────────────────────────────────────────────────────────
SH_C0 = 0.2820947917738781  # 1 / (2 * sqrt(pi))


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert zero-order SH to RGB.
    sh0: [N, 1, 3]  → rgb: [N, 3]  in range [0, 1]
    """
    rgb = sh0.squeeze(1) * SH_C0 + 0.5
    return rgb.clamp(0.0, 1.0)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)  # (..., 3, 3)


class CameraOptModuleMLP(torch.nn.Module):
    """Matches the training code in src/utils/cam_utils.py"""

    def __init__(self, n: int, mlp_width: int = 64, mlp_depth: int = 2, cam_scale: float = 1.0):
        super().__init__()
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.embeds = torch.nn.Embedding(n, mlp_width)
        self.cam_scale = cam_scale
        activation = torch.nn.ReLU(inplace=True)
        layers = []
        layers.append(torch.nn.Linear(mlp_width, mlp_width))
        layers.append(activation)
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(activation)
        layers.append(torch.nn.Linear(mlp_width, 9))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch_shape = camtoworlds.shape[:-2]
        embeddings = self.embeds(embed_ids)
        pose_deltas = self.mlp(embeddings)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch_shape, -1))
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx * self.cam_scale
        return torch.matmul(camtoworlds, transform)


# ─── Scene data loading ──────────────────────────────────────────────────────

def load_scene(ckpt_path: str, data_dir: str, n_points: int = 10000) -> dict:
    print(f"[scene_viewer] Loading checkpoint: {ckpt_path}")
    t0 = time.time()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Point cloud ──────────────────────────────────────────────────────────
    splats = ckpt["splats"]
    means = splats["means"]      # [N, 3]
    opacities = splats["opacities"]  # [N]
    sh0 = splats["sh0"]          # [N, 1, 3]

    # Sample by highest opacity
    sig_opacity = torch.sigmoid(opacities)
    n_total = means.shape[0]
    n_sample = min(n_points, n_total)
    top_idx = torch.argsort(sig_opacity, descending=True)[:n_sample]

    pts_xyz = means[top_idx].numpy()   # [n_sample, 3]
    pts_rgb = sh0_to_rgb(sh0[top_idx]).numpy()  # [n_sample, 3]

    print(f"[scene_viewer] Sampled {n_sample} / {n_total} Gaussian points")

    # ── Camera poses ─────────────────────────────────────────────────────────
    from datasets.mast3r import Parser, Dataset

    parser = Parser(data_dir, factor=1, normalize=True, test_every=8)
    trainset = Dataset(parser, split="train")
    testset = Dataset(parser, split="test")

    cam_scale = trainset.cam_scale

    # Reconstruct pose_adjust MLP
    n_train = len(trainset)   # 78
    pose_adjust = CameraOptModuleMLP(n_train, cam_scale=cam_scale)
    if "pose_adjust" in ckpt:
        pose_adjust.load_state_dict(ckpt["pose_adjust"])
    pose_adjust.eval()

    # Compute refined poses for training cameras
    train_camtoworlds_orig = torch.tensor(trainset.camtoworlds, dtype=torch.float32)  # [78, 4, 4]
    embed_ids = torch.arange(n_train)
    with torch.no_grad():
        train_camtoworlds_refined = pose_adjust(train_camtoworlds_orig, embed_ids)  # [78, 4, 4]

    # ── Build camera list ────────────────────────────────────────────────────
    # We collect all cameras (train + test) with their global image index
    cameras = []

    # All image names in sorted order (as returned by parser)
    all_names = parser.image_names       # sorted list of "stem.ext"
    all_camtoworlds = parser.camtoworlds  # [90, 4, 4] normalised

    # Build lookup: image stem → train index (0-based within trainset)
    train_name_to_idx = {}
    for local_i, global_i in enumerate(trainset.indices):
        stem = os.path.splitext(all_names[global_i])[0]
        train_name_to_idx[stem] = local_i

    for global_i, name in enumerate(all_names):
        stem = os.path.splitext(name)[0]
        c2w_orig = all_camtoworlds[global_i]   # [4, 4]
        K = parser.intrinsics[global_i]        # [3, 3]
        is_train = stem in train_name_to_idx

        cam_entry = {
            "image_name": name,
            "global_idx": global_i,
            "is_train": is_train,
            # Original pose (4×4, row-major)
            "c2w": c2w_orig.tolist(),
            # Intrinsics
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": parser.image_size[0],
            "height": parser.image_size[1],
        }

        if is_train:
            local_i = train_name_to_idx[stem]
            c2w_ref = train_camtoworlds_refined[local_i].numpy()
            cam_entry["c2w_refined"] = c2w_ref.tolist()
            cam_entry["train_idx"] = local_i

        cameras.append(cam_entry)

    print(f"[scene_viewer] {len(cameras)} cameras ({len(trainset)} train, {len(testset)} test)")
    print(f"[scene_viewer] Data loaded in {time.time() - t0:.1f}s")

    return {
        "points": {
            "xyz": pts_xyz.tolist(),
            "rgb": pts_rgb.tolist(),
        },
        "cameras": cameras,
        "image_dir": str(Path(data_dir) / "images"),
    }


# ─── React frontend dist path ────────────────────────────────────────────────
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def _ensure_frontend_dist() -> None:
    """Fail fast when the built React frontend is missing."""
    if not FRONTEND_DIST.exists():
        raise RuntimeError(
            f"Missing frontend build: {FRONTEND_DIST}. "
            "Run `npm install && npm run build` in tools/scene_viewer/frontend."
        )


def _serve_react_app():
    """Serve the React SPA index.html from the built frontend."""
    _ensure_frontend_dist()
    return send_from_directory(str(FRONTEND_DIST), "index.html")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve Vite build assets (JS/CSS chunks)."""
    return send_from_directory(str(FRONTEND_DIST / "assets"), filename)


# ─── Routes (scene mode) ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return _serve_react_app()


@app.route("/api/scene")
def api_scene():
    if SCENE_DATA is None:
        return jsonify({"error": "scene mode not active"}), 503
    data = {
        "points": SCENE_DATA["points"],
        "cameras": SCENE_DATA["cameras"],
    }
    return jsonify(data)


@app.route("/api/image/<int:img_idx>")
def api_image(img_idx: int):
    data = current_view_data()
    if data is None:
        return "Viewer data is not loaded", 503

    cameras = data["cameras"]
    if img_idx < 0 or img_idx >= len(cameras):
        return "Not found", 404
    image_dir = data.get("image_dir")
    if not image_dir:
        return "Image directory is not configured", 404
    name = cameras[img_idx]["image_name"]
    return send_from_directory(image_dir, name)


# ─── Routes (court-result mode) ─────────────────────────────────────────────

@app.route("/court_result")
def court_result_page():
    return _serve_react_app()


@app.route("/api/court_result_scene")
def api_court_result_scene():
    """Return court alignment result data for visualization."""
    if COURT_RESULT_DATA is None:
        return jsonify({"error": "court-result mode not active"}), 503
    return jsonify(COURT_RESULT_DATA)


# ─── Routes (court-init mode) ─────────────────────────────────────────────────

@app.route("/court_init")
def court_init_page():
    return _serve_react_app()


@app.route("/api/court_scene")
def api_court_scene():
    """Return MASt3R point cloud, cameras, initial Sim(3), and court skeleton."""
    if COURT_DATA is None:
        return jsonify({"error": "court-init mode not active"}), 503
    return jsonify(COURT_DATA)


@app.route("/api/save_sim3", methods=["POST"])
def api_save_sim3():
    """Save the Sim(3) parameters sent by the UI to init_sim3.json."""
    if COURT_INIT_OUTPUT_PATH is None:
        return jsonify({"error": "output path not configured"}), 503
    body = request.get_json(force=True)
    required_keys = {"scale", "rotation", "translation", "adjacent_gap", "adjacent_direction"}
    missing = required_keys - set(body.keys())
    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 400

    sim3 = {
        "scale": float(body["scale"]),
        "rotation": body["rotation"],
        "translation": body["translation"],
        "adjacent_gap": float(body["adjacent_gap"]),
        "adjacent_direction": str(body["adjacent_direction"]),
    }
    COURT_INIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COURT_INIT_OUTPUT_PATH.write_text(json.dumps(sim3, indent=2), encoding="utf-8")
    print(f"[court_init] Saved Sim(3) → {COURT_INIT_OUTPUT_PATH}", flush=True)
    return jsonify({"ok": True, "path": str(COURT_INIT_OUTPUT_PATH)})


@app.route("/api/load_sim3")
def api_load_sim3():
    """Load a previously saved init_sim3.json (or return the auto-estimated values)."""
    if COURT_INIT_OUTPUT_PATH is not None and COURT_INIT_OUTPUT_PATH.exists():
        sim3 = json.loads(COURT_INIT_OUTPUT_PATH.read_text(encoding="utf-8"))
        return jsonify({"ok": True, "sim3": sim3, "source": "saved"})
    # Fall back to the auto-estimated initial values
    if COURT_DATA is not None:
        sim3 = {
            "scale": COURT_DATA["scale"],
            "rotation": COURT_DATA["rotation"],
            "translation": COURT_DATA["translation"],
            "adjacent_gap": COURT_DATA["adjacent_gap"],
            "adjacent_direction": COURT_DATA["adjacent_direction"],
        }
        return jsonify({"ok": True, "sim3": sim3, "source": "auto"})
    return jsonify({"error": "no data"}), 503


# ─── Court-result data loader ────────────────────────────────────────────────

def load_court_result(scene_dir: Path, align_dir: Path, n_sample: int = 50_000) -> dict:
    """Load fit_court_sim3.py output for visualization."""
    t0 = time.time()
    mast3r_dir = scene_dir / "mast3r"

    # Point cloud via fast numpy reader
    estimator_dir = Path(__file__).resolve().parent / "utils"
    if str(estimator_dir) not in sys.path:
        sys.path.insert(0, str(estimator_dir))
    from court_init_estimator import CourtInitEstimator  # noqa: PLC0415

    est = CourtInitEstimator(mast3r_dir)
    pts_xyz, pts_rgb = est._load_point_cloud(n_sample)

    # Original camera poses + intrinsics (MASt3R)
    orig_poses = np.load(mast3r_dir / "camera_poses.npy")    # (N, 4, 4)
    intrinsics = np.load(mast3r_dir / "camera_intrinsics.npy")  # (N, 3, 3)
    n_cams = orig_poses.shape[0]

    # Refined camera poses
    ref_poses = np.load(align_dir / "camera_poses_refined.npy")  # (N, 4, 4)

    # Sim3 (refined)
    npz = np.load(align_dir / "sim3_refined.npz")
    sim3 = {
        "scale": float(npz["scale"]),
        "rotation": npz["rotation"].tolist(),
        "translation": npz["translation"].tolist(),
        "matrix": npz["matrix"].tolist(),
        "adjacent_gap": float(npz["adjacent_gap"]),
        "adjacent_direction": npz["adjacent_direction"].tolist(),
    }

    # Metrics
    metrics = json.loads((align_dir / "metrics.json").read_text(encoding="utf-8"))
    court_pair = metrics.get("court_pair", {})
    train_stems = metrics.get("train_stems", [])

    # Image names
    img_list_path = scene_dir / "images_train.txt"
    if img_list_path.exists():
        stems = img_list_path.read_text(encoding="utf-8").splitlines()
    else:
        stems = train_stems

    # Camera list with per-camera refinement delta
    from utils.court_scheme import COURT_SKELETON, court_keypoints_3d  # noqa: PLC0415
    cameras = []
    for i in range(n_cams):
        K = intrinsics[i]
        stem = stems[i] if i < len(stems) else f"cam_{i:04d}"
        orig_t = orig_poses[i, :3, 3]
        ref_t  = ref_poses[i, :3, 3]
        delta  = float(np.linalg.norm(ref_t - orig_t))
        cameras.append({
            "image_name": stem + ".jpg",
            "idx": i,
            "c2w": orig_poses[i].tolist(),
            "c2w_refined": ref_poses[i].tolist(),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width":  int(round(K[0, 2] * 2)),
            "height": int(round(K[1, 2] * 2)),
            "delta": delta,
        })

    kp_court = court_keypoints_3d().cpu().numpy().tolist()

    print(f"[court_result] Loaded {n_cams} cameras in {time.time()-t0:.1f}s", flush=True)
    return {
        "point_cloud": {
            "xyz": pts_xyz.tolist(),
            "rgb": pts_rgb.tolist() if pts_rgb is not None else [],
        },
        "cameras": cameras,
        "sim3": sim3,
        "court_pair": court_pair,
        "court_keypoints_court": kp_court,
        "court_skeleton": COURT_SKELETON,
        "metrics_summary": {
            "num_cameras": metrics.get("num_train_cameras", n_cams),
            "pose_trans_sigma": float(metrics.get("pose_trans_sigma", 0.0)),
            "adjacent_gap": court_pair.get("adjacent_gap", 0.0),
            "adjacent_direction": court_pair.get("adjacent_court_direction", "+x"),
        },
        "image_dir": str(scene_dir / "images"),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    global SCENE_DATA, COURT_DATA, COURT_INIT_OUTPUT_PATH, COURT_RESULT_DATA

    _ensure_frontend_dist()

    parser = argparse.ArgumentParser(description="Scene Viewer Server")
    parser.add_argument(
        "--mode",
        choices=("scene", "court-init", "court-result"),
        default="scene",
        help="Viewer mode: 'scene' for 3DGS checkpoint viewer, "
             "'court-init' for interactive Sim(3) initializer, "
             "'court-result' for fit_court_sim3.py output visualization",
    )
    parser.add_argument(
        "--ckpt",
        default="results/tennis_court_30k/ckpts/ckpt_29999_rank0.pt",
        help="[scene mode] Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--data-dir",
        default="data/tennis_court",
        help="[scene mode] Path to dataset directory",
    )
    parser.add_argument(
        "--scene-dir",
        default="data/tennis_court",
        help="[court-init mode] Path to scene directory (contains mast3r/ sub-dir)",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument(
        "--n-points", type=int, default=10000, help="[scene mode] Number of point cloud points to display"
    )
    parser.add_argument(
        "--n-sample", type=int, default=50_000,
        help="[court-init mode] Number of point cloud points to sub-sample from PLY",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="[court-init mode] Output path for init_sim3.json. "
             "Defaults to <scene-dir>/court/transform/init_sim3.json",
    )
    parser.add_argument(
        "--adjacent-direction",
        choices=("+x", "-x", "+y", "-y"),
        default="+x",
        help="[court-init mode] Direction of the second court in base court frame",
    )
    parser.add_argument(
        "--init-gap",
        type=float,
        default=3.0,
        help="[court-init mode] Initial edge-to-edge gap between adjacent courts (m)",
    )
    parser.add_argument(
        "--align-dir",
        default=None,
        help="[court-result mode] Path to court_alignment directory containing "
             "sim3_refined.npz, camera_poses_refined.npy etc. "
             "Defaults to <scene-dir>/mast3r/court_alignment",
    )
    args = parser.parse_args()

    if args.mode == "scene":
        # Resolve paths relative to repo root
        ckpt_path = str(REPO_ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt
        data_dir = str(REPO_ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() else args.data_dir
        SCENE_DATA = load_scene(ckpt_path, data_dir, n_points=args.n_points)
        print(f"[scene_viewer] Starting server at http://localhost:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)

    elif args.mode == "court-init":
        # Lazy import to avoid loading torch/datasets when not needed
        scene_dir = Path(args.scene_dir)
        if not scene_dir.is_absolute():
            scene_dir = REPO_ROOT / scene_dir
        scene_dir = scene_dir.resolve()
        mast3r_dir = scene_dir / "mast3r"

        # Output path
        if args.output is not None:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = REPO_ROOT / out_path
        else:
            out_path = scene_dir / "court" / "transform" / "init_sim3.json"
        COURT_INIT_OUTPUT_PATH = out_path.resolve()

        # Run estimator
        estimator_dir = Path(__file__).resolve().parent / "utils"
        if str(estimator_dir) not in sys.path:
            sys.path.insert(0, str(estimator_dir))
        from court_init_estimator import CourtInitEstimator  # noqa: PLC0415

        est = CourtInitEstimator(mast3r_dir)
        print(f"[court_init] Estimating initial Sim(3) from {mast3r_dir} …", flush=True)
        t0 = time.time()
        COURT_DATA = est.estimate(
            n_sample=args.n_sample,
            adjacent_direction=args.adjacent_direction,
            init_gap=args.init_gap,
        )
        print(f"[court_init] Done in {time.time() - t0:.1f}s  "
              f"scale={COURT_DATA['scale']:.4f}  "
              f"n_pts={len(COURT_DATA['point_cloud']['xyz'])}", flush=True)
        print(f"[court_init] Starting server at http://localhost:{args.port}/court_init")
        app.run(host="0.0.0.0", port=args.port, debug=False)

    elif args.mode == "court-result":
        scene_dir = Path(args.scene_dir)
        if not scene_dir.is_absolute():
            scene_dir = REPO_ROOT / scene_dir
        scene_dir = scene_dir.resolve()

        if args.align_dir is not None:
            align_dir = Path(args.align_dir)
            if not align_dir.is_absolute():
                align_dir = REPO_ROOT / align_dir
        else:
            align_dir = scene_dir / "mast3r" / "court_alignment"
        align_dir = align_dir.resolve()

        print(f"[court_result] Loading alignment results from {align_dir} …", flush=True)
        COURT_RESULT_DATA = load_court_result(scene_dir, align_dir, n_sample=args.n_sample)
        print(f"[court_result] Starting server at http://localhost:{args.port}/court_result")
        app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
