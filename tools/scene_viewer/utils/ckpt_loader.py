#!/usr/bin/env python3
"""
ckpt_loader.py
--------------
Load scene data from a 3DGS training checkpoint (.pt).

Outputs JSON to stdout:
{
  "mode": "ckpt",
  "points": {"xyz": [[x,y,z],...], "rgb": [[r,g,b],...]},
  "cameras": [{"image_name":..., "global_idx":..., "is_train":...,
               "c2w":[[4x4]], "c2w_refined":[[4x4]] (train only),
               "fx","fy","cx","cy","width","height"}, ...],
  "image_dir": "/abs/path/to/images"
}

Usage:
    python tools/scene_viewer/utils/ckpt_loader.py \
        --ckpt results/tennis_court_30k/ckpts/ckpt_29999_rank0.pt \
        --data-dir data/tennis_court \
        --n-points 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ─── Repo root ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

SH_C0 = 0.2820947917738781


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return (sh0.squeeze(1) * SH_C0 + 0.5).clamp(0.0, 1.0)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


class CameraOptModuleMLP(torch.nn.Module):
    def __init__(self, n: int, mlp_width: int = 64, mlp_depth: int = 2, cam_scale: float = 1.0):
        super().__init__()
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.embeds = torch.nn.Embedding(n, mlp_width)
        self.cam_scale = cam_scale
        layers = []
        layers.append(torch.nn.Linear(mlp_width, mlp_width))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
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


def load(ckpt_path: str, data_dir: str, n_points: int = 10000) -> dict:
    print(f"[ckpt_loader] Loading {ckpt_path}", file=sys.stderr)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Point cloud ──────────────────────────────────────────────────────────
    splats = ckpt["splats"]
    means = splats["means"]
    opacities = splats["opacities"]
    sh0 = splats["sh0"]

    sig_opacity = torch.sigmoid(opacities)
    n_sample = min(n_points, means.shape[0])
    top_idx = torch.argsort(sig_opacity, descending=True)[:n_sample]

    pts_xyz = means[top_idx].numpy()
    pts_rgb = sh0_to_rgb(sh0[top_idx]).numpy()
    print(f"[ckpt_loader] {n_sample} points sampled", file=sys.stderr)

    # ── Camera poses ─────────────────────────────────────────────────────────
    from datasets.mast3r import Parser, Dataset

    parser = Parser(data_dir, factor=1, normalize=True, test_every=8)
    trainset = Dataset(parser, split="train")
    testset = Dataset(parser, split="test")

    cam_scale = trainset.cam_scale
    n_train = len(trainset)

    pose_adjust = CameraOptModuleMLP(n_train, cam_scale=cam_scale)
    if "pose_adjust" in ckpt:
        pose_adjust.load_state_dict(ckpt["pose_adjust"])
    pose_adjust.eval()

    train_c2w_orig = torch.tensor(trainset.camtoworlds, dtype=torch.float32)
    embed_ids = torch.arange(n_train)
    with torch.no_grad():
        train_c2w_refined = pose_adjust(train_c2w_orig, embed_ids)

    # Build name → local train idx
    all_names = parser.image_names
    train_name_to_idx = {}
    for local_i, global_i in enumerate(trainset.indices):
        stem = os.path.splitext(all_names[global_i])[0]
        train_name_to_idx[stem] = local_i

    cameras = []
    for global_i, name in enumerate(all_names):
        stem = os.path.splitext(name)[0]
        K = parser.intrinsics[global_i]
        c2w = parser.camtoworlds[global_i]
        is_train = stem in train_name_to_idx

        entry = {
            "image_name": name,
            "global_idx": global_i,
            "is_train": is_train,
            "c2w": c2w.tolist(),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": parser.image_size[0],
            "height": parser.image_size[1],
        }
        if is_train:
            local_i = train_name_to_idx[stem]
            entry["c2w_refined"] = train_c2w_refined[local_i].numpy().tolist()
            entry["train_idx"] = local_i

        cameras.append(entry)

    print(f"[ckpt_loader] {len(cameras)} cameras ({n_train} train)", file=sys.stderr)

    return {
        "mode": "ckpt",
        "points": {
            "xyz": pts_xyz.tolist(),
            "rgb": pts_rgb.tolist(),
        },
        "cameras": cameras,
        "image_dir": str(Path(data_dir).resolve() / "images"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-points", type=int, default=10000)
    args = parser.parse_args()

    data = load(args.ckpt, args.data_dir, args.n_points)
    # Output JSON to stdout (compact, no indent)
    json.dump(data, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
