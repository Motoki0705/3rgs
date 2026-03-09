#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.tennis_domain import annotation_path, build_manual_pair_matches, load_annotations


def load_train_stems(scene_dir: Path) -> list[str]:
    train_file = scene_dir / "images_train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} is missing. Run scripts/preprocess.py first.")
    return [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_train_names(scene_dir: Path, train_stems: list[str]) -> list[str]:
    image_dir = scene_dir / "images"
    suffix_by_stem = {path.stem: path.name for path in sorted(image_dir.iterdir()) if path.is_file()}
    missing = [stem for stem in train_stems if stem not in suffix_by_stem]
    if missing:
        raise FileNotFoundError(f"Missing train images under {image_dir}: {missing[:5]}")
    return [suffix_by_stem[stem] for stem in train_stems]


def ensure_resized_images(scene_dir: Path, factor: int, image_names: list[str]) -> Path:
    image_dir = scene_dir / ("images" if factor == 1 else f"images_{factor}")
    source_dir = scene_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in image_names:
        src = source_dir / name
        dst = image_dir / name
        if dst.exists():
            continue
        with Image.open(src) as image:
            width, height = image.size
            resized = image.resize((width // factor, height // factor), Image.Resampling.LANCZOS)
            resized.save(dst)
    return image_dir


def encode_xy(x: float, y: float, actual_width: int) -> int:
    scale = 512.0 / float(actual_width)
    x_512 = int(np.clip(np.rint(x * scale), 0, 511))
    y_512 = int(np.clip(np.rint(y * scale), 0, 511))
    return y_512 * 512 + x_512


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge manual annotation correspondences into MASt3R-generated epipolar tensors."
    )
    parser.add_argument("--scene_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--annotations_json", type=Path, default=None)
    parser.add_argument("--max_manual_correspondences", type=int, default=40)
    parser.add_argument("--manual_base_weight", type=float, default=1.0)
    args = parser.parse_args()

    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = (args.output_dir or scene_dir / "mast3r").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stems = load_train_stems(scene_dir)
    train_names = resolve_train_names(scene_dir, train_stems)
    image_dir = ensure_resized_images(scene_dir, args.data_factor, train_names)
    actual_sample = imageio.imread(image_dir / train_names[0])
    actual_height, actual_width = actual_sample.shape[:2]
    _ = actual_height

    corr_i = np.load(output_dir / "corr_i.npy")
    corr_j = np.load(output_dir / "corr_j.npy")
    corr_mask = np.load(output_dir / "corr_mask.npy")
    corr_weight = np.load(output_dir / "corr_weight.npy")
    corr_batch_idx = np.load(output_dir / "corr_batch_idx.npy")
    ei = np.load(output_dir / "ei.npy")
    ej = np.load(output_dir / "ej.npy")

    if (output_dir / "corr_is_manual.npy").exists():
        corr_is_manual = np.load(output_dir / "corr_is_manual.npy")
    else:
        corr_is_manual = np.zeros_like(corr_mask, dtype=np.float32)

    total_corr = corr_i.shape[1]
    if total_corr < args.max_manual_correspondences:
        raise ValueError(
            f"corr_i.npy only has {total_corr} slots, fewer than max_manual_correspondences={args.max_manual_correspondences}."
        )
    manual_offset = total_corr - args.max_manual_correspondences

    payload = load_annotations(annotation_path(scene_dir, args.annotations_json))

    for pair_idx, (left_idx, right_idx) in enumerate(zip(ei.tolist(), ej.tolist())):
        left_name = train_names[left_idx]
        right_name = train_names[right_idx]
        manual_matches = build_manual_pair_matches(payload, left_name, right_name)

        corr_i[pair_idx, manual_offset:] = 0
        corr_j[pair_idx, manual_offset:] = 0
        corr_mask[pair_idx, manual_offset:] = 0.0
        corr_weight[pair_idx, manual_offset:] = 0.0
        corr_is_manual[pair_idx, manual_offset:] = 0.0
        corr_batch_idx[pair_idx, manual_offset:] = 0

        for manual_idx, (_, xy_i, xy_j) in enumerate(manual_matches[: args.max_manual_correspondences]):
            slot = manual_offset + manual_idx
            corr_i[pair_idx, slot] = encode_xy(float(xy_i[0]), float(xy_i[1]), actual_width)
            corr_j[pair_idx, slot] = encode_xy(float(xy_j[0]), float(xy_j[1]), actual_width)
            corr_mask[pair_idx, slot] = 1.0
            corr_weight[pair_idx, slot] = args.manual_base_weight
            corr_is_manual[pair_idx, slot] = 1.0

    np.save(output_dir / "corr_i.npy", corr_i)
    np.save(output_dir / "corr_j.npy", corr_j)
    np.save(output_dir / "corr_batch_idx.npy", corr_batch_idx)
    np.save(output_dir / "corr_mask.npy", corr_mask)
    np.save(output_dir / "corr_weight.npy", corr_weight)
    np.save(output_dir / "corr_is_manual.npy", corr_is_manual)

    print(f"Updated epipolar data: {output_dir}")
    print(f"Train images: {len(train_names)}")
    print(f"Pairs: {len(ei)}")
    print(f"Manual correspondence slots per pair: {args.max_manual_correspondences}")


if __name__ == "__main__":
    main()
