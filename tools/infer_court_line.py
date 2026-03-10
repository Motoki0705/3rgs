#!/usr/bin/env python3
"""Run court white-line inference on a directory of images.

Preprocessing matches `experiments/court_detection/configs/line.py` validation:
- resize so the short side is 288
- keep aspect ratio
- round height/width down to multiples of 8
- normalize with ImageNet mean/std

Outputs are written under the requested output directory:
- `masks/<stem>.png`: binary mask at the original image resolution
- `manifest.json`: run metadata

If a scene directory is provided, the script also writes:
- `mast3r/court_line_masks.npy`: cleaned train-split masks stacked in
  `images_train.txt` order
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from numpy.lib.format import open_memmap


TENNIS_LAB_ROOT = Path("/root/repos/tennis-lab")
if str(TENNIS_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(TENNIS_LAB_ROOT))

from experiments.court_detection.models.court_unet import CourtUNet  # noqa: E402


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferenceConfig:
    image_dir: Path
    checkpoint: Path
    output_dir: Path
    scene_dir: Path | None = None
    mast3r_output_path: Path | None = None
    batch_size: int = 32
    short_side: int = 288
    threshold: float = 0.5
    num_workers: int = min(8, max(2, (os.cpu_count() or 4) // 2))
    prefetch_factor: int = 4
    save_workers: int = 4
    device: str = "cuda"
    fallback_last_on_corrupt: bool = True


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/root/repos/3rgs/data/tennis_court/images"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/repos/tennis-lab/outputs/experiments/court_detection/line/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/repos/3rgs/results"),
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("/root/repos/3rgs/data/tennis_court"),
        help="Scene root containing images_train.txt and mast3r/.",
    )
    parser.add_argument(
        "--mast3r-output-path",
        type=Path,
        default=None,
        help="Override the .npy output path. Defaults to <scene-dir>/mast3r/court_line_masks.npy.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--short-side", type=int, default=288)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-fallback-last-on-corrupt",
        action="store_true",
        help="Do not fall back to sibling last.pt if the requested checkpoint is corrupted.",
    )
    args = parser.parse_args()
    return InferenceConfig(
        image_dir=args.image_dir.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        scene_dir=args.scene_dir.expanduser().resolve() if args.scene_dir is not None else None,
        mast3r_output_path=(
            args.mast3r_output_path.expanduser().resolve()
            if args.mast3r_output_path is not None
            else None
        ),
        batch_size=args.batch_size,
        short_side=args.short_side,
        threshold=args.threshold,
        num_workers=max(1, args.num_workers),
        prefetch_factor=max(2, args.prefetch_factor),
        save_workers=max(1, args.save_workers),
        device=args.device,
        fallback_last_on_corrupt=not args.no_fallback_last_on_corrupt,
    )


def compute_resized_hw(height: int, width: int, short_side: int) -> tuple[int, int]:
    if height <= width:
        new_h = short_side
        new_w = int(round(width * new_h / height))
    else:
        new_w = short_side
        new_h = int(round(height * new_w / width))
    new_h = max(8, (new_h // 8) * 8)
    new_w = max(8, (new_w // 8) * 8)
    return new_h, new_w


class CourtLineInferenceDataset(Dataset):
    def __init__(self, image_dir: Path, short_side: int) -> None:
        self.image_dir = image_dir
        self.short_side = short_side
        self.paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        orig_h, orig_w = image_bgr.shape[:2]
        resized_h, resized_w = compute_resized_hw(orig_h, orig_w, self.short_side)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        image = image_rgb.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "image_id": path.stem,
            "path": str(path),
            "orig_size": (orig_h, orig_w),
            "infer_size": (resized_h, resized_w),
        }


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images = []
    image_ids = []
    paths = []
    orig_sizes = []
    infer_sizes = []
    for item in batch:
        image = item["image"]
        _, height, width = image.shape
        padded = torch.zeros((3, max_h, max_w), dtype=image.dtype)
        padded[:, :height, :width] = image
        images.append(padded)
        image_ids.append(item["image_id"])
        paths.append(item["path"])
        orig_sizes.append(item["orig_size"])
        infer_sizes.append(item["infer_size"])

    return {
        "image": torch.stack(images, dim=0),
        "image_id": image_ids,
        "path": paths,
        "orig_size": orig_sizes,
        "infer_size": infer_sizes,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {path}. "
            "The file may be incomplete or corrupted."
        ) from exc
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    fallback_last_on_corrupt: bool,
) -> tuple[dict[str, Any], Path]:
    try:
        return _torch_load_checkpoint(path, device), path
    except RuntimeError:
        fallback_path = path.with_name("last.pt")
        if (
            fallback_last_on_corrupt
            and path.name == "best.pt"
            and fallback_path.exists()
            and fallback_path != path
        ):
            print(
                f"Warning: requested checkpoint is corrupted: {path}\n"
                f"Falling back to: {fallback_path}",
                file=sys.stderr,
            )
            return _torch_load_checkpoint(fallback_path, device), fallback_path
        raise


def write_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write output: {output_path}")


def clean_upper_half(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    cleaned[: cleaned.shape[0] // 2, :] = 0
    return cleaned


def resolve_mast3r_output_path(cfg: InferenceConfig) -> Path | None:
    if cfg.mast3r_output_path is not None:
        return cfg.mast3r_output_path
    if cfg.scene_dir is None:
        return None
    return cfg.scene_dir / "mast3r" / "court_line_masks.npy"


def load_train_split(scene_dir: Path) -> list[str]:
    train_split_path = scene_dir / "images_train.txt"
    if not train_split_path.exists():
        raise FileNotFoundError(f"Train split file not found: {train_split_path}")
    return [
        line.strip()
        for line in train_split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)
    mast3r_output_path = resolve_mast3r_output_path(cfg)
    train_split = load_train_split(cfg.scene_dir) if cfg.scene_dir is not None else []

    dataset = CourtLineInferenceDataset(cfg.image_dir, cfg.short_side)
    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": pad_collate,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    dataloader = DataLoader(**loader_kwargs)

    checkpoint, loaded_checkpoint_path = load_checkpoint(
        cfg.checkpoint,
        device,
        fallback_last_on_corrupt=cfg.fallback_last_on_corrupt,
    )
    model_cfg = checkpoint.get("config", {})
    model = CourtUNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 1)),
    ).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_mask_dir = cfg.output_dir / "masks"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    futures: list[Future[None]] = []
    manifest_items: list[dict[str, Any]] = []
    train_split_index = {image_id: idx for idx, image_id in enumerate(train_split)}
    train_mask_memmap: np.memmap | None = None
    train_mask_shape: tuple[int, int] | None = None
    train_written = np.zeros((len(train_split),), dtype=bool) if train_split else None

    use_amp = device.type == "cuda"
    with ThreadPoolExecutor(max_workers=cfg.save_workers) as pool, torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(
                device=device,
                non_blocking=pin_memory,
                memory_format=torch.channels_last,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).float()

            for idx, image_id in enumerate(batch["image_id"]):
                infer_h, infer_w = batch["infer_size"][idx]
                orig_h, orig_w = batch["orig_size"][idx]
                prob = probs[idx, :infer_h, :infer_w].unsqueeze(0).unsqueeze(0)
                prob = F.interpolate(
                    prob,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                mask = (prob >= cfg.threshold).astype(np.uint8) * 255
                mask = clean_upper_half(mask)
                output_path = output_mask_dir / f"{image_id}.png"
                futures.append(pool.submit(write_mask, mask, output_path))
                if image_id in train_split_index:
                    binary_mask = (mask > 0).astype(np.uint8)
                    if train_mask_memmap is None:
                        train_mask_shape = binary_mask.shape
                        if mast3r_output_path is None:
                            raise RuntimeError("Internal error: missing mast3r output path for train-split export.")
                        mast3r_output_path.parent.mkdir(parents=True, exist_ok=True)
                        train_mask_memmap = open_memmap(
                            mast3r_output_path,
                            mode="w+",
                            dtype=np.uint8,
                            shape=(len(train_split), train_mask_shape[0], train_mask_shape[1]),
                        )
                    if binary_mask.shape != train_mask_shape:
                        raise RuntimeError("Train-split mask sizes are inconsistent; cannot write a single .npy array.")
                    split_idx = train_split_index[image_id]
                    train_mask_memmap[split_idx] = binary_mask
                    if train_written is not None:
                        train_written[split_idx] = True
                manifest_items.append(
                    {
                        "image_id": image_id,
                        "input_path": batch["path"][idx],
                        "mask_path": str(output_path),
                        "orig_size": [orig_h, orig_w],
                        "infer_size": [infer_h, infer_w],
                        "upper_half_cleaned": True,
                    }
                )

        for future in futures:
            future.result()

    if mast3r_output_path is not None:
        if train_written is None or train_mask_memmap is None:
            raise RuntimeError("No train-split masks were written.")
        missing = [image_id for idx, image_id in enumerate(train_split) if not train_written[idx]]
        if missing:
            raise RuntimeError(
                "Missing masks for train split images: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        train_mask_memmap.flush()

    manifest = {
        "requested_checkpoint": str(cfg.checkpoint),
        "loaded_checkpoint": str(loaded_checkpoint_path),
        "device": str(device),
        "batch_size": cfg.batch_size,
        "short_side": cfg.short_side,
        "threshold": cfg.threshold,
        "upper_half_cleaned": True,
        "num_images": len(dataset),
        "scene_dir": str(cfg.scene_dir) if cfg.scene_dir is not None else None,
        "mast3r_output_path": str(mast3r_output_path) if mast3r_output_path is not None else None,
        "train_split_count": len(train_split),
        "items": manifest_items,
    }
    manifest_path = cfg.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Inference complete: {len(dataset)} images")
    print(f"Masks: {output_mask_dir}")
    if mast3r_output_path is not None:
        print(f"MASt3R masks: {mast3r_output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
