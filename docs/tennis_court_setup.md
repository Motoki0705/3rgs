# 3RGS Scene Setup and Runbook

## Summary

This repository now treats `3rgs` as the parent workspace and `third_party/mast3r` as the MASt3R-SfM engine.
The expected input is a scene directory of the form `data/{scene}/images/`.

The integrated entrypoint is:

```bash
python scripts/preprocess.py --scene_dir data/{scene}
```

That command:

- runs `third_party/mast3r/.venv/bin/python third_party/mast3r/scripts/run_mast3r_sfm.py`
- writes raw MASt3R outputs to `data/{scene}/mast3r_sfm/`
- packages 3RGS-ready metadata into `data/{scene}/mast3r/`
- creates `images_train.txt`, `images_test.txt`, `pose_gt_train.npy`, and `pose_gt_test.npy`

No `gaussian-exp` assets or COLMAP conversion outputs are required.

## Repository Layout

```text
3rgs/
├── data/
│   └── {scene}/
│       ├── images/
│       ├── images_train.txt
│       ├── images_test.txt
│       ├── pose_gt_train.npy
│       ├── pose_gt_test.npy
│       ├── mast3r/
│       └── mast3r_sfm/
├── scripts/preprocess.py
└── third_party/mast3r/
```

## Environment Setup

Create the 3RGS environment:

```bash
cd ~/repos/3rgs
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0
uv pip install -r requirements.txt
```

Create the MASt3R environment under the submodule:

```bash
cd ~/repos/3rgs/third_party/mast3r
uv venv --python 3.10 .venv
source .venv/bin/activate
UV_HTTP_TIMEOUT=1000 uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0
UV_HTTP_TIMEOUT=1000 uv pip install --python .venv/bin/python \
  -r requirements.txt -r dust3r/requirements.txt
```

`preprocess.py` is launched from `3rgs/.venv`, but it invokes `third_party/mast3r/.venv/bin/python` internally for MASt3R.

## Scene Preparation

Copy source images into `data/{scene}/images/`.

Example:

```bash
mkdir -p ~/repos/3rgs/data/tennis_court/images
cp /path/to/source_images/* ~/repos/3rgs/data/tennis_court/images/
```

Then run preprocessing:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
python scripts/preprocess.py \
  --scene_dir ~/repos/3rgs/data/{scene} \
  --shared_intrinsics \
  --test_every 8 \
  --device cuda \
  --scene_graph swin \
  --winsize 2
```

Important outputs:

- `data/{scene}/mast3r/camera_intrinsics.npy`
- `data/{scene}/mast3r/camera_poses.npy`
- `data/{scene}/mast3r/pointcloud.ply`
- `data/{scene}/mast3r/corr_i.npy`
- `data/{scene}/mast3r/corr_j.npy`
- `data/{scene}/mast3r/corr_mask.npy`
- `data/{scene}/mast3r/corr_weight.npy`
- `data/{scene}/mast3r/corr_batch_idx.npy`
- `data/{scene}/mast3r/corr_is_manual.npy`
- `data/{scene}/mast3r/ei.npy`
- `data/{scene}/mast3r/ej.npy`
- `data/{scene}/mast3r/depthmaps.npy`

## Manual Annotation Workflow

The tennis-court-specific annotation tooling is still available on top of the MASt3R-generated pair graph.

Launch the UI:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
python tools/court_annotation_server.py \
  --scene_dir ~/repos/3rgs/data/{scene} \
  --data_factor 4 \
  --pair_window 2 \
  --port 8123
```

After annotating image pairs, rebuild the manual portion of the epipolar tensors:

```bash
python scripts/prepare_epipolar_data.py \
  --scene_dir ~/repos/3rgs/data/{scene} \
  --data_factor 4 \
  --max_manual_correspondences 40 \
  --manual_base_weight 1.0
```

This updates the tail slots of `corr_*.npy` and marks those entries in `corr_is_manual.npy`.

## Training

Minimal training example:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data/{scene} \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/{scene}_default \
  --pose_opt_type sfm \
  --use-corres-epipolar-loss
```

MLP pose refinement example:

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc \
  --data_dir ~/repos/3rgs/data/{scene} \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/{scene}_mlp \
  --pose_opt_type mlp \
  --use-corres-epipolar-loss
```

## Notes

- `pose_gt_train.npy` and `pose_gt_test.npy` now store packaged MASt3R reference poses rather than COLMAP output.
- `src/datasets/mast3r.py` no longer depends on `sparse/0/`.
- Raw MASt3R viewer artifacts remain under `data/{scene}/mast3r_sfm/` for inspection.
