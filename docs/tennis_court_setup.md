# 3RGS `tennis_court` Setup and Runbook

## Summary

This repository has been prepared to run `3RGS` on the dataset at `~/repos/gaussian-exp/data/tennis_court`, which was built from `gaussian-exp/convert.py` and therefore starts as a COLMAP-style scene, not as a native MASt3R scene.

To bridge that gap, this repo now includes:

- `scripts/prepare_colmap_scene.py`
  - Generates the minimum MASt3R-style metadata expected by `src/datasets/mast3r.py`
  - Creates:
    - `images_train.txt`
    - `images_test.txt`
    - `pose_gt_train.npy`
    - `pose_gt_test.npy`
    - `mast3r/camera_intrinsics.npy`
    - `mast3r/camera_poses.npy`
    - `mast3r/pointcloud.ply`
- checkpoint resume support in `src/trainer.py`
- compatibility fixes for current `pycolmap` and newer `torch` / `gsplat`
- SSIM loss fallback when `fused-ssim` is unavailable

Important constraint:

- `scripts/prepare_colmap_scene.py` only creates the base MASt3R-style metadata.
- `epipolar-loss` requires an extra preparation step:
  - `scripts/prepare_epipolar_data.py`

## Verified environment

Verified on `2026-03-08` with:

- Python `3.10`
- `uv 0.9.5`
- `torch 2.10.0+cu128`
- `torchvision 0.25.0`
- `gsplat 1.5.3`
- GPU: `NVIDIA GeForce RTX 5060 Ti`

The older pinned `gsplat` revision in the original repo failed on this GPU generation.

## Repository and dataset location

Clone target:

```bash
git clone https://github.com/zsh523/3rgs.git ~/repos/3rgs
cd ~/repos/3rgs
```

Dataset link used in this setup:

```bash
ln -sfn ~/repos/gaussian-exp/data/tennis_court ~/repos/3rgs/data_tennis_court
```

## Reproducible environment setup with `uv`

Create the virtual environment:

```bash
cd ~/repos/3rgs
uv venv --python 3.10 .venv
source .venv/bin/activate
```

Install PyTorch first:

```bash
uv pip install torch==2.10.0 torchvision==0.25.0
```

Install the remaining dependencies:

```bash
uv pip install -r requirements.txt
```

Optional:

- `fused-ssim` is not required anymore because `trainer.py` now falls back to `torchmetrics` SSIM automatically.
- If you still want to try it, install it after PyTorch is finalized, not before.

## Dataset preparation

Generate the MASt3R-style metadata from the COLMAP scene:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
python scripts/prepare_colmap_scene.py \
  --scene_dir ~/repos/3rgs/data_tennis_court \
  --test_every 8
```

Notes:

- `--test_every 8` creates the split used in the verified run.
- The script reads `sparse/0` and `images/`.
- It copies `sparse/0/points3D.ply` into `mast3r/pointcloud.ply`.
- It uses COLMAP camera poses both as pseudo-MASt3R poses and as GT poses.

### Prepare epipolar correspondence data

To enable `--use-corres-epipolar-loss`, generate the required correspondence tensors:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
python scripts/prepare_epipolar_data.py \
  --scene_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --pair_window 2 \
  --min_shared_points 64 \
  --num_correspondences 256
```

This writes:

- `mast3r/corr_i.npy`
- `mast3r/corr_j.npy`
- `mast3r/corr_batch_idx.npy`
- `mast3r/corr_mask.npy`
- `mast3r/corr_weight.npy`
- `mast3r/ei.npy`
- `mast3r/ej.npy`
- `mast3r/depthmaps.npy`

These tensors are generated from COLMAP sparse 3D tracks, not from MASt3R itself.

## Verified training commands

### 1. Minimal smoke test training

This is the command that was actually run successfully:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_min \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --max_steps 10 \
  --save_steps 5 10 \
  --eval_steps 10 \
  --tb_every 1
```

What this produced:

- checkpoints:
  - `results/tennis_court_min/ckpts/ckpt_4_rank0.pt`
  - `results/tennis_court_min/ckpts/ckpt_9_rank0.pt`
- metrics:
  - `results/tennis_court_min/stats/val_step0009.json`
  - `results/tennis_court_min/stats/train_step0009.json`
- rendered validation images:
  - `results/tennis_court_min/renders/`
- visualization video:
  - `results/tennis_court_min/videos/traj_9.mp4`

Observed final summary from the verified run:

- `TRAIN PSNR: 14.409`
- `TRAIN SSIM: 0.6077`
- `TRAIN LPIPS: 0.809`
- `TEST PSNR: 14.282`
- `TEST SSIM: 0.6025`
- `TEST LPIPS: 0.809`

### 2. Resume training from checkpoint

Resume was also verified.

Command:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_resume_check \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --max_steps 6 \
  --save_steps 6 \
  --eval_steps 6 \
  --tb_every 1 \
  --resume_ckpt ~/repos/3rgs/results/tennis_court_min/ckpts/ckpt_4_rank0.pt
```

What this verified:

- training resumed from `step 5`
- optimizer state, scheduler state, pose module state, and splat state were restored
- a new checkpoint was written after the resumed step
- evaluation and trajectory rendering completed after resume

### 3. Visualization / evaluation from a trained checkpoint

This command was also verified:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_viz_check \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --ckpt ~/repos/3rgs/results/tennis_court_min/ckpts/ckpt_9_rank0.pt
```

What it does:

- loads the checkpoint without training
- runs evaluation
- renders the trajectory video
- writes the video to:
  - `results/tennis_court_viz_check/videos/traj_9.mp4`

### 4. Minimal epipolar-loss smoke test

This was also verified successfully:

```bash
source ~/repos/3rgs/.venv/bin/activate
cd ~/repos/3rgs
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_epi_min \
  --pose_opt_type sfm \
  --use-corres-epipolar-loss \
  --max_steps 3 \
  --save_steps 3 \
  --eval_steps 3 \
  --tb_every 1
```

Observed in the verified run:

- epipolar loss became non-zero after step 1
- example loss values:
  - `corres epipolar loss=1.120529`
  - `corres epipolar loss=1.115972`

## Command examples for normal use

### Standard short run

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_default \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --max_steps 1000 \
  --save_steps 250 500 1000 \
  --eval_steps 500 1000
```

### MCMC variant

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_mcmc \
  --pose_opt_type mlp \
  --use-corres-epipolar-loss
```

### Resume example

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_resume \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --max_steps 2000 \
  --save_steps 1500 2000 \
  --eval_steps 2000 \
  --resume_ckpt ~/repos/3rgs/results/tennis_court_default/ckpts/ckpt_999_rank0.pt
```

### Evaluation and visualization only

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir ~/repos/3rgs/data_tennis_court \
  --data_factor 4 \
  --result_dir ~/repos/3rgs/results/tennis_court_eval_only \
  --pose_opt_type sfm \
  --no-use-corres-epipolar-loss \
  --ckpt ~/repos/3rgs/results/tennis_court_default/ckpts/ckpt_1999_rank0.pt
```

## All configurable settings

The CLI is `python src/trainer.py <mode> [options]`, where `<mode>` is one of:

- `default`
  - original densification strategy
- `mcmc`
  - MCMC-based densification strategy

### Core training and I/O

- `--data_dir STR`
  - input scene directory
- `--data_factor INT`
  - image downsample factor used by the dataset loader
- `--result_dir STR`
  - output directory for checkpoints, stats, renders, tensorboard, and videos
- `--max_steps INT`
  - total training steps
- `--save_steps [INT ...]`
  - save checkpoints at these logical steps
- `--eval_steps [INT ...]`
  - run evaluation and trajectory rendering at these logical steps
- `--steps_scaler FLOAT`
  - scales `max_steps`, `save_steps`, `eval_steps`, and some strategy thresholds together
- `--batch_size INT`
  - batch size for the dataloader and LR scaling
- `--tb_every INT`
  - tensorboard logging interval
- `--tb_save_image`
  - also store rendered training images in tensorboard
- `--disable_viewer`
  - disables the live viewer server
- `--port INT`
  - viewer port when the viewer is enabled

### Checkpoints and post-processing

- `--ckpt [PATH ...]`
  - evaluation-only checkpoint load
- `--resume_ckpt [PATH ...]`
  - resume training from checkpoint
- `--compression {png}`
  - optional compression stage after eval
- `--render_traj_path STR`
  - trajectory type for visualization
  - supported values in code:
    - `interp`
    - `ellipse`
    - `spiral`

### Pose, intrinsics, and epipolar settings

- `--use-corres-epipolar-loss`
  - enables global correspondence epipolar loss
  - requires correspondence tensors generated by `scripts/prepare_epipolar_data.py`
- `--no-use-corres-epipolar-loss`
  - disables that loss
- `--epi-loss-weight FLOAT`
  - weight for epipolar loss
- `--pose-opt`
  - enables pose optimization
- `--pose-opt-type {sfm,mlp}`
  - pose optimization module type
- `--pose-opt-lr FLOAT`
  - pose optimization learning rate
- `--pose-opt-reg FLOAT`
  - pose optimization weight decay
- `--pose-noise FLOAT`
  - adds pose noise for experiments
- `--intrinsics-opt`
  - enables intrinsics optimization
- `--focal-opt-lr FLOAT`
  - focal length learning rate
- `--pp-opt-lr FLOAT`
  - principal point learning rate
- `--focal-opt-reg FLOAT`
  - focal optimizer weight decay
- `--pp-opt-reg FLOAT`
  - principal point optimizer weight decay
- `--intrinsics-noise FLOAT`
  - adds intrinsics noise for experiments

### Scene and camera settings

- `--test_every INT`
  - dataset split frequency metadata
- `--patch_size INT`
  - random crop size during training
- `--global_scale FLOAT`
  - extra multiplier on scene scale parameters
- `--normalize_world_space`
  - normalizes world coordinates
- `--camera_model {pinhole,ortho,fisheye}`
  - rasterization camera model
- `--near_plane FLOAT`
  - near clipping plane
- `--far_plane FLOAT`
  - far clipping plane

### Initialization and rendering parameters

- `--init-type STR`
  - usually `sfm` or `random`
- `--init-num-pts INT`
  - initial random point count when not using `sfm`
- `--init-extent FLOAT`
  - extent multiplier for random init
- `--sh-degree INT`
  - target spherical harmonics degree
- `--sh-degree-interval INT`
  - steps between SH degree increases
- `--init-opa FLOAT`
  - initial opacity
- `--init-scale FLOAT`
  - initial scale multiplier
- `--ssim-lambda FLOAT`
  - blend ratio between L1 and SSIM in the loss

### Rasterization and optimizer behavior

- `--packed`
  - packed rasterization mode
- `--sparse-grad`
  - sparse gradient optimization mode
- `--visible-adam`
  - visibility-aware Adam
- `--antialiased`
  - anti-aliased rasterization mode
- `--random-bkgd`
  - randomizes training background color
- `--opacity-reg FLOAT`
  - opacity regularization term
- `--scale-reg FLOAT`
  - scale regularization term

### Appearance and bilateral grid

- `--app-opt`
  - enables appearance optimization module
- `--app-embed-dim INT`
  - appearance embedding size
- `--app-opt-lr FLOAT`
  - appearance optimizer learning rate
- `--app-opt-reg FLOAT`
  - appearance optimizer weight decay
- `--use-bilateral-grid`
  - enables bilateral grid
- `--bilateral-grid-shape INT INT INT`
  - bilateral grid dimensions `(X, Y, W)`

### Depth and evaluation

- `--depth-loss`
  - enables depth supervision
- `--depth-lambda FLOAT`
  - depth loss weight
- `--lpips-net {vgg,alex}`
  - LPIPS backbone used in evaluation

### Strategy settings for `default` mode

- `--strategy.prune-opa FLOAT`
  - prune Gaussians below this opacity
- `--strategy.grow-grad2d FLOAT`
  - split/duplicate threshold from 2D gradient
- `--strategy.grow-scale3d FLOAT`
  - 3D scale threshold controlling duplicate vs split
- `--strategy.grow-scale2d FLOAT`
  - 2D scale split threshold
- `--strategy.prune-scale3d FLOAT`
  - 3D scale prune threshold
- `--strategy.prune-scale2d FLOAT`
  - 2D scale prune threshold
- `--strategy.refine-scale2d-stop-iter INT`
  - stop 2D-scale-driven refinement after this step
- `--strategy.refine-start-iter INT`
  - first refinement step
- `--strategy.refine-stop-iter INT`
  - last refinement step
- `--strategy.reset-every INT`
  - opacity reset interval
- `--strategy.refine-every INT`
  - refinement interval
- `--strategy.pause-refine-after-reset INT`
  - refinement cooldown after reset
- `--strategy.absgrad`
  - use absolute gradients
- `--strategy.revised-opacity`
  - revised opacity heuristic
- `--strategy.verbose`
  - verbose strategy logging
- `--strategy.key-for-gradient {means2d,gradient_2dgs}`
  - densification gradient source

### Mode-specific note for `mcmc`

- `mcmc` swaps the densification strategy and changes some defaults in code:
  - `init_opa=0.5`
  - `init_scale=0.1`
  - `opacity_reg=0.01`
  - `scale_reg=0.01`

## Output layout

For any `--result_dir`, the trainer writes:

- `cfg.yml`
  - dumped runtime config
- `ckpts/`
  - checkpoint files
- `stats/`
  - train and val metrics json
- `renders/`
  - validation render images
- `videos/`
  - rendered trajectory videos
- `tb/`
  - tensorboard logs

## Files changed in this repo for this setup

- `scripts/prepare_colmap_scene.py`
- `scripts/prepare_epipolar_data.py`
- `src/datasets/mast3r.py`
- `src/trainer.py`
- `requirements.txt`
- `tools/colab_setup.py`

## Google Colab

There is now a Colab setup-and-train helper modeled after `gaussian-exp/tools/colab_setup.py`:

- `tools/colab_setup.py`

It handles:

- Drive mount
- dependency installation
- copying a dataset tar from Drive to `/content`
- extraction into `/content/data/<scene_name>`
- repo symlink creation
- `prepare_colmap_scene.py`
- `prepare_epipolar_data.py`
- runtime verification
- optional direct training

Example:

```bash
python tools/colab_setup.py \
  --drive-tar-path MyDrive/datasets/tennis_court_colab_court_depth_min_v2.tar \
  --scene-name tennis_court \
  --result-dir /content/drive/MyDrive/3rgs_runs/tennis_court_full \
  --train-mode default \
  --pose-opt-type sfm \
  --data-factor 4 \
  --use-epipolar-loss \
  --max-steps 30000 \
  --save-steps 7000 30000 \
  --eval-steps 7000 30000
```
