# Learning

## Training Entry Point

3RGS training is started through:

```bash
python src/trainer.py <mode> \
  --data_dir data/{scene} \
  --data_factor <factor> \
  --result_dir results/{run_name} \
  --pose_opt_type <sfm|mlp>
```

Common modes:

- `default`
- `mcmc`

Common pose optimization types:

- `sfm`
- `mlp`

## Example Full Training

```bash
cd ~/repos/3rgs
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python src/trainer.py mcmc \
  --data_dir data/tennis_court \
  --data_factor 4 \
  --result_dir results/tennis_court_mcmc \
  --pose_opt_type mlp \
  --use-corres-epipolar-loss
```

## Training Outputs

Training artifacts are written under:

```text
results/{run_name}/
├── cfg.yml
├── ckpts/
├── renders/
├── stats/
├── tb/
└── videos/
```

Main files:

- `ckpts/ckpt_<step>_rank0.pt`
- `stats/train_step<step>.json`
- `stats/val_step<step>.json`
- `videos/traj_<step>.mp4`

Example from the smoke test:

```text
results/tennis_court_smoke/
├── ckpts/ckpt_99_rank0.pt
├── stats/train_step0099.json
├── stats/val_step0099.json
└── videos/traj_99.mp4
```

Observed smoke-test metrics at step 99:

- train: `PSNR 16.432`, `SSIM 0.6568`, `LPIPS 0.7539`
- val: `PSNR 15.999`, `SSIM 0.6504`, `LPIPS 0.7762`
- ATE RMSE: `0.00608 m`
- mean rotation error: `0.13 deg`

## Resume Training

Resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python src/trainer.py default \
  --data_dir data/tennis_court \
  --data_factor 4 \
  --result_dir results/tennis_court_resume \
  --pose_opt_type sfm \
  --use-corres-epipolar-loss \
  --ckpt results/tennis_court_smoke/ckpts/ckpt_99_rank0.pt
```
