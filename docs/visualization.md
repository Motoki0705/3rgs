# Visualization

## MASt3R Viewer

`scripts/preprocess.py` writes raw MASt3R viewer assets under:

```text
data/{scene}/mast3r_sfm/viewer/
```

Serve them with the MASt3R helper:

```bash
cd ~/repos/3rgs/third_party/mast3r
source .venv/bin/activate
python scripts/serve_pose_viewer.py \
  --viewer-dir /root/repos/3rgs/data/tennis_court/mast3r_sfm/viewer \
  --host 0.0.0.0 \
  --port 8000
```

Then open:

```text
http://localhost:8000/
```

The viewer shows:

- reconstructed camera centers
- sparse points
- clickable camera markers
- source image previews

## TensorBoard

TensorBoard logs are written under:

```text
results/{run_name}/tb/
```

Launch TensorBoard:

```bash
cd ~/repos/3rgs
source .venv/bin/activate
tensorboard --logdir results/tennis_court_smoke/tb --port 6006
```

Open:

```text
http://localhost:6006/
```