# Setup

## Repositories

The expected layout is:

```text
~/repos/3rgs
~/repos/3rgs/third_party/mast3r
```

Initialize submodules after cloning:

```bash
cd ~/repos/3rgs
git submodule update --init --recursive
```

## 3RGS Environment

Create the main environment:

```bash
cd ~/repos/3rgs
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0
uv pip install -r requirements.txt
```

Verified runtime used in this workspace:

- Python `3.10`
- `torch 2.10.0+cu128`
- `torchvision 0.25.0`
- CUDA GPU runtime available

## MASt3R Environment

Create a separate environment under the submodule:

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

`scripts/preprocess.py` is launched from `3rgs/.venv`, but it internally calls:

```text
third_party/mast3r/.venv/bin/python
```

## Input Data Setup

Create a scene directory and copy images:

```bash
cd ~/repos/3rgs
mkdir -p data/tennis_court/images
cp /path/to/source_images/* data/tennis_court/images/
```

## Preprocessing

Run the integrated MASt3R-to-3RGS pipeline:

```bash
cd ~/repos/3rgs
source .venv/bin/activate
python scripts/preprocess.py \
  --scene_dir data/tennis_court \
  --shared_intrinsics \
  --test_every 8 \
  --device cuda \
  --scene_graph swin \
  --winsize 2
```

This command:

- runs MASt3R-SfM
- writes raw outputs under `data/tennis_court/mast3r_sfm/`
- packages 3RGS-ready metadata under `data/tennis_court/mast3r/`
