# Scene Viewer

`tools/scene_viewer` は、3DGS 学習結果と MASt3R 前処理結果をブラウザで確認するための可視化ツールです。現状の主なエントリポイントは [server.py](/root/repos/3rgs/tools/scene_viewer/server.py) で、用途に応じて 3 つのモードを切り替えて使います。

## 何ができるか

- `scene` モード
  3DGS のチェックポイントから点群を取り出し、元カメラと refine 後カメラを比較表示します。
- `court-init` モード
  MASt3R 点群の上にテニスコートのワイヤーフレームを重ね、`init_sim3.json` を手動調整して保存します。
- `court-result` モード
  `fit_from_ground_heatmap.py` の transform 結果を可視化し、最終コート配置を確認します。

## 推奨起動方法

Python 版が 3 モードすべてをサポートします。リポジトリ直下で実行してください。

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py --help
```

## モード別の役割と起動コマンド

### 1. `scene`

役割:
- 3DGS チェックポイント (`.pt`) から opacity 上位の Gaussian を点群として表示します。
- 学習カメラについては MLP による refine 後 pose も表示します。
- カメラをクリックすると対応画像を右パネルに表示できます。

開く URL:
- `http://localhost:8080/`

起動例:

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py \
  --mode scene \
  --ckpt results/tennis_court_30k/ckpts/ckpt_29999_rank0.pt \
  --data-dir data/tennis_court \
  --port 8080 \
  --n-points 10000
```

主な入力:
- `--ckpt`: 3DGS 学習チェックポイント
- `--data-dir`: データセットディレクトリ
- `--n-points`: 表示する点群数

画面上で分かること:
- 青: 元カメラ
- 橙: refine 後カメラ
- 灰: テストカメラ
- 線: 元 pose から refined pose への移動量

### 2. `court-init`

役割:
- `mast3r/pointcloud.ply` とカメラ pose から初期 Sim(3) を自動推定します。
- UI 上で平行移動、回転、スケール、隣接コート方向、コート間 gap を調整できます。
- 調整結果を `init_sim3.json` として保存します。

開く URL:
- `http://localhost:8090/court_init`

起動例:

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py \
  --mode court-init \
  --scene-dir data/tennis_court \
  --port 8090 \
  --n-sample 50000 \
  --adjacent-direction +x \
  --init-gap 3.0
```

保存先を明示したい場合:

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py \
  --mode court-init \
  --scene-dir data/tennis_court \
  --output data/tennis_court/court/transform/init_sim3.json \
  --port 8090
```

主な入力:
- `--scene-dir`: `mast3r/` を含むシーンディレクトリ
- `--n-sample`: PLY から読み込む点群サンプル数
- `--adjacent-direction`: 隣接コートの向き (`+x`, `-x`, `+y`, `-y`)
- `--init-gap`: 隣接コート間の初期 gap
- `--output`: 保存先。未指定時は `<scene-dir>/court/transform/init_sim3.json`

画面上でできること:
- スライダーで `translation / rotation / scale / gap` を調整
- `adjacent_direction` を切り替え
- カメラを選択してその視点へワープ
- 自動初期値へリセット
- `init_sim3.json` を保存

### 3. `court-result`

役割:
- `fit_from_ground_heatmap.py` の出力を最終確認するためのビューアです。
- MASt3R 点群、元カメラ、2 面分のコートを重ねて表示します。
- heatmap fit から得た Sim(3) と gap の要約を確認できます。

開く URL:
- `http://localhost:8092/court_result`

起動例:

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py \
  --mode court-result \
  --scene-dir data/tennis_court \
  --port 8092 \
  --n-sample 50000
```

`court/transform` ディレクトリを明示する場合:

```bash
cd /root/repos/3rgs
source .venv/bin/activate
python tools/scene_viewer/server.py \
  --mode court-result \
  --scene-dir data/tennis_court \
  --transform-dir data/tennis_court/court/transform \
  --port 8092
```

主な入力:
- `--scene-dir`: 元シーン
- `--transform-dir`: `ground_heatmap_fit.json` と `ground_heatmap_fit_sim3.json` を含む結果ディレクトリ
- `--n-sample`: 表示する MASt3R 点群サンプル数

画面上でできること:
- 点群、元カメラ、コートの表示切替
- カメラ一覧から個別カメラを選択してワープ
- `Sim3 scale`、`adjacent gap`、`adjacent direction`、fit source、total loss を確認

## モードごとの必要ファイル

### `scene`

最低限必要なもの:
- チェックポイント: `results/.../ckpt_*.pt`
- データセット: `data/<scene>/`

`--data-dir` 側では、少なくとも画像群と MASt3R 由来のカメラ情報を [server.py](/root/repos/3rgs/tools/scene_viewer/server.py) が読める必要があります。

### `court-init`

`--scene-dir` 配下に以下が必要です。

- `mast3r/pointcloud.ply`
- `mast3r/camera_poses.npy`
- `mast3r/camera_intrinsics.npy`
- `images/`
- `images_train.txt`

### `court-result`

`court-init` と同じ入力に加えて、通常は `court/transform/` 配下に以下が必要です。

- `ground_heatmap_fit.json`
- `ground_heatmap_fit_sim3.json`
