# court_ground_fit

テニスコートのラインマスクを、推定された地面平面上で扱うためのツール群です。

このディレクトリには、連続した 2 つのスクリプトがあります。

- `project_court_lines_to_ground.py`
  各カメラのコートラインマスクを地面平面へ投影し、
  後段が使う最小成果物を `data/.../court/ground/` に保存します。
- `fit_from_ground_heatmap.py`
  地面平面上に集約したヒートマップに対して 2 面コートのテンプレートを当てはめ、
  chunk 単位の fit と clustering を行い、支配的クラスタを再 fit して
  `court/transform/init_sim3.json` を seed に最終 transform を `data/.../court/transform/` に出力します。

## 入力

以下のファイルを前提とします。

- `<scene_dir>/court/ground/plane_frame.json`
- `<scene_dir>/court/ground/projected_train.npz`
- `<scene_dir>/court/ground/raster_grid.json`
- `<scene_dir>/court/ground/visibility_train.npz`
- `<scene_dir>/court/transform/init_sim3.json`

デフォルトの対象シーンは以下です。

- `data/tennis_court`

## ディレクトリ構成

正規成果物:

- `<scene_dir>/court/ground/plane_frame.json`
- `<scene_dir>/court/ground/projected_train.npz`
- `<scene_dir>/court/ground/raster_grid.json`
- `<scene_dir>/court/ground/visibility_train.npz`
- `<scene_dir>/court/ground/manifest.json`
- `<scene_dir>/court/transform/ground_heatmap_fit.json`
- `<scene_dir>/court/transform/ground_heatmap_fit_sim3.json`
- `<scene_dir>/court/transform/manifest.json`

デバッグ可視化:

- `results/tennis_court/court/ground/`
- `results/tennis_court/court/transform/`

## 使い方

各カメラのマスクを地面平面へ投影する:

```bash
python tools/court_ground_fit/project_court_lines_to_ground.py
```

地面平面ヒートマップに対して 2 面コートを fit する:

```bash
python tools/court_ground_fit/fit_from_ground_heatmap.py
```

よく使う実行例:

```bash
python tools/court_ground_fit/fit_from_ground_heatmap.py \
  --scene-dir data/tennis_court \
  --ground-dir data/tennis_court/court/ground \
  --transform-dir data/tennis_court/court/transform \
  --init-sim3-path data/tennis_court/court/transform/init_sim3.json \
  --output-dir results/tennis_court/court/transform
```

## 主な出力

`fit_from_ground_heatmap.py` の出力:

- 正規成果物
  - `data/.../court/transform/ground_heatmap_fit.json`
  - `data/.../court/transform/ground_heatmap_fit_sim3.json`
  - `data/.../court/transform/manifest.json`
- デバッグ成果物
  - `results/.../court/transform/metadata.json`
  - `results/.../court/transform/heatmaps.npz`
  - `results/.../court/transform/dominant_cluster_overlay.png`
  - `results/.../court/transform/selected_fit_overlay.png`
  - `results/.../court/transform/chunk_overlays_contact_sheet.png`
  - `results/.../court/transform/chunks/*.png`

`project_court_lines_to_ground.py` の出力:

- 正規成果物
  - `data/.../court/ground/plane_frame.json`
  - `data/.../court/ground/projected_train.npz`
  - `data/.../court/ground/raster_grid.json`
  - `data/.../court/ground/visibility_train.npz`
  - `data/.../court/ground/manifest.json`
- デバッグ成果物
  - `results/.../court/ground/merged_projection_heatmap.png`
  - `results/.../court/ground/merged_projection_binary.png`
  - `results/.../court/ground/per_camera/*.png`
  - `results/.../court/ground/per_camera_contact_sheet.png`
  - `results/.../court/ground/reliability/*.png`
  - `results/.../court/ground/reliability_contact_sheet.png`
  - `results/.../court/ground/metadata.json`

## 出力の見方

- `dominant_cluster_overlay.png`
  chunk clustering 後に、支配的なクラスタだけで再 fit した結果です。
- `selected_fit_overlay.png`
  最終的に採用された fit 結果です。
- `metadata.json`
  chunk ごとの fit 結果、クラスタ割当、silhouette score、採用されたカメラ集合などを含みます。

## 注意

- `fit_from_ground_heatmap.py` は `project_court_lines_to_ground.py` の成果物を直接読み込みます。
- `project_court_lines_to_ground.py` は共通 raster grid と per-camera reliability map を生成します。
- `fit_from_ground_heatmap.py` の forward loss は chunk reliability map で重み付けされます。
- `fit_from_ground_heatmap.py` は `court/transform/init_sim3.json` を seed として読み込み、`CourtInitEstimator` は呼びません。
- この fit は画像平面ではなく、推定された地面平面上で行います。
- chunk clustering は、カメラポーズのドリフトの影響を減らすために入れています。
- 再 fit に失敗した場合は、最良の chunk fit にフォールバックします。
