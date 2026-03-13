# court_ground_fit

テニスコートのラインマスクを、推定された地面平面上で扱うためのツール群です。

このディレクトリには、独立した 2 つのスクリプトがあります。

- `project_court_lines_to_ground.py`
  各カメラのコートラインマスクを地面平面へ投影し、俯瞰画像や統計情報を保存します。
- `fit_from_ground_heatmap.py`
  地面平面上に集約したヒートマップに対して 2 面コートのテンプレートを当てはめ、
  chunk 単位の fit と clustering を行い、支配的クラスタを再 fit して
  `init_sim3.json` 形式の初期値を出力します。

どちらも `tools/fit_court_sim3.py` とは独立しています。

## 入力

以下のファイルを前提とします。

- `<scene_dir>/images_train.txt`
- `<scene_dir>/mast3r/camera_intrinsics.npy`
- `<scene_dir>/mast3r/camera_poses.npy`
- `<scene_dir>/mast3r/court_line_masks.npy`

デフォルトの対象シーンは以下です。

- `data/tennis_court`

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
  --output-dir results/tennis_court/court_ground_heatmap_fit \
  --adjacent-court-direction +x
```

## 主な出力

`fit_from_ground_heatmap.py` の出力:

- `init_sim3_from_ground_heatmap.json`
- `metadata.json`
- `heatmaps.npz`
- `dominant_cluster_overlay.png`
- `selected_fit_overlay.png`
- `chunk_overlays_contact_sheet.png`
- `chunks/*.png`

`project_court_lines_to_ground.py` の出力:

- `merged_projection_heatmap.png`
- `merged_projection_binary.png`
- `per_camera/*.png`
- `per_camera_contact_sheet.png`
- `metadata.json`

## 出力の見方

- `dominant_cluster_overlay.png`
  chunk clustering 後に、支配的なクラスタだけで再 fit した結果です。
- `selected_fit_overlay.png`
  最終的に採用された fit 結果です。
- `metadata.json`
  chunk ごとの fit 結果、クラスタ割当、silhouette score、採用されたカメラ集合などを含みます。

## 注意

- この fit は画像平面ではなく、推定された地面平面上で行います。
- chunk clustering は、カメラポーズのドリフトの影響を減らすために入れています。
- 再 fit に失敗した場合は、最良の chunk fit にフォールバックします。
