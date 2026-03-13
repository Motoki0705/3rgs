/** Adapters: convert raw API responses to unified ViewModels. */

import type {
  CameraModel,
  PointCloudModel,
  SceneViewModel,
  CourtInitViewModel,
  CourtResultViewModel,
  MetricsModel,
} from './models';

// ── Helpers ─────────────────────────────────────────────────────────────────

function buildPointCloud(raw: { xyz: number[][]; rgb: number[][] }): PointCloudModel {
  const N = raw.xyz.length;
  const positions = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    positions[i * 3] = raw.xyz[i][0];
    positions[i * 3 + 1] = raw.xyz[i][1];
    positions[i * 3 + 2] = raw.xyz[i][2];
    if (raw.rgb && raw.rgb.length > i) {
      colors[i * 3] = raw.rgb[i][0];
      colors[i * 3 + 1] = raw.rgb[i][1];
      colors[i * 3 + 2] = raw.rgb[i][2];
    } else {
      colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.55;
    }
  }
  return { positions, colors, count: N };
}

// ── Scene adapter ───────────────────────────────────────────────────────────

export function sceneAdapter(raw: any): SceneViewModel {
  const pc = buildPointCloud(raw.points ?? raw.point_cloud ?? { xyz: [], rgb: [] });
  const cameras: CameraModel[] = (raw.cameras ?? []).map((c: any) => ({
    imageName: c.image_name,
    globalIdx: c.global_idx ?? c.idx ?? 0,
    isTrain: c.is_train ?? true,
    c2w: c.c2w,
    c2wRefined: c.c2w_refined ?? undefined,
    trainIdx: c.train_idx,
    fx: c.fx,
    fy: c.fy,
    cx: c.cx,
    cy: c.cy,
    width: c.width ?? Math.round((c.cx ?? 0) * 2),
    height: c.height ?? Math.round((c.cy ?? 0) * 2),
    delta: c.delta,
  }));
  return {
    pointCloud: pc,
    cameras,
    mode: raw.mode ?? 'ckpt',
  };
}

// ── Court-init adapter ──────────────────────────────────────────────────────

export function courtInitAdapter(raw: any): CourtInitViewModel {
  const pc = buildPointCloud(raw.point_cloud ?? { xyz: [], rgb: [] });
  const cameras: CameraModel[] = (raw.cameras ?? []).map((c: any, i: number) => ({
    imageName: c.image_name ?? `cam_${i}`,
    globalIdx: c.global_idx ?? i,
    isTrain: true,
    c2w: c.c2w,
    fx: c.fx ?? 346,
    fy: c.fy ?? 346,
    cx: c.cx ?? 248,
    cy: c.cy ?? 188,
    width: c.width ?? Math.round((c.cx ?? 248) * 2),
    height: c.height ?? Math.round((c.cy ?? 188) * 2),
  }));
  return {
    pointCloud: pc,
    cameras,
    autoSim3: {
      scale: raw.scale,
      rotation: raw.rotation,
      translation: raw.translation,
      adjacentGap: raw.adjacent_gap ?? 3.0,
      adjacentDirection: raw.adjacent_direction ?? '+x',
    },
    court: {
      keypointsCourt: raw.court_keypoints_court,
      skeleton: raw.court_skeleton,
    },
    imageDir: raw.image_dir ?? '',
  };
}

// ── Court-result adapter ────────────────────────────────────────────────────

export function courtResultAdapter(raw: any): CourtResultViewModel {
  const pc = buildPointCloud(raw.point_cloud ?? { xyz: [], rgb: [] });
  const cameras: CameraModel[] = (raw.cameras ?? []).map((c: any) => ({
    imageName: c.image_name,
    globalIdx: c.idx ?? c.global_idx ?? 0,
    isTrain: true,
    c2w: c.c2w,
    c2wRefined: c.c2w_refined,
    fx: c.fx,
    fy: c.fy,
    cx: c.cx,
    cy: c.cy,
    width: c.width ?? Math.round((c.cx ?? 0) * 2),
    height: c.height ?? Math.round((c.cy ?? 0) * 2),
    delta: c.delta ?? 0,
  }));

  const deltas = cameras.map((c) => c.delta ?? 0);
  const maxDelta = Math.max(...deltas, 0);
  const meanDelta = deltas.length > 0 ? deltas.reduce((a, b) => a + b, 0) / deltas.length : 0;

  const ms = raw.metrics_summary ?? {};
  const rawCourtPair = raw.court_pair ?? {};
  const metrics: MetricsModel = {
    numCameras: ms.num_cameras ?? cameras.length,
    poseTransSigma: ms.pose_trans_sigma ?? 0,
    adjacentGap: ms.adjacent_gap ?? 0,
    adjacentDirection: ms.adjacent_direction ?? '+x',
    maxDelta,
    meanDelta,
    totalLoss: ms.total_loss,
    selectedFitSource: ms.selected_fit_source,
  };

  return {
    pointCloud: pc,
    cameras,
    sim3: {
      scale: raw.sim3.scale,
      rotation: raw.sim3.rotation,
      translation: raw.sim3.translation,
      adjacentGap: raw.sim3.adjacent_gap ?? 0,
      adjacentDirection:
        typeof raw.sim3.adjacent_direction === 'string'
          ? raw.sim3.adjacent_direction
          : `[${(raw.sim3.adjacent_direction ?? []).map((v: number) => v.toFixed(2)).join(', ')}]`,
      matrix: raw.sim3.matrix,
    },
    court: {
      keypointsCourt: raw.court_keypoints_court,
      skeleton: raw.court_skeleton,
    },
    courtPair: {
      ...rawCourtPair,
      adjacentCenterOffset:
        rawCourtPair.adjacentCenterOffset ?? rawCourtPair.adjacent_center_offset,
      adjacentDirectionVector:
        rawCourtPair.adjacentDirectionVector ?? rawCourtPair.adjacent_direction_vector,
    },
    metrics,
  };
}
