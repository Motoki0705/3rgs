/** Shared data models used across all viewer modes. */
import * as THREE from 'three';

// ── Camera model ────────────────────────────────────────────────────────────

export interface CameraModel {
  imageName: string;
  globalIdx: number;
  isTrain: boolean;
  c2w: number[][];          // 4×4 row-major
  c2wRefined?: number[][];  // 4×4 row-major (train only)
  trainIdx?: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  delta?: number;           // |orig - refined| distance
}

// ── Point cloud model ───────────────────────────────────────────────────────

export interface PointCloudModel {
  positions: Float32Array;  // [N*3]
  colors: Float32Array;     // [N*3]
  count: number;
}

// ── Sim(3) model ────────────────────────────────────────────────────────────

export interface Sim3Model {
  scale: number;
  rotation: number[][];     // 3×3 row-major
  translation: number[];    // [tx, ty, tz]
  adjacentGap: number;
  adjacentDirection: string; // "+x" | "-x" | "+y" | "-y"
  matrix?: number[][];       // 4×4 (court-result only)
}

// ── Court model ─────────────────────────────────────────────────────────────

export interface CourtModel {
  keypointsCourt: number[][]; // (20, 3)
  skeleton: [number, number][];
}

// ── Metrics model (court-result) ────────────────────────────────────────────

export interface MetricsModel {
  numCameras: number;
  poseTransSigma: number;
  adjacentGap: number;
  adjacentDirection: string;
  maxDelta: number;
  meanDelta: number;
  totalLoss?: number;
  selectedFitSource?: string;
}

// ── Court pair model (court-result) ─────────────────────────────────────────

export interface CourtPairModel {
  adjacentCenterOffset?: number;
  adjacentDirectionVector?: number[];
  adjacent_center_offset?: number;
  adjacent_direction_vector?: number[];
  [key: string]: unknown;
}

// ── Mode ────────────────────────────────────────────────────────────────────

export type ViewerMode = 'scene' | 'court-init' | 'court-result';

// ── ViewModel: the unified shape every page receives ────────────────────────

export interface SceneViewModel {
  pointCloud: PointCloudModel;
  cameras: CameraModel[];
  mode: 'ckpt' | 'mast3r';
}

export interface CourtInitViewModel {
  pointCloud: PointCloudModel;
  cameras: CameraModel[];
  autoSim3: Sim3Model;
  court: CourtModel;
  imageDir: string;
}

export interface CourtResultViewModel {
  pointCloud: PointCloudModel;
  cameras: CameraModel[];
  sim3: Sim3Model;
  court: CourtModel;
  courtPair: CourtPairModel;
  metrics: MetricsModel;
}

// ── Math helpers ────────────────────────────────────────────────────────────

/** ITF court constant */
export const DOUBLES_WIDTH = 10.97;

export const ADJ_DIRS: Record<string, [number, number, number]> = {
  '+x': [1, 0, 0],
  '-x': [-1, 0, 0],
  '+y': [0, 1, 0],
  '-y': [0, -1, 0],
};

/** Convert row-major 4×4 array to THREE.Matrix4 */
export function c2wToMatrix4(c2w: number[][]): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(
    c2w[0][0], c2w[0][1], c2w[0][2], c2w[0][3],
    c2w[1][0], c2w[1][1], c2w[1][2], c2w[1][3],
    c2w[2][0], c2w[2][1], c2w[2][2], c2w[2][3],
    c2w[3][0], c2w[3][1], c2w[3][2], c2w[3][3],
  );
  return m;
}

/** Extract camera position + quaternion from c2w, applying the NeRF→Three.js flip. */
export function extractCameraTransform(c2w: number[][]): { pos: THREE.Vector3; q: THREE.Quaternion } {
  const m4 = c2wToMatrix4(c2w);
  const flip = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1,
  );
  m4.multiply(flip);
  const pos = new THREE.Vector3().setFromMatrixPosition(m4);
  const q = new THREE.Quaternion().setFromRotationMatrix(m4);
  return { pos, q };
}

/** Multiply 3×3 matrix (row-major) by 3-vector. */
export function matVec3(R: number[][], v: number[]): [number, number, number] {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

/** Apply 4×4 matrix (row-major) to 3D point (homogeneous). */
export function applyMatrix4ToPoint(mat4: number[][], p: number[]): [number, number, number] {
  return [
    mat4[0][0] * p[0] + mat4[0][1] * p[1] + mat4[0][2] * p[2] + mat4[0][3],
    mat4[1][0] * p[0] + mat4[1][1] * p[1] + mat4[1][2] * p[2] + mat4[1][3],
    mat4[2][0] * p[0] + mat4[2][1] * p[1] + mat4[2][2] * p[2] + mat4[2][3],
  ];
}

/** ZYX intrinsic Euler → 3×3 rotation matrix. R = Rz(yaw) @ Ry(pitch) @ Rx(roll) */
export function eulerZYXtoMatrix(yaw: number, pitch: number, roll: number): number[][] {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

/** 3×3 rotation matrix → ZYX intrinsic Euler (yaw, pitch, roll) in radians. */
export function matrixToEulerZYX(R: number[][]): { yaw: number; pitch: number; roll: number } {
  const pitch = Math.atan2(-R[2][0], Math.sqrt(R[0][0] ** 2 + R[1][0] ** 2));
  const roll = Math.atan2(R[2][1], R[2][2]);
  const yaw = Math.atan2(R[1][0], R[0][0]);
  return { yaw, pitch, roll };
}

/** Transform court keypoints to world via Sim(3): p_world = scale * (R @ p) + t */
export function courtToWorld(
  kpCourt: number[][],
  scale: number,
  R: number[][],
  t: number[],
): number[][] {
  return kpCourt.map((p) => {
    const rp = matVec3(R, p);
    return [scale * rp[0] + t[0], scale * rp[1] + t[1], scale * rp[2] + t[2]];
  });
}

/** Transform court keypoints to world via 4×4 matrix. */
export function courtToWorldMatrix(kpCourt: number[][], mat4: number[][]): number[][] {
  return kpCourt.map((p) => applyMatrix4ToPoint(mat4, p));
}

/** Offset keypoints in court space. */
export function offsetKeypoints(kpCourt: number[][], offset: number[]): number[][] {
  return kpCourt.map((p) => [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]);
}

export function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}
