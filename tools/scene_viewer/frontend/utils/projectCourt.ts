/**
 * projectCourt.ts — Project 3D court keypoints onto a 2D camera image.
 *
 * Uses the camera's c2w (camera-to-world) matrix and intrinsics (fx, fy, cx, cy)
 * to project world-space court keypoints into pixel coordinates.
 *
 * Coordinate convention:
 *   c2w is NeRF/COLMAP convention where the camera looks along +Z in camera space.
 *   w2c = inverse(c2w).  After w2c transform, z > 0 means the point is in front
 *   of the camera.
 */

import type { CameraModel } from '@/data/models';

// ── Types ───────────────────────────────────────────────────────────────────

export interface Point2D {
  u: number;
  v: number;
  /** Whether the point is in front of the camera and within reasonable bounds. */
  valid: boolean;
}

export interface Line2D {
  from: Point2D;
  to: Point2D;
}

export interface CourtProjection {
  /** Projected skeleton line segments (both endpoints must be valid). */
  lines: Line2D[];
  /** All projected keypoints. */
  keypoints: Point2D[];
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Invert a 4×4 row-major matrix (rigid body: R^T, -R^T @ t). */
function invertC2W(c2w: number[][]): number[][] {
  // R = c2w[:3,:3],  t = c2w[:3,3]
  // w2c[:3,:3] = R^T,  w2c[:3,3] = -R^T @ t
  const R = [
    [c2w[0][0], c2w[0][1], c2w[0][2]],
    [c2w[1][0], c2w[1][1], c2w[1][2]],
    [c2w[2][0], c2w[2][1], c2w[2][2]],
  ];
  const t = [c2w[0][3], c2w[1][3], c2w[2][3]];

  // R^T
  const Rt = [
    [R[0][0], R[1][0], R[2][0]],
    [R[0][1], R[1][1], R[2][1]],
    [R[0][2], R[1][2], R[2][2]],
  ];

  // -R^T @ t
  const nt = [
    -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
  ];

  return [
    [Rt[0][0], Rt[0][1], Rt[0][2], nt[0]],
    [Rt[1][0], Rt[1][1], Rt[1][2], nt[1]],
    [Rt[2][0], Rt[2][1], Rt[2][2], nt[2]],
    [0, 0, 0, 1],
  ];
}

/** Project a single 3D world point to 2D pixel coordinates. */
function projectPoint(
  pWorld: number[],
  w2c: number[][],
  fx: number,
  fy: number,
  cx: number,
  cy: number,
  imgW: number,
  imgH: number,
): Point2D {
  // Transform to camera space
  const x = w2c[0][0] * pWorld[0] + w2c[0][1] * pWorld[1] + w2c[0][2] * pWorld[2] + w2c[0][3];
  const y = w2c[1][0] * pWorld[0] + w2c[1][1] * pWorld[1] + w2c[1][2] * pWorld[2] + w2c[1][3];
  const z = w2c[2][0] * pWorld[0] + w2c[2][1] * pWorld[1] + w2c[2][2] * pWorld[2] + w2c[2][3];

  if (z <= 0.001) {
    return { u: 0, v: 0, valid: false };
  }

  const u = fx * (x / z) + cx;
  const v = fy * (y / z) + cy;

  // Allow some margin outside the image (50%) for lines that cross the boundary
  const margin = Math.max(imgW, imgH) * 0.5;
  const valid = u > -margin && u < imgW + margin && v > -margin && v < imgH + margin;

  return { u, v, valid };
}

// ── Main projection function ────────────────────────────────────────────────

/**
 * Project court keypoints (world coordinates) onto the camera image.
 *
 * @param kpWorld      World-coordinate keypoints (e.g. 20 points per court).
 * @param skeleton     Connectivity pairs (indices into kpWorld).
 * @param camera       Camera model with c2w, intrinsics.
 * @param useRefined   If true, use c2wRefined instead of c2w (court-result mode).
 */
export function projectCourtToImage(
  kpWorld: number[][],
  skeleton: [number, number][],
  camera: CameraModel,
  useRefined = false,
): CourtProjection {
  const c2w = (useRefined && camera.c2wRefined) ? camera.c2wRefined : camera.c2w;
  const w2c = invertC2W(c2w);

  const { fx, fy, cx, cy, width, height } = camera;

  // Project all keypoints
  const keypoints: Point2D[] = kpWorld.map((p) =>
    projectPoint(p, w2c, fx, fy, cx, cy, width, height),
  );

  // Build line segments — only if both endpoints are valid (in front of camera)
  const lines: Line2D[] = [];
  for (const [i, j] of skeleton) {
    if (i < keypoints.length && j < keypoints.length) {
      const from = keypoints[i];
      const to = keypoints[j];
      if (from.valid && to.valid) {
        lines.push({ from, to });
      }
    }
  }

  return { lines, keypoints };
}
