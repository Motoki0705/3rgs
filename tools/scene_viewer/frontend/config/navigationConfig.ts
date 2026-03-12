/** Navigation / camera control configuration.
 *  All tuning constants for WASD movement, look sensitivity, warp, auto-fit
 *  are centralised here so adjustments only need one file.
 */

// ── Movement speed ──────────────────────────────────────────────────────────
/** Base move speed multiplier, applied per frame to camera position. */
export const MOVE_SPEED_MID = 0.08;

/** Compute speed tiers from scene radius. */
export function speedsFromRadius(radius: number) {
  const base = Math.max(radius * 0.015, 0.02);
  return {
    low: base * 0.5,
    mid: base,
    high: base * 2.0,
  };
}

// ── Look sensitivity ────────────────────────────────────────────────────────
/** Mouse look sensitivity (radians per pixel). */
export const LOOK_SENSITIVITY = 0.003;

// ── Warp ────────────────────────────────────────────────────────────────────
/** Duration of camera warp animation in seconds. */
export const WARP_DURATION = 0.9;

// ── Scroll ──────────────────────────────────────────────────────────────────
/** Scroll movement multiplier (relative to current move speed). */
export const SCROLL_MULTIPLIER = 8;

// ── Auto-fit ────────────────────────────────────────────────────────────────
/** Camera offset scale when auto-fitting to bounding box (relative to size). */
export const AUTO_FIT_DISTANCE_FACTOR = 0.75;
/** The normalised offset direction for auto-fit placement. */
export const AUTO_FIT_OFFSET_DIR: [number, number, number] = [0.25, 0.55, 1.0];

// ── Frustum rendering ───────────────────────────────────────────────────────
export const FRUSTUM_DEPTH_SCENE = 0.12;
export const FRUSTUM_DEPTH_COURT_INIT = 0.3;
export const FRUSTUM_DEPTH_COURT_RESULT = 0.08;

// ── Point sizes ─────────────────────────────────────────────────────────────
export const POINT_SIZE_DEFAULT = 0.012;
export const POINT_SIZE_MAST3R = 0.025;

// ── Dot sizes ───────────────────────────────────────────────────────────────
export const DOT_RADIUS_SMALL = 0.018;
export const DOT_RADIUS_DEFAULT = 0.026;
export const DOT_RADIUS_COURT_INIT = 0.055;
export const DOT_RADIUS_COURT_RESULT = 0.035;

// ── Click sphere sizes ──────────────────────────────────────────────────────
export const CLICK_SPHERE_RADIUS_SCENE = 0.045;
export const CLICK_SPHERE_RADIUS_COURT_INIT = 0.18;
export const CLICK_SPHERE_RADIUS_COURT_RESULT = 0.15;

// ── Pitch clamp ─────────────────────────────────────────────────────────────
export const PITCH_MAX = Math.PI * 0.499;
