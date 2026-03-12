/** Shared colour palette for the viewer UI. */
import * as THREE from 'three';

// ── Scene colours ───────────────────────────────────────────────────────────
export const COLOR_ORIG     = new THREE.Color(0x4a90e2); // blue – original camera
export const COLOR_REFINED  = new THREE.Color(0xe97c2d); // orange – refined camera
export const COLOR_TEST     = new THREE.Color(0x666666); // grey – test camera
export const COLOR_MAST3R   = new THREE.Color(0x5ba85b); // green – mast3r camera
export const COLOR_SELECTED = new THREE.Color(0xff3333); // red – selected camera

export const COLOR_COURT_1 = 0x44cc44;  // green court wireframe
export const COLOR_COURT_2 = 0xeecc22;  // yellow adjacent court
export const COLOR_DELTA   = 0xcc44cc;  // purple delta line orig → refined

// ── CSS colours (for UI overlays) ───────────────────────────────────────────
export const CSS_BG      = '#0d0d0d';
export const CSS_PANEL   = 'rgba(14,14,14,0.94)';
export const CSS_BORDER  = '#2a2a2a';
export const CSS_TEXT     = '#e0e0e0';
export const CSS_MUTED   = '#888';
export const CSS_DIMMED  = '#555';
export const CSS_ACCENT  = '#4a90e2';

export const CSS_ORIG    = '#4a90e2';
export const CSS_REFINED = '#e97c2d';
export const CSS_TEST    = '#666666';
export const CSS_MAST3R  = '#5ba85b';
export const CSS_DELTA   = '#cc44cc';
export const CSS_COURT1  = '#44cc44';
export const CSS_COURT2  = '#eecc22';
