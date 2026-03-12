/** useFlyNavigation — WASD + right-click-drag flight camera controller.
 *
 *  Manages key/mouse state and updates the R3F camera each frame.
 *  Works with both Y-up (scene mode) and Z-up (court modes) worlds.
 */

import { useRef, useEffect, useCallback } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import {
  LOOK_SENSITIVITY,
  SCROLL_MULTIPLIER,
  PITCH_MAX,
  speedsFromRadius,
} from '@/config/navigationConfig';

export interface FlyNavigationOptions {
  /** When true the world up axis is Z (court modes). Default: false (Y-up). */
  zUp?: boolean;
  /** Whether the controller is enabled. */
  enabled?: boolean;
}

interface FlyState {
  yaw: number;
  pitch: number;
  isRightDragging: boolean;
  prevMx: number;
  prevMy: number;
  keys: Record<string, boolean>;
  speeds: { low: number; mid: number; high: number };
  moveSpeed: number;
}

export function useFlyNavigation(opts: FlyNavigationOptions = {}) {
  const { zUp = false, enabled = true } = opts;
  const { camera, gl } = useThree();
  const state = useRef<FlyState>({
    yaw: 0,
    pitch: 0,
    isRightDragging: false,
    prevMx: 0,
    prevMy: 0,
    keys: {},
    speeds: speedsFromRadius(5),
    moveSpeed: speedsFromRadius(5).mid,
  });

  const rotationOrder = zUp ? 'ZXY' : 'YXZ';

  /** Apply current yaw/pitch to camera. */
  const applyRotation = useCallback(() => {
    const s = state.current;
    s.pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, s.pitch));
    if (zUp) {
      camera.quaternion.setFromEuler(new THREE.Euler(s.pitch, 0, s.yaw, 'ZXY'));
    } else {
      camera.rotation.order = 'YXZ';
      camera.rotation.y = s.yaw;
      camera.rotation.x = s.pitch;
    }
  }, [camera, zUp]);

  /** Sync yaw/pitch from camera quaternion (e.g. after a warp). */
  const syncFromCamera = useCallback(() => {
    const s = state.current;
    const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, rotationOrder);
    if (zUp) {
      s.yaw = euler.z;
      s.pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, euler.x));
    } else {
      s.yaw = euler.y;
      s.pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, euler.x));
    }
  }, [camera, zUp, rotationOrder]);

  /** Update movement speeds from scene radius. */
  const setSceneRadius = useCallback((radius: number) => {
    state.current.speeds = speedsFromRadius(radius);
    state.current.moveSpeed = state.current.speeds.mid;
  }, []);

  // ── Event listeners ─────────────────────────────────────────────────────

  useEffect(() => {
    if (!enabled) return;
    const s = state.current;
    const canvas = gl.domElement;

    function isEditingForm(): boolean {
      const el = document.activeElement;
      return !!el && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(el.tagName);
    }

    function onKeyDown(e: KeyboardEvent) {
      if (isEditingForm()) {
        if (e.key === 'Escape') (document.activeElement as HTMLElement)?.blur();
        return;
      }
      s.keys[e.key.toLowerCase()] = true;
    }

    function onKeyUp(e: KeyboardEvent) {
      s.keys[e.key.toLowerCase()] = false;
    }

    function onContextMenu(e: Event) {
      e.preventDefault();
    }

    function onMouseDown(e: MouseEvent) {
      if (e.button === 2) {
        s.isRightDragging = true;
        s.prevMx = e.clientX;
        s.prevMy = e.clientY;
        canvas.style.cursor = 'grabbing';
      }
    }

    function onMouseMove(e: MouseEvent) {
      if (!s.isRightDragging) return;
      const dx = e.clientX - s.prevMx;
      const dy = e.clientY - s.prevMy;
      s.prevMx = e.clientX;
      s.prevMy = e.clientY;
      s.yaw -= dx * LOOK_SENSITIVITY;
      s.pitch -= dy * LOOK_SENSITIVITY;
      applyRotation();
    }

    function onMouseUp(e: MouseEvent) {
      if (e.button === 2) {
        s.isRightDragging = false;
        canvas.style.cursor = 'crosshair';
      }
    }

    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const fwd = new THREE.Vector3();
      camera.getWorldDirection(fwd);
      const sign = e.deltaY > 0 ? -1 : 1;
      camera.position.addScaledVector(fwd, sign * s.moveSpeed * SCROLL_MULTIPLIER);
    }

    function onBlur() {
      Object.keys(s.keys).forEach((k) => (s.keys[k] = false));
      s.isRightDragging = false;
      s.moveSpeed = s.speeds.mid;
      canvas.style.cursor = 'crosshair';
    }

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    canvas.addEventListener('contextmenu', onContextMenu);
    canvas.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('blur', onBlur);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      canvas.removeEventListener('wheel', onWheel);
      window.removeEventListener('blur', onBlur);
    };
  }, [enabled, gl, camera, applyRotation]);

  // ── Per-frame movement ──────────────────────────────────────────────────

  useFrame(() => {
    if (!enabled) return;
    const s = state.current;
    const speed = s.moveSpeed;

    let fwd: THREE.Vector3;
    let right: THREE.Vector3;
    let up: THREE.Vector3;

    if (zUp) {
      fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      up = new THREE.Vector3(0, 0, 1);
    } else {
      fwd = new THREE.Vector3(0, 0, -1).applyEuler(camera.rotation);
      right = new THREE.Vector3(1, 0, 0).applyEuler(camera.rotation);
      up = new THREE.Vector3(0, 1, 0);
    }

    if (s.keys['w'] || s.keys['arrowup']) camera.position.addScaledVector(fwd, speed);
    if (s.keys['s'] || s.keys['arrowdown']) camera.position.addScaledVector(fwd, -speed);
    if (s.keys['a'] || s.keys['arrowleft']) camera.position.addScaledVector(right, -speed);
    if (s.keys['d'] || s.keys['arrowright']) camera.position.addScaledVector(right, speed);
    if (s.keys[' ']) camera.position.addScaledVector(up, speed);
    if (s.keys['shift']) camera.position.addScaledVector(up, -speed);
  });

  return { syncFromCamera, setSceneRadius, stateRef: state };
}
