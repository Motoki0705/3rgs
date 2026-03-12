/** useWarpAnimation — smooth camera warp with smoothstep interpolation. */

import { useRef, useCallback } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { WARP_DURATION } from '@/config/navigationConfig';
import { smoothstep, extractCameraTransform } from '@/data/models';

interface WarpState {
  active: boolean;
  startPos: THREE.Vector3;
  startQ: THREE.Quaternion;
  endPos: THREE.Vector3;
  endQ: THREE.Quaternion;
  t: number;
  onDone?: () => void;
}

export function useWarpAnimation(onWarpDone?: () => void) {
  const warp = useRef<WarpState>({
    active: false,
    startPos: new THREE.Vector3(),
    startQ: new THREE.Quaternion(),
    endPos: new THREE.Vector3(),
    endQ: new THREE.Quaternion(),
    t: 0,
  });

  /** Start warping the current camera to the given c2w. */
  const warpTo = useCallback(
    (c2w: number[][], camera: THREE.Camera, afterDone?: () => void) => {
      const { pos, q } = extractCameraTransform(c2w);
      const w = warp.current;
      w.startPos.copy(camera.position);
      w.startQ.copy(camera.quaternion);
      w.endPos.copy(pos);
      w.endQ.copy(q);
      w.t = 0;
      w.active = true;
      w.onDone = afterDone;
    },
    [],
  );

  const isWarping = useCallback(() => warp.current.active, []);

  useFrame((state, delta) => {
    const w = warp.current;
    if (!w.active) return;

    w.t += delta / WARP_DURATION;
    const t = Math.min(w.t, 1.0);
    const s = smoothstep(t);

    const cam = state.camera;
    cam.position.lerpVectors(w.startPos, w.endPos, s);
    cam.quaternion.slerpQuaternions(w.startQ, w.endQ, s);

    if (t >= 1.0) {
      w.active = false;
      cam.position.copy(w.endPos);
      cam.quaternion.copy(w.endQ);
      w.onDone?.();
      onWarpDone?.();
    }
  });

  return { warpTo, isWarping };
}
