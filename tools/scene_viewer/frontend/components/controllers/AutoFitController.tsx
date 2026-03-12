/** AutoFitController — positions the camera to see the whole scene on first load. */

import React, { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { PITCH_MAX } from '@/config/navigationConfig';
import type { CameraModel, PointCloudModel } from '@/data/models';

interface AutoFitControllerProps {
  pointCloud: PointCloudModel;
  cameras: CameraModel[];
  zUp?: boolean;
  onFit?: (radius: number) => void;
}

export const AutoFitController: React.FC<AutoFitControllerProps> = ({
  pointCloud,
  cameras,
  zUp = false,
  onFit,
}) => {
  const { camera } = useThree();
  const fitted = useRef(false);

  useEffect(() => {
    if (fitted.current) return;
    if (pointCloud.count === 0 && cameras.length === 0) return;
    fitted.current = true;

    // Compute scene center from camera centroids
    let cx = 0, cy = 0, cz = 0;
    for (const cam of cameras) {
      cx += cam.c2w[0][3];
      cy += cam.c2w[1][3];
      cz += cam.c2w[2][3];
    }
    const n = cameras.length || 1;
    const center = new THREE.Vector3(cx / n, cy / n, cz / n);

    // Scene radius
    const pos = pointCloud.positions;
    let maxDist = 0;
    for (let i = 0; i < pointCloud.count; i++) {
      const dx = pos[i * 3] - center.x;
      const dy = pos[i * 3 + 1] - center.y;
      const dz = pos[i * 3 + 2] - center.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > maxDist) maxDist = d;
    }
    maxDist = Math.max(maxDist, 3);

    if (zUp) {
      camera.position.set(
        center.x,
        center.y - maxDist * 1.5,
        center.z + maxDist * 0.8,
      );
      camera.lookAt(center);
      // Sync Euler from quaternion for Z-up
      const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'ZXY');
      camera.quaternion.setFromEuler(euler);
    } else {
      // Y-up mode
      const offset = new THREE.Vector3(0.25, 0.55, 1.0).normalize().multiplyScalar(maxDist * 0.75);
      camera.position.copy(center).add(offset);
      // Set yaw/pitch from direction
      const dir = center.clone().sub(camera.position).normalize();
      const pitch = Math.asin(Math.max(-PITCH_MAX, Math.min(PITCH_MAX, dir.y)));
      const yaw = Math.atan2(-dir.x, -dir.z);
      camera.rotation.order = 'YXZ';
      camera.rotation.y = yaw;
      camera.rotation.x = pitch;
    }

    onFit?.(maxDist);
  }, [camera, cameras, pointCloud, zUp, onFit]);

  return null;
};
