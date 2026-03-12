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

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
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

    // Navigation speed should follow the playable region, not far outlier points.
    // Use a robust camera-spread radius when possible, and only fall back to a
    // point-cloud percentile when camera coverage is unavailable.
    let navigationRadius = 0;
    if (cameras.length > 0) {
      const cameraDistances = cameras.map((cam) => {
        const dx = cam.c2w[0][3] - center.x;
        const dy = cam.c2w[1][3] - center.y;
        const dz = cam.c2w[2][3] - center.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      });
      navigationRadius = percentile(cameraDistances, 0.8);
    } else if (pointCloud.count > 0) {
      const pointDistances: number[] = [];
      const step = Math.max(1, Math.floor(pointCloud.count / 5000));
      for (let i = 0; i < pointCloud.count; i += step) {
        const dx = pos[i * 3] - center.x;
        const dy = pos[i * 3 + 1] - center.y;
        const dz = pos[i * 3 + 2] - center.z;
        pointDistances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
      navigationRadius = percentile(pointDistances, 0.7);
    }
    navigationRadius = Math.max(navigationRadius, 1.5);

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

    onFit?.(navigationRadius);
  }, [camera, cameras, pointCloud, zUp, onFit]);

  return null;
};
