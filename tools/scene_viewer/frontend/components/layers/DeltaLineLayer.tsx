/** DeltaLineLayer — draws origin → refined camera delta lines. */

import React, { useMemo } from 'react';
import * as THREE from 'three';
import type { CameraModel } from '@/data/models';
import { c2wToMatrix4 } from '@/data/models';
import { COLOR_DELTA } from '@/config/viewerTheme';

interface DeltaLineLayerProps {
  cameras: CameraModel[];
  visible?: boolean;
}

export const DeltaLineLayer: React.FC<DeltaLineLayerProps> = ({
  cameras,
  visible = true,
}) => {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    for (const cam of cameras) {
      if (!cam.c2wRefined) continue;
      const matOrig = c2wToMatrix4(cam.c2w);
      const matRef = c2wToMatrix4(cam.c2wRefined);
      const posOrig = new THREE.Vector3().setFromMatrixPosition(matOrig);
      const posRef = new THREE.Vector3().setFromMatrixPosition(matRef);
      positions.push(
        posOrig.x, posOrig.y, posOrig.z,
        posRef.x, posRef.y, posRef.z,
      );
    }
    const geo = new THREE.BufferGeometry();
    if (positions.length > 0) {
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    }
    return geo;
  }, [cameras]);

  if (geometry.attributes.position === undefined) return null;

  return (
    <lineSegments geometry={geometry} visible={visible}>
      <lineBasicMaterial color={COLOR_DELTA} transparent opacity={0.6} />
    </lineSegments>
  );
};
