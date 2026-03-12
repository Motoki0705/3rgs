/** PointCloudLayer — renders point cloud as THREE.Points. */

import React, { useMemo } from 'react';
import * as THREE from 'three';
import type { PointCloudModel } from '@/data/models';

interface PointCloudLayerProps {
  data: PointCloudModel;
  pointSize?: number;
  visible?: boolean;
}

export const PointCloudLayer: React.FC<PointCloudLayerProps> = ({
  data,
  pointSize = 0.012,
  visible = true,
}) => {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(data.colors, 3));
    return geo;
  }, [data]);

  return (
    <points geometry={geometry} visible={visible}>
      <pointsMaterial
        size={pointSize}
        vertexColors
        sizeAttenuation
      />
    </points>
  );
};
