/** CameraLayer — renders camera frustums, dots, and click-target spheres. */

import React, { useMemo, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import type { CameraModel } from '@/data/models';
import { c2wToMatrix4 } from '@/data/models';
import {
  COLOR_ORIG,
  COLOR_REFINED,
  COLOR_TEST,
  COLOR_MAST3R,
  COLOR_SELECTED,
} from '@/config/viewerTheme';

// ── Frustum geometry builder ────────────────────────────────────────────────

function buildFrustumPositions(fovX: number, fovY: number, depth: number): Float32Array {
  const hw = Math.tan(fovX / 2) * depth;
  const hh = Math.tan(fovY / 2) * depth;
  return new Float32Array([
    0, 0, 0, hw, hh, depth,
    0, 0, 0, -hw, hh, depth,
    0, 0, 0, hw, -hh, depth,
    0, 0, 0, -hw, -hh, depth,
    hw, hh, depth, -hw, hh, depth,
    -hw, hh, depth, -hw, -hh, depth,
    -hw, -hh, depth, hw, -hh, depth,
    hw, -hh, depth, hw, hh, depth,
  ]);
}

// ── Types ───────────────────────────────────────────────────────────────────

export type CameraVariant = 'orig' | 'refined' | 'test' | 'mast3r';

interface CameraLayerProps {
  cameras: CameraModel[];
  /** Which variant(s) to show. */
  variants: CameraVariant[];
  /** Frustum depth. */
  frustumDepth?: number;
  /** Dot radius. */
  dotRadius?: number;
  /** Click sphere radius. */
  clickRadius?: number;
  /** Currently selected camera index. */
  selectedIdx?: number | null;
  /** Called when a camera sphere is clicked. */
  onSelect?: (idx: number, variant: CameraVariant) => void;
  /** Visibility per variant. */
  visibleOrig?: boolean;
  visibleRefined?: boolean;
}

const VARIANT_COLOR: Record<CameraVariant, THREE.Color> = {
  orig: COLOR_ORIG,
  refined: COLOR_REFINED,
  test: COLOR_TEST,
  mast3r: COLOR_MAST3R,
};

interface CamRenderItem {
  camIdx: number;
  variant: CameraVariant;
  fovX: number;
  fovY: number;
  matrix: THREE.Matrix4;
  position: THREE.Vector3;
  color: THREE.Color;
}

export const CameraLayer: React.FC<CameraLayerProps> = ({
  cameras,
  variants,
  frustumDepth = 0.12,
  dotRadius = 0.026,
  clickRadius = 0.045,
  selectedIdx = null,
  onSelect,
  visibleOrig = true,
  visibleRefined = true,
}) => {
  const items = useMemo(() => {
    const result: CamRenderItem[] = [];
    for (let i = 0; i < cameras.length; i++) {
      const cam = cameras[i];
      const fovX = 2 * Math.atan(cam.width / (2 * cam.fx));
      const fovY = 2 * Math.atan(cam.height / (2 * cam.fy));

      for (const variant of variants) {
        let c2w: number[][] | undefined;
        let shouldInclude = false;

        if (variant === 'orig' && cam.isTrain) {
          c2w = cam.c2w;
          shouldInclude = true;
        } else if (variant === 'refined' && cam.c2wRefined) {
          c2w = cam.c2wRefined;
          shouldInclude = true;
        } else if (variant === 'test' && !cam.isTrain) {
          c2w = cam.c2w;
          shouldInclude = true;
        } else if (variant === 'mast3r' && cam.isTrain) {
          c2w = cam.c2w;
          shouldInclude = true;
        }

        if (shouldInclude && c2w) {
          const mat = c2wToMatrix4(c2w);
          const pos = new THREE.Vector3().setFromMatrixPosition(mat);
          result.push({
            camIdx: i,
            variant,
            fovX,
            fovY,
            matrix: mat,
            position: pos,
            color: VARIANT_COLOR[variant],
          });
        }
      }
    }
    return result;
  }, [cameras, variants]);

  const handleClick = useCallback(
    (e: any) => {
      e.stopPropagation();
      const obj = e.object as THREE.Mesh;
      if (obj.userData.camIdx !== undefined && onSelect) {
        onSelect(obj.userData.camIdx, obj.userData.variant);
      }
    },
    [onSelect],
  );

  return (
    <group>
      {items.map((item, idx) => {
        const isSelected = selectedIdx === item.camIdx;
        const dotColor = isSelected ? COLOR_SELECTED : item.color;
        const visible =
          (item.variant === 'orig' || item.variant === 'test' || item.variant === 'mast3r')
            ? visibleOrig
            : visibleRefined;

        const frustumPositions = buildFrustumPositions(item.fovX, item.fovY, frustumDepth);
        const frustumGeo = new THREE.BufferGeometry();
        frustumGeo.setAttribute('position', new THREE.BufferAttribute(frustumPositions, 3));

        return (
          <group key={`${item.variant}-${item.camIdx}`} visible={visible}>
            {/* Frustum lines */}
            <lineSegments geometry={frustumGeo} matrixAutoUpdate={false} matrix={item.matrix}>
              <lineBasicMaterial color={item.color} />
            </lineSegments>

            {/* Dot */}
            <mesh position={item.position}>
              <sphereGeometry args={[dotRadius, 8, 8]} />
              <meshBasicMaterial color={dotColor} />
            </mesh>

            {/* Invisible click target */}
            {item.variant !== 'test' && (
              <mesh
                position={item.position}
                userData={{ camIdx: item.camIdx, variant: item.variant }}
                onClick={handleClick}
              >
                <sphereGeometry args={[clickRadius, 6, 6]} />
                <meshBasicMaterial transparent opacity={0} depthWrite={false} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
};
