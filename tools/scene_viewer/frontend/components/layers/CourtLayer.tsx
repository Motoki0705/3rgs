/** CourtLayer — draws court wireframes. Supports 1 or 2 courts, read-only or editable. */

import React, { useMemo } from 'react';
import * as THREE from 'three';
import type { CourtModel, Sim3Model, CourtPairModel } from '@/data/models';
import {
  courtToWorld,
  courtToWorldMatrix,
  offsetKeypoints,
  DOUBLES_WIDTH,
  ADJ_DIRS,
} from '@/data/models';
import { COLOR_COURT_1, COLOR_COURT_2 } from '@/config/viewerTheme';

interface CourtLayerProps {
  court: CourtModel;
  sim3: Sim3Model;
  visible?: boolean;
  /** court-result uses matrix mode. */
  useMatrix?: boolean;
  /** court_pair from court-result for adjacent offset calculation. */
  courtPair?: CourtPairModel;
}

function buildCourtLinePositions(
  kpWorld: number[][],
  skeleton: [number, number][],
): Float32Array {
  const positions: number[] = [];
  for (const [i, j] of skeleton) {
    const a = kpWorld[i];
    const b = kpWorld[j];
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  return new Float32Array(positions);
}

export const CourtLayer: React.FC<CourtLayerProps> = ({
  court,
  sim3,
  visible = true,
  useMatrix = false,
  courtPair,
}) => {
  const { geo1, geo2 } = useMemo(() => {
    const { keypointsCourt, skeleton } = court;
    const { scale, rotation, translation, adjacentGap, adjacentDirection, matrix } = sim3;

    // Court 1
    let kpWorld1: number[][];
    if (useMatrix && matrix) {
      kpWorld1 = courtToWorldMatrix(keypointsCourt, matrix);
    } else {
      kpWorld1 = courtToWorld(keypointsCourt, scale, rotation, translation);
    }
    const pos1 = buildCourtLinePositions(kpWorld1, skeleton);
    const g1 = new THREE.BufferGeometry();
    g1.setAttribute('position', new THREE.BufferAttribute(pos1, 3));

    // Court 2
    let adjOffset: number[];
    if (useMatrix && courtPair) {
      const off = courtPair.adjacentCenterOffset ?? courtPair.adjacent_center_offset ?? 0;
      const vec = courtPair.adjacentDirectionVector ?? courtPair.adjacent_direction_vector ?? [1, 0, 0];
      adjOffset = [off * vec[0], off * vec[1], off * vec[2]];
    } else {
      const adjDirVec = ADJ_DIRS[adjacentDirection] ?? [1, 0, 0];
      const adjShift = DOUBLES_WIDTH + adjacentGap;
      adjOffset = adjDirVec.map((d) => d * adjShift);
    }
    const kpCourt2 = offsetKeypoints(keypointsCourt, adjOffset);
    let kpWorld2: number[][];
    if (useMatrix && matrix) {
      kpWorld2 = courtToWorldMatrix(kpCourt2, matrix);
    } else {
      kpWorld2 = courtToWorld(kpCourt2, scale, rotation, translation);
    }
    const pos2 = buildCourtLinePositions(kpWorld2, skeleton);
    const g2 = new THREE.BufferGeometry();
    g2.setAttribute('position', new THREE.BufferAttribute(pos2, 3));

    return { geo1: g1, geo2: g2 };
  }, [court, sim3, useMatrix, courtPair]);

  return (
    <group visible={visible}>
      <lineSegments geometry={geo1}>
        <lineBasicMaterial color={COLOR_COURT_1} />
      </lineSegments>
      <lineSegments geometry={geo2}>
        <lineBasicMaterial color={COLOR_COURT_2} />
      </lineSegments>
    </group>
  );
};
