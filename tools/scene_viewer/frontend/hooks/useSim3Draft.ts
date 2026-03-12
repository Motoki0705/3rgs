/** useSim3Draft — mutable Sim(3) editing state for court-init mode. */

import { useState, useCallback, useMemo } from 'react';
import type { Sim3Model } from '@/data/models';
import { eulerZYXtoMatrix, matrixToEulerZYX } from '@/data/models';

export interface Sim3DraftState {
  tx: number;
  ty: number;
  tz: number;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  logScale: number;
  adjacentGap: number;
  adjacentDirection: string;
}

function sim3ToDraft(sim3: Sim3Model): Sim3DraftState {
  const { yaw, pitch, roll } = matrixToEulerZYX(sim3.rotation);
  return {
    tx: sim3.translation[0],
    ty: sim3.translation[1],
    tz: sim3.translation[2],
    yawDeg: yaw * 180 / Math.PI,
    pitchDeg: pitch * 180 / Math.PI,
    rollDeg: roll * 180 / Math.PI,
    logScale: Math.log10(Math.max(sim3.scale, 1e-6)),
    adjacentGap: sim3.adjacentGap,
    adjacentDirection: sim3.adjacentDirection,
  };
}

function draftToSim3(draft: Sim3DraftState): Sim3Model {
  const yaw = draft.yawDeg * Math.PI / 180;
  const pitch = draft.pitchDeg * Math.PI / 180;
  const roll = draft.rollDeg * Math.PI / 180;
  return {
    scale: Math.pow(10, draft.logScale),
    rotation: eulerZYXtoMatrix(yaw, pitch, roll),
    translation: [draft.tx, draft.ty, draft.tz],
    adjacentGap: draft.adjacentGap,
    adjacentDirection: draft.adjacentDirection,
  };
}

export function useSim3Draft(initial: Sim3Model) {
  const [draft, setDraft] = useState<Sim3DraftState>(() => sim3ToDraft(initial));

  const update = useCallback((partial: Partial<Sim3DraftState>) => {
    setDraft((prev) => ({ ...prev, ...partial }));
  }, []);

  const resetTo = useCallback((sim3: Sim3Model) => {
    setDraft(sim3ToDraft(sim3));
  }, []);

  const sim3 = useMemo<Sim3Model>(() => draftToSim3(draft), [draft]);

  return { draft, update, resetTo, sim3 };
}
