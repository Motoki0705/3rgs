/** CourtInitModePage — Sim(3) editor for court alignment (Z-up, FlyNav). */

import React, { useCallback, useRef, useState, useEffect } from 'react';
import { useCourtInitQuery } from '@/data/useSceneQuery';
import { useSelection } from '@/hooks/useSelection';
import { useSim3Draft } from '@/hooks/useSim3Draft';
import { CanvasViewport } from '@/components/viewport/CanvasViewport';
import { SceneRoot } from '@/components/viewport/SceneRoot';
import { PointCloudLayer } from '@/components/layers/PointCloudLayer';
import { CameraLayer } from '@/components/layers/CameraLayer';
import { CourtLayer } from '@/components/layers/CourtLayer';
import { FlyNavigationController } from '@/components/controllers/FlyNavigationController';
import { AutoFitController } from '@/components/controllers/AutoFitController';
import { WarpController, type WarpControllerRef } from '@/components/controllers/WarpController';
import { LoadingOverlay } from '@/components/overlays/LoadingOverlay';
import { ErrorOverlay } from '@/components/overlays/ErrorOverlay';
import { StatsOverlay } from '@/components/overlays/StatsOverlay';
import { NavigationHelpOverlay } from '@/components/overlays/NavigationHelpOverlay';
import { Sim3EditorPanel } from '@/components/panels/Sim3EditorPanel';
import { FRUSTUM_DEPTH_COURT_INIT, DOT_RADIUS_COURT_INIT, CLICK_SPHERE_RADIUS_COURT_INIT } from '@/config/navigationConfig';

const PANEL_WIDTH = 340;

export const CourtInitModePage: React.FC = () => {
  const { data, isLoading, error } = useCourtInitQuery();
  const { selectedIdx, select, deselect } = useSelection();
  const warpRef = useRef<WarpControllerRef>(null);
  const flyCtrlRef = useRef<{ syncFromCamera: () => void } | null>(null);

  const [sceneRadius, setSceneRadius] = useState(5);
  const handleAutoFit = useCallback((r: number) => setSceneRadius(r), []);

  // Sim3 draft — only initialize once data is ready
  const [draftReady, setDraftReady] = useState(false);
  const sim3Draft = useSim3Draft(
    data?.autoSim3 ?? {
      scale: 1,
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      translation: [0, 0, 0],
      adjacentGap: 3,
      adjacentDirection: '+x',
    },
  );

  // When data first arrives, reset draft to auto Sim3
  useEffect(() => {
    if (data && !draftReady) {
      sim3Draft.resetTo(data.autoSim3);
      setDraftReady(true);
    }
  }, [data, draftReady, sim3Draft]);

  const handleWarp = useCallback(
    (c2w: number[][]) => {
      warpRef.current?.warpTo(c2w, () => {
        flyCtrlRef.current?.syncFromCamera();
      });
    },
    [],
  );

  const handleReset = useCallback(() => {
    if (data) sim3Draft.resetTo(data.autoSim3);
  }, [data, sim3Draft]);

  if (isLoading) return <LoadingOverlay message="コートシーンを読み込み中..." />;
  if (error || !data) return <ErrorOverlay message={String(error ?? 'データ取得失敗')} />;

  const selectedCam = selectedIdx !== null ? data.cameras[selectedIdx] : null;

  return (
    <>
      <CanvasViewport panelWidth={PANEL_WIDTH} zUp>
        <SceneRoot>
          {/* Controllers */}
          <FlyNavigationController
            zUp
            sceneRadius={sceneRadius}
            onReady={(ctrl) => { flyCtrlRef.current = ctrl; }}
          />
          <AutoFitController
            pointCloud={data.pointCloud}
            cameras={data.cameras}
            zUp
            onFit={handleAutoFit}
          />
          <WarpController
            ref={warpRef}
            onWarpDone={() => flyCtrlRef.current?.syncFromCamera()}
          />

          {/* Layers */}
          <PointCloudLayer data={data.pointCloud} />
          <CameraLayer
            cameras={data.cameras}
            variants={['orig']}
            frustumDepth={FRUSTUM_DEPTH_COURT_INIT}
            dotRadius={DOT_RADIUS_COURT_INIT}
            clickRadius={CLICK_SPHERE_RADIUS_COURT_INIT}
            selectedIdx={selectedIdx}
            onSelect={(idx) => select(idx)}
          />
          <CourtLayer
            court={data.court}
            sim3={sim3Draft.sim3}
          />
        </SceneRoot>
      </CanvasViewport>

      {/* Overlays */}
      <StatsOverlay pointCount={data.pointCloud.count} cameraCount={data.cameras.length} />
      <NavigationHelpOverlay />

      {/* Editor panel */}
      <Sim3EditorPanel
        draft={sim3Draft.draft}
        sim3={sim3Draft.sim3}
        autoSim3={data.autoSim3}
        selectedCamera={selectedCam}
        onDraftChange={sim3Draft.update}
        onReset={handleReset}
        onWarp={handleWarp}
        court={data.court}
      />
    </>
  );
};
