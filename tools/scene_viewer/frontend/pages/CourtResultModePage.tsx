/** CourtResultModePage — result viewer with visibility toggles & metrics. */

import React, { useCallback, useRef, useState, useMemo } from 'react';
import { useCourtResultQuery } from '@/data/useSceneQuery';
import { useSelection } from '@/hooks/useSelection';
import { useVisibilityState } from '@/hooks/useVisibilityState';
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
import { LegendOverlay } from '@/components/overlays/LegendOverlay';
import { NavigationHelpOverlay } from '@/components/overlays/NavigationHelpOverlay';
import { CameraInspectorPanel } from '@/components/panels/CameraInspectorPanel';
import { CameraListPanel } from '@/components/panels/CameraListPanel';
import { MetricsPanel } from '@/components/panels/MetricsPanel';
import { VisibilityToolbar } from '@/components/panels/VisibilityToolbar';
import { ModeHeader } from '@/components/panels/ModeHeader';
import { FRUSTUM_DEPTH_COURT_RESULT, DOT_RADIUS_COURT_RESULT, CLICK_SPHERE_RADIUS_COURT_RESULT } from '@/config/navigationConfig';
import {
  CSS_ORIG,
  CSS_COURT1,
  CSS_COURT2,
} from '@/config/viewerTheme';
import type { CourtOverlayData } from '@/components/panels/CameraInspectorPanel';
import {
  courtToWorldMatrix,
  offsetKeypoints,
} from '@/data/models';

const PANEL_WIDTH = 340;

export const CourtResultModePage: React.FC = () => {
  const { data, isLoading, error } = useCourtResultQuery();
  const { selectedIdx, select, deselect } = useSelection();
  const { flags, toggle } = useVisibilityState();
  const warpRef = useRef<WarpControllerRef>(null);
  const flyCtrlRef = useRef<{ syncFromCamera: () => void } | null>(null);

  const [sceneRadius, setSceneRadius] = useState(5);
  const handleAutoFit = useCallback((r: number) => setSceneRadius(r), []);

  // Compute court world keypoints for image overlay
  const courtOverlay = useMemo<CourtOverlayData | undefined>(() => {
    if (!data) return undefined;
    const { court, sim3, courtPair } = data;
    const { keypointsCourt, skeleton } = court;
    const { matrix } = sim3;
    if (!matrix) return undefined;

    const kpWorld1 = courtToWorldMatrix(keypointsCourt, matrix);

    const off = courtPair.adjacentCenterOffset ?? courtPair.adjacent_center_offset ?? 0;
    const vec = courtPair.adjacentDirectionVector ?? courtPair.adjacent_direction_vector ?? [1, 0, 0];
    const adjOffset = [off * vec[0], off * vec[1], off * vec[2]];
    const kpCourt2 = offsetKeypoints(keypointsCourt, adjOffset);
    const kpWorld2 = courtToWorldMatrix(kpCourt2, matrix);

    return {
      courtWorldKp1: kpWorld1,
      courtWorldKp2: kpWorld2,
      skeleton,
      useRefined: true,
    };
  }, [data]);

  const handleWarp = useCallback(
    (c2w: number[][]) => {
      warpRef.current?.warpTo(c2w, () => {
        flyCtrlRef.current?.syncFromCamera();
      });
    },
    [],
  );

  if (isLoading) return <LoadingOverlay message="結果データを読み込み中..." />;
  if (error || !data) return <ErrorOverlay message={String(error ?? 'データ取得失敗')} />;

  const panelOpen = selectedIdx !== null;
  const selectedCam = selectedIdx !== null ? data.cameras[selectedIdx] : null;

  const legendItems = [
    { color: CSS_ORIG, label: '最適化カメラ', shape: 'dot' as const },
    { color: CSS_COURT1, label: 'コート 1', shape: 'line' as const },
    { color: CSS_COURT2, label: 'コート 2', shape: 'line' as const },
  ];

  return (
    <>
      <ModeHeader mode="court-result" />
      <VisibilityToolbar flags={flags} onToggle={toggle} />

      <CanvasViewport panelWidth={panelOpen ? PANEL_WIDTH : 0} zUp={false}>
        <SceneRoot>
          {/* Controllers */}
          <FlyNavigationController
            zUp={false}
            sceneRadius={sceneRadius}
            onReady={(ctrl) => { flyCtrlRef.current = ctrl; }}
          />
          <AutoFitController
            pointCloud={data.pointCloud}
            cameras={data.cameras}
            zUp={false}
            onFit={handleAutoFit}
          />
          <WarpController
            ref={warpRef}
            onWarpDone={() => flyCtrlRef.current?.syncFromCamera()}
          />

          {/* Layers */}
          <PointCloudLayer data={data.pointCloud} visible={flags.pointCloud} />
          <CameraLayer
            cameras={data.cameras}
            variants={['refined']}
            frustumDepth={FRUSTUM_DEPTH_COURT_RESULT}
            dotRadius={DOT_RADIUS_COURT_RESULT}
            clickRadius={CLICK_SPHERE_RADIUS_COURT_RESULT}
            selectedIdx={selectedIdx}
            onSelect={(idx) => select(idx)}
            visibleRefined={flags.origCameras}
          />
          <CourtLayer
            court={data.court}
            sim3={data.sim3}
            visible={flags.court}
            useMatrix
            courtPair={data.courtPair}
          />
        </SceneRoot>
      </CanvasViewport>

      {/* Overlays */}
      <StatsOverlay pointCount={data.pointCloud.count} cameraCount={data.cameras.length} />
      <LegendOverlay items={legendItems} />
      <NavigationHelpOverlay />

      {/* Right panel: inspector or camera list + metrics */}
      {panelOpen && selectedCam ? (
        <CameraInspectorPanel
          camera={selectedCam}
          onWarp={handleWarp}
          onClose={deselect}
          courtOverlay={courtOverlay}
        />
      ) : (
        <div style={styles.rightPanel}>
          <MetricsPanel metrics={data.metrics} sim3={data.sim3} />
          <CameraListPanel
            cameras={data.cameras}
            selectedIdx={selectedIdx}
            meanDelta={data.metrics.meanDelta}
            onSelect={select}
          />
        </div>
      )}
    </>
  );
};

const styles: Record<string, React.CSSProperties> = {
  rightPanel: {
    position: 'fixed',
    top: 0,
    right: 0,
    width: PANEL_WIDTH,
    height: '100vh',
    background: 'rgba(14,14,14,0.94)',
    backdropFilter: 'blur(6px)',
    borderLeft: '1px solid #2a2a2a',
    display: 'flex',
    flexDirection: 'column',
    zIndex: 10,
    overflowY: 'auto',
  },
};
