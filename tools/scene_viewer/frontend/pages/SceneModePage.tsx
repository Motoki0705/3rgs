/** SceneModePage — SfM point cloud & camera viewer (Y-up, FlyNav). */

import React, { useCallback, useRef, useState } from 'react';
import { useSceneQuery } from '@/data/useSceneQuery';
import { useSelection } from '@/hooks/useSelection';
import { CanvasViewport } from '@/components/viewport/CanvasViewport';
import { SceneRoot } from '@/components/viewport/SceneRoot';
import { PointCloudLayer } from '@/components/layers/PointCloudLayer';
import { CameraLayer, type CameraVariant } from '@/components/layers/CameraLayer';
import { FlyNavigationController } from '@/components/controllers/FlyNavigationController';
import { AutoFitController } from '@/components/controllers/AutoFitController';
import { WarpController, type WarpControllerRef } from '@/components/controllers/WarpController';
import { LoadingOverlay } from '@/components/overlays/LoadingOverlay';
import { ErrorOverlay } from '@/components/overlays/ErrorOverlay';
import { StatsOverlay } from '@/components/overlays/StatsOverlay';
import { LegendOverlay } from '@/components/overlays/LegendOverlay';
import { NavigationHelpOverlay } from '@/components/overlays/NavigationHelpOverlay';
import { CameraInspectorPanel } from '@/components/panels/CameraInspectorPanel';
import { ModeHeader } from '@/components/panels/ModeHeader';
import { FRUSTUM_DEPTH_SCENE, DOT_RADIUS_DEFAULT, CLICK_SPHERE_RADIUS_SCENE } from '@/config/navigationConfig';
import {
  CSS_ORIG,
  CSS_REFINED,
  CSS_TEST,
  CSS_MAST3R,
  CSS_DELTA,
} from '@/config/viewerTheme';

const PANEL_WIDTH = 320;

export const SceneModePage: React.FC = () => {
  const { data, isLoading, error } = useSceneQuery();
  const { selectedIdx, select, deselect } = useSelection();
  const warpRef = useRef<WarpControllerRef>(null);
  const flyCtrlRef = useRef<{ syncFromCamera: () => void } | null>(null);

  const [sceneRadius, setSceneRadius] = useState(5);

  const handleAutoFit = useCallback((r: number) => setSceneRadius(r), []);

  const handleCameraSelect = useCallback(
    (idx: number, _variant: CameraVariant) => select(idx),
    [select],
  );

  const handleWarp = useCallback(
    (c2w: number[][]) => {
      warpRef.current?.warpTo(c2w, () => {
        flyCtrlRef.current?.syncFromCamera();
      });
    },
    [],
  );

  const handleClose = useCallback(() => deselect(), [deselect]);

  if (isLoading) return <LoadingOverlay message="シーンデータを読み込み中..." />;
  if (error || !data) return <ErrorOverlay message={String(error ?? 'データ取得失敗')} />;

  const panelOpen = selectedIdx !== null;
  const selectedCam = selectedIdx !== null ? data.cameras[selectedIdx] : null;

  const isMast3r = data.mode === 'mast3r';
  const hasRefined = data.cameras.some((c) => c.c2wRefined);

  const variants: CameraVariant[] = isMast3r
    ? ['mast3r']
    : hasRefined
      ? ['orig', 'refined']
      : ['orig', 'test'];

  const legendItems = isMast3r
    ? [{ color: CSS_MAST3R, label: 'MASt3R カメラ', shape: 'dot' as const }]
    : hasRefined
      ? [
          { color: CSS_ORIG, label: '元カメラ (SfM)', shape: 'dot' as const },
          { color: CSS_REFINED, label: '最適化カメラ', shape: 'dot' as const },
          { color: CSS_DELTA, label: 'Δ 移動', shape: 'line' as const },
        ]
      : [
          { color: CSS_ORIG, label: '学習カメラ', shape: 'dot' as const },
          { color: CSS_TEST, label: 'テストカメラ', shape: 'dot' as const },
        ];

  return (
    <>
      <ModeHeader mode="scene" />

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
          <WarpController ref={warpRef} onWarpDone={() => flyCtrlRef.current?.syncFromCamera()} />

          {/* Layers */}
          <PointCloudLayer data={data.pointCloud} />
          <CameraLayer
            cameras={data.cameras}
            variants={variants}
            frustumDepth={FRUSTUM_DEPTH_SCENE}
            dotRadius={DOT_RADIUS_DEFAULT}
            clickRadius={CLICK_SPHERE_RADIUS_SCENE}
            selectedIdx={selectedIdx}
            onSelect={handleCameraSelect}
          />
        </SceneRoot>
      </CanvasViewport>

      {/* Overlays */}
      <StatsOverlay pointCount={data.pointCloud.count} cameraCount={data.cameras.length} />
      <LegendOverlay items={legendItems} />
      <NavigationHelpOverlay />

      {/* Side panel */}
      {panelOpen && selectedCam && (
        <CameraInspectorPanel
          camera={selectedCam}
          onWarp={handleWarp}
          onClose={handleClose}
        />
      )}
    </>
  );
};
