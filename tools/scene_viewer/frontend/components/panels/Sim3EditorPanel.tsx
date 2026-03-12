/** Sim3EditorPanel — full right panel for court-init Sim(3) editing. */

import React, { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import type { Sim3Model, CameraModel, CourtModel } from '@/data/models';
import {
  courtToWorld,
  offsetKeypoints,
  DOUBLES_WIDTH,
  ADJ_DIRS,
} from '@/data/models';
import type { Sim3DraftState } from '@/hooks/useSim3Draft';
import { TransformSliderGroup } from './TransformSliderGroup';
import { AdjacentCourtEditor } from './AdjacentCourtEditor';
import { Sim3SummaryPanel } from './Sim3SummaryPanel';
import { SaveActions } from './SaveActions';
import { CourtOverlayImage } from './CourtOverlayImage';

interface Sim3EditorPanelProps {
  draft: Sim3DraftState;
  sim3: Sim3Model;
  autoSim3: Sim3Model;
  selectedCamera: CameraModel | null;
  onDraftChange: (partial: Partial<Sim3DraftState>) => void;
  onReset: () => void;
  onWarp?: (c2w: number[][]) => void;
  /** Court model for projecting court lines onto camera images. */
  court?: CourtModel;
}

export const Sim3EditorPanel: React.FC<Sim3EditorPanelProps> = ({
  draft,
  sim3,
  autoSim3,
  selectedCamera,
  onDraftChange,
  onReset,
  onWarp,
  court,
}) => {
  // Compute court world keypoints for overlay projection
  const courtWorldData = useMemo(() => {
    if (!court) return null;
    const { keypointsCourt, skeleton } = court;
    const { scale, rotation, translation, adjacentGap, adjacentDirection } = sim3;

    const kpWorld1 = courtToWorld(keypointsCourt, scale, rotation, translation);

    const adjDirVec = ADJ_DIRS[adjacentDirection] ?? [1, 0, 0];
    const adjShift = DOUBLES_WIDTH + adjacentGap;
    const adjOffset = adjDirVec.map((d) => d * adjShift);
    const kpCourt2 = offsetKeypoints(keypointsCourt, adjOffset);
    const kpWorld2 = courtToWorld(kpCourt2, scale, rotation, translation);

    return { kpWorld1, kpWorld2, skeleton };
  }, [court, sim3]);

  return (
    <div style={styles.panel}>
      {/* Header */}
      <div style={styles.header}>
        <h2 style={styles.heading}>Court Init — Sim(3) 調整</h2>
        <div style={styles.subtitle}>コートを点群に合わせてから「保存」してください</div>
      </div>

      {/* Sliders */}
      <TransformSliderGroup draft={draft} onChange={onDraftChange} />
      <AdjacentCourtEditor draft={draft} onChange={onDraftChange} />

      {/* Save / Reset */}
      <SaveActions sim3={sim3} onReset={onReset} />

      {/* Selected camera */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>選択中カメラ</div>
        <div style={styles.imgWrap}>
          {selectedCamera ? (
            courtWorldData ? (
              <CourtOverlayImage
                camera={selectedCamera}
                courtWorldKp1={courtWorldData.kpWorld1}
                courtWorldKp2={courtWorldData.kpWorld2}
                skeleton={courtWorldData.skeleton}
              />
            ) : (
              <img
                src={`/api/image/${selectedCamera.globalIdx}`}
                alt="camera"
                style={styles.img}
                onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
              />
            )
          ) : (
            <div style={styles.noCam}>
              カメラをクリックすると画像とワープ操作を表示します
            </div>
          )}
        </div>
        {onWarp && selectedCamera && (
          <button
            style={styles.warpBtn}
            onClick={() => onWarp(selectedCamera.c2w)}
          >
            📷 この視点へワープ
          </button>
        )}
        {selectedCamera && (
          <div style={styles.camInfo}>
            <CamRow label="種別" value="学習カメラ" />
            <CamRow label="画像" value={selectedCamera.imageName} />
            <CamRow
              label="解像度"
              value={`${selectedCamera.width} × ${selectedCamera.height}`}
            />
            <CamRow
              label="fx / fy"
              value={`${selectedCamera.fx.toFixed(1)} / ${selectedCamera.fy.toFixed(1)}`}
            />
            <CamRow
              label="位置"
              value={[
                selectedCamera.c2w[0][3],
                selectedCamera.c2w[1][3],
                selectedCamera.c2w[2][3],
              ]
                .map((v) => v.toFixed(3))
                .join(', ')}
            />
          </div>
        )}
      </div>

      {/* Sim3 summary */}
      <Sim3SummaryPanel sim3={sim3} />

      <div style={{ height: 24 }} />
    </div>
  );
};

const CamRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={styles.camRow}>
    <span style={styles.camLabel}>{label}</span>
    <span style={styles.camVal}>{value}</span>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: 'fixed',
    top: 0,
    right: 0,
    width: 340,
    height: '100vh',
    background: 'rgba(14,14,14,0.94)',
    backdropFilter: 'blur(6px)',
    borderLeft: '1px solid #2a2a2a',
    display: 'flex',
    flexDirection: 'column',
    zIndex: 10,
    overflowY: 'auto' as const,
  },
  header: {
    padding: '14px 16px',
    background: '#111',
    borderBottom: '1px solid #2a2a2a',
    position: 'sticky' as const,
    top: 0,
    zIndex: 1,
  },
  heading: {
    fontSize: 13,
    fontWeight: 600,
    color: '#ccc',
    margin: 0,
  },
  subtitle: {
    fontSize: 11,
    color: '#555',
    marginTop: 3,
  },
  section: {
    padding: '12px 14px',
    borderBottom: '1px solid #1e1e1e',
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: '#555',
    marginBottom: 10,
  },
  imgWrap: {
    minHeight: 160,
    background: '#111',
    border: '1px solid #222',
    borderRadius: 6,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  img: {
    width: '100%',
    objectFit: 'contain' as const,
    background: '#111',
  },
  noCam: {
    color: '#555',
    fontSize: 11,
    textAlign: 'center' as const,
    padding: 16,
    lineHeight: 1.5,
  },
  warpBtn: {
    width: '100%',
    marginTop: 10,
    padding: '8px 0',
    background: '#214f7c',
    color: '#fff',
    border: 'none',
    borderRadius: 5,
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
  },
  camInfo: {
    marginTop: 10,
    fontSize: 11,
    color: '#aaa',
  },
  camRow: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 8,
    margin: '6px 0',
  },
  camLabel: { color: '#666' },
  camVal: {
    color: '#ddd',
    textAlign: 'right' as const,
    wordBreak: 'break-word' as const,
  },
};
