/** CameraInspectorPanel — selected camera image, pose info, warp button. */

import React, { useState, useCallback } from 'react';
import type { CameraModel } from '@/data/models';
import { CourtOverlayImage, type CourtOverlayProps } from './CourtOverlayImage';

export interface CourtOverlayData {
  courtWorldKp1: number[][];
  courtWorldKp2: number[][];
  skeleton: [number, number][];
  useRefined?: boolean;
}

interface CameraInspectorPanelProps {
  camera: CameraModel | null;
  onClose: () => void;
  onWarp?: (c2w: number[][]) => void;
  /** When provided, court lines are projected onto the camera image. */
  courtOverlay?: CourtOverlayData;
}

export const CameraInspectorPanel: React.FC<CameraInspectorPanelProps> = ({
  camera: cam,
  onClose,
  onWarp,
  courtOverlay,
}) => {
  const [warpStatus, setWarpStatus] = useState<'idle' | 'warping' | 'done'>('idle');

  const handleWarp = useCallback(() => {
    if (!cam || !onWarp) return;
    const c2w = cam.c2wRefined ?? cam.c2w;
    setWarpStatus('warping');
    onWarp(c2w);
    // Will be reset by external caller or timeout
    setTimeout(() => setWarpStatus('idle'), 2500);
  }, [cam, onWarp]);

  if (!cam) return null;

  const currentPose = cam.c2wRefined ?? cam.c2w;
  const pos = [currentPose[0][3], currentPose[1][3], currentPose[2][3]];
  const origPos = [cam.c2w[0][3], cam.c2w[1][3], cam.c2w[2][3]];

  return (
    <div style={styles.panel}>
      {/* Header */}
      <div style={styles.head}>
        <span style={styles.title}>{cam.imageName}</span>
        <button style={styles.closeBtn} onClick={onClose}>✕</button>
      </div>

      {/* Image */}
      <div style={styles.imgWrap}>
        {cam.isTrain ? (
          courtOverlay ? (
            <CourtOverlayImage
              camera={cam}
              courtWorldKp1={courtOverlay.courtWorldKp1}
              courtWorldKp2={courtOverlay.courtWorldKp2}
              skeleton={courtOverlay.skeleton}
              useRefined={courtOverlay.useRefined}
            />
          ) : (
            <img
              src={`/api/image/${cam.globalIdx}`}
              alt="camera"
              style={styles.img}
              onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
            />
          )
        ) : (
          <span style={styles.noImg}>テストカメラ（学習画像なし）</span>
        )}
      </div>

      {/* Warp button */}
      {onWarp && (
        <button
          style={{
            ...styles.warpBtn,
            ...(warpStatus === 'warping' ? styles.warpBtnActive : {}),
          }}
          onClick={handleWarp}
        >
          {warpStatus === 'warping' ? '✈ ワープ中…' : warpStatus === 'done' ? '✓ 到着' : '📷 この視点へワープ'}
        </button>
      )}

      {/* Info */}
      <div style={styles.info}>
        <Row label="種別" value={cam.isTrain ? '学習カメラ' : 'テストカメラ'} />
        <Row label="画像" value={cam.imageName} />
        <Row label="解像度" value={`${cam.width} × ${cam.height}`} />
        <Row label="fx / fy" value={`${cam.fx.toFixed(1)} / ${cam.fy.toFixed(1)}`} />
        <Row label="表示位置" value={pos.map((v) => v.toFixed(3)).join(', ')} />
        {cam.c2wRefined && (
          <>
            <div style={styles.secHd}>Pose Optimization</div>
            <Row
              label="元位置"
              value={origPos.map((v) => v.toFixed(3)).join(', ')}
            />
            <Row
              label="最適化後位置"
              value={[cam.c2wRefined[0][3], cam.c2wRefined[1][3], cam.c2wRefined[2][3]]
                .map((v) => v.toFixed(3))
                .join(', ')}
            />
            <Row
              label="移動量 |Δt|"
              value={(cam.delta ?? Math.sqrt(
                (cam.c2wRefined[0][3] - pos[0]) ** 2 +
                (cam.c2wRefined[1][3] - pos[1]) ** 2 +
                (cam.c2wRefined[2][3] - pos[2]) ** 2,
              )).toFixed(5)}
            />
          </>
        )}
      </div>
    </div>
  );
};

const Row: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={styles.row}>
    <span style={styles.lbl}>{label}</span>
    <span style={styles.val}>{value}</span>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: 'fixed',
    top: 0,
    right: 0,
    width: 320,
    height: '100vh',
    background: 'rgba(14,14,14,0.94)',
    backdropFilter: 'blur(6px)',
    borderLeft: '1px solid #2a2a2a',
    display: 'flex',
    flexDirection: 'column',
    zIndex: 10,
  },
  head: {
    padding: '12px 14px',
    background: '#1a1a1a',
    borderBottom: '1px solid #2a2a2a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  title: {
    fontSize: 12,
    fontWeight: 600,
    color: '#bbb',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    maxWidth: 240,
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: '#666',
    fontSize: 18,
    cursor: 'pointer',
    lineHeight: '1',
  },
  imgWrap: {
    background: '#111',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 160,
    borderBottom: '1px solid #2a2a2a',
    flexShrink: 0,
  },
  img: {
    maxWidth: '100%',
    maxHeight: 220,
    objectFit: 'contain' as const,
  },
  noImg: {
    fontSize: 11,
    color: '#444',
  },
  warpBtn: {
    margin: '10px 14px',
    padding: '8px 0',
    borderRadius: 6,
    background: '#1e3a5f',
    color: '#7ab8f5',
    border: '1px solid #2d5588',
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: '0.04em',
    textAlign: 'center' as const,
  },
  warpBtnActive: {
    background: '#1a2e1a',
    color: '#6ab56a',
    borderColor: '#2d5a2d',
    pointerEvents: 'none' as const,
  },
  info: {
    padding: '0 14px 14px',
    flex: 1,
    overflowY: 'auto' as const,
    fontSize: 11,
  },
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '3px 0',
    borderBottom: '1px solid #1e1e1e',
  },
  lbl: { color: '#666' },
  val: { color: '#ccc', fontFamily: 'monospace', fontSize: 10 },
  secHd: {
    margin: '10px 0 6px',
    fontSize: 10,
    color: '#555',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
  },
};
