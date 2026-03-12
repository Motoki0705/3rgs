/** CameraListPanel — scrollable camera list for court-result mode. */

import React from 'react';
import type { CameraModel } from '@/data/models';

interface CameraListPanelProps {
  cameras: CameraModel[];
  selectedIdx: number | null;
  meanDelta: number;
  onSelect: (idx: number) => void;
}

export const CameraListPanel: React.FC<CameraListPanelProps> = ({
  cameras,
  selectedIdx,
  meanDelta,
  onSelect,
}) => (
  <div style={styles.section}>
    <div style={styles.sectionTitle}>カメラリスト（クリックでワープ）</div>
    <div style={styles.list}>
      {cameras.map((cam, i) => {
        const stem = cam.imageName.replace(/\.[^.]+$/, '');
        const high = (cam.delta ?? 0) > meanDelta * 2;
        const selected = i === selectedIdx;
        return (
          <div
            key={i}
            style={{
              ...styles.item,
              ...(selected ? styles.itemSelected : {}),
            }}
            onClick={() => onSelect(i)}
          >
            <span style={styles.stem} title={stem}>{stem.slice(-14)}</span>
            <span style={{ ...styles.delta, ...(high ? styles.deltaHigh : {}) }}>
              {(cam.delta ?? 0).toFixed(4)}
            </span>
            <span style={{ ...styles.delta, color: '#666' }}>{i}</span>
          </div>
        );
      })}
    </div>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  section: {
    padding: '10px 14px',
    borderBottom: '1px solid #1e1e1e',
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: '#555',
    marginBottom: 8,
  },
  list: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#888',
    overflowY: 'auto' as const,
    maxHeight: 220,
  },
  item: {
    display: 'grid',
    gridTemplateColumns: '1fr 56px 56px',
    gap: 4,
    padding: '2px 0',
    cursor: 'pointer',
    borderRadius: 2,
    transition: 'background .1s',
  },
  itemSelected: {
    background: '#1d2f40',
    color: '#90c0e8',
  },
  stem: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  delta: {
    textAlign: 'right' as const,
    color: '#6a6a6a',
  },
  deltaHigh: {
    color: '#e8704a',
  },
};
