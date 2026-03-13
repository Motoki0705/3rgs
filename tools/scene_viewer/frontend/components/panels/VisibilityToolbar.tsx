/** VisibilityToolbar — toggle buttons for scene layers (court-result mode). */

import React from 'react';
import type { VisibilityFlags } from '@/hooks/useVisibilityState';

interface VisibilityToolbarProps {
  flags: VisibilityFlags;
  onToggle: (key: keyof VisibilityFlags) => void;
}

const ITEMS: { key: keyof VisibilityFlags; label: string; emoji: string }[] = [
  { key: 'pointCloud', label: '点群', emoji: '☁️' },
  { key: 'origCameras', label: '最適化カメラ', emoji: '🔵' },
  { key: 'court', label: 'コート', emoji: '🏟️' },
];

export const VisibilityToolbar: React.FC<VisibilityToolbarProps> = ({
  flags,
  onToggle,
}) => {
  return (
    <div style={styles.bar}>
      {ITEMS.map((item) => (
        <button
          key={item.key}
          title={item.label}
          style={{
            ...styles.btn,
            ...(flags[item.key] ? styles.on : styles.off),
          }}
          onClick={() => onToggle(item.key)}
        >
          <span style={styles.emoji}>{item.emoji}</span>
          <span style={styles.label}>{item.label}</span>
        </button>
      ))}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  bar: {
    display: 'flex',
    gap: 4,
    padding: '6px 8px',
    background: 'rgba(10,10,10,0.9)',
    borderBottom: '1px solid #2a2a2a',
  },
  btn: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    padding: '4px 8px',
    border: '1px solid #333',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 11,
    fontWeight: 500,
    transition: 'all 0.15s',
  },
  on: {
    background: '#1a3a5c',
    borderColor: '#3399ff55',
    color: '#ccc',
  },
  off: {
    background: '#111',
    borderColor: '#222',
    color: '#555',
  },
  emoji: { fontSize: 13 },
  label: { whiteSpace: 'nowrap' as const },
};
