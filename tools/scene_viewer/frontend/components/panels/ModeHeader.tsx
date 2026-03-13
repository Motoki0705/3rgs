/** ModeHeader — top-left mode name and short description. */

import React from 'react';

type ModeName = 'scene' | 'court-init' | 'court-result';

interface ModeHeaderProps {
  mode: ModeName;
}

const LABEL: Record<ModeName, { title: string; subtitle: string }> = {
  scene: {
    title: 'Scene Viewer',
    subtitle: 'SfM 点群 & カメラ可視化 (WASD + 右ドラッグ)',
  },
  'court-init': {
    title: 'Court Init Editor',
    subtitle: 'Sim(3) を調整してコートを合わせてください',
  },
  'court-result': {
    title: 'Court Result Viewer',
    subtitle: 'ground heatmap fit の transform 結果表示',
  },
};

export const ModeHeader: React.FC<ModeHeaderProps> = ({ mode }) => {
  const info = LABEL[mode];
  return (
    <div style={styles.wrap}>
      <div style={styles.title}>{info.title}</div>
      <div style={styles.sub}>{info.subtitle}</div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: 'fixed',
    top: 10,
    left: 14,
    zIndex: 20,
    pointerEvents: 'none',
  },
  title: {
    fontSize: 14,
    fontWeight: 700,
    color: '#ddd',
    textShadow: '0 1px 4px rgba(0,0,0,0.7)',
  },
  sub: {
    fontSize: 11,
    color: '#888',
    marginTop: 2,
    textShadow: '0 1px 4px rgba(0,0,0,0.7)',
  },
};
