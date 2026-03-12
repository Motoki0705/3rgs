/** NavigationHelpOverlay — common control instructions. */

import React from 'react';

export const NavigationHelpOverlay: React.FC = () => (
  <div style={styles.container}>
    右ドラッグ: 視点回転<br />
    WASD: 水平移動<br />
    Space: 上移動 / Shift: 下移動<br />
    スクロール: 前後移動<br />
    左クリック: カメラ選択
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'fixed',
    bottom: 10,
    right: 10,
    fontSize: 10,
    color: '#444',
    lineHeight: 1.6,
    pointerEvents: 'none',
    zIndex: 10,
    textAlign: 'right' as const,
  },
};
