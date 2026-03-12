/** LoadingOverlay — full-screen loading spinner. */

import React from 'react';

interface LoadingOverlayProps {
  message?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  message = 'データを読込中…',
}) => (
  <div style={styles.container}>
    <div style={styles.spinner} />
    <div style={styles.msg}>{message}</div>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'fixed',
    inset: 0,
    background: '#0d0d0d',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
    zIndex: 200,
  },
  spinner: {
    width: 38,
    height: 38,
    border: '3px solid #333',
    borderTopColor: '#4a90e2',
    borderRadius: '50%',
    animation: 'viewer-spin .75s linear infinite',
  },
  msg: {
    fontSize: 13,
    color: '#aaa',
  },
};

// Inject spin keyframe once
if (typeof document !== 'undefined' && !document.getElementById('viewer-spin-style')) {
  const style = document.createElement('style');
  style.id = 'viewer-spin-style';
  style.textContent = `@keyframes viewer-spin { to { transform: rotate(360deg); } }`;
  document.head.appendChild(style);
}
