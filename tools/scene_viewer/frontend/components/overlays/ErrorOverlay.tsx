/** ErrorOverlay — full-screen error display. */

import React from 'react';

interface ErrorOverlayProps {
  message: string;
}

export const ErrorOverlay: React.FC<ErrorOverlayProps> = ({ message }) => (
  <div style={styles.container}>
    <div style={styles.icon}>⚠</div>
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
  icon: {
    fontSize: 36,
    color: '#f85149',
  },
  msg: {
    fontSize: 14,
    color: '#f85149',
    maxWidth: 500,
    textAlign: 'center' as const,
  },
};
