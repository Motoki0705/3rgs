/** LegendOverlay — colour legend and mode indicator. */

import React from 'react';

interface LegendItem {
  color: string;
  label: string;
  shape?: 'dot' | 'line';
}

interface LegendOverlayProps {
  items: LegendItem[];
}

export const LegendOverlay: React.FC<LegendOverlayProps> = ({ items }) => (
  <div style={styles.container}>
    <h3 style={styles.title}>凡例</h3>
    {items.map((item, i) => (
      <div key={i} style={styles.row}>
        <div
          style={
            item.shape === 'line'
              ? { ...styles.line, background: item.color }
              : { ...styles.dot, background: item.color }
          }
        />
        <span style={styles.label}>{item.label}</span>
      </div>
    ))}
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'fixed',
    bottom: 14,
    left: 14,
    background: 'rgba(14,14,14,0.88)',
    backdropFilter: 'blur(4px)',
    border: '1px solid #2a2a2a',
    borderRadius: 8,
    padding: '10px 14px',
    fontSize: 11,
    zIndex: 10,
  },
  title: {
    fontSize: 10,
    color: '#555',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    marginBottom: 7,
    fontWeight: 600,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    margin: '3px 0',
    color: '#999',
  },
  dot: {
    width: 9,
    height: 9,
    borderRadius: '50%',
    flexShrink: 0,
  },
  line: {
    width: 20,
    height: 2,
    borderRadius: 1,
    flexShrink: 0,
  },
  label: {
    fontSize: 11,
    color: '#aaa',
  },
};
