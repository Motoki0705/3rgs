/** Sim3SummaryPanel — text display of current Sim(3) params. */

import React from 'react';
import type { Sim3Model } from '@/data/models';

interface Sim3SummaryPanelProps {
  sim3: Sim3Model;
}

export const Sim3SummaryPanel: React.FC<Sim3SummaryPanelProps> = ({ sim3 }) => (
  <div style={styles.section}>
    <div style={styles.sectionTitle}>現在の Sim(3) パラメータ</div>
    <div style={styles.display}>
      {`scale: ${sim3.scale.toFixed(5)}\n` +
        `t: [${sim3.translation.map((v) => v.toFixed(3)).join(', ')}]\n` +
        `gap: ${sim3.adjacentGap.toFixed(3)} m  dir: ${sim3.adjacentDirection}`}
    </div>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
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
    marginBottom: 8,
  },
  display: {
    background: '#111',
    border: '1px solid #222',
    borderRadius: 5,
    padding: '8px 10px',
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#666',
    wordBreak: 'break-all' as const,
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap' as const,
  },
};
