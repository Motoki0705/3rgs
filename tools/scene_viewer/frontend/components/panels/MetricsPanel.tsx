/** MetricsPanel — court-result transform metrics display. */

import React from 'react';
import type { MetricsModel, Sim3Model } from '@/data/models';

interface MetricsPanelProps {
  metrics: MetricsModel;
  sim3: Sim3Model;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics, sim3 }) => (
  <div style={styles.section}>
    <div style={styles.sectionTitle}>Transform 結果</div>
    <KV k="カメラ数" v={String(metrics.numCameras)} />
    <KV k="Sim3 scale" v={sim3.scale.toExponential(4)} />
    <KV k="隣接gap" v={`${metrics.adjacentGap.toFixed(4)} m`} />
    <KV k="隣接方向" v={metrics.adjacentDirection} />
    <KV k="fit source" v={metrics.selectedFitSource ?? 'unknown'} />
    <KV k="total loss" v={(metrics.totalLoss ?? 0).toFixed(4)} />
    <div style={styles.sim3Text}>
      {`t: [${sim3.translation.map((v) => v.toFixed(4)).join(', ')}]\n` +
        `scale: ${sim3.scale.toExponential(4)}\n` +
        `adj_dir: ${sim3.adjacentDirection}`}
    </div>
  </div>
);

const KV: React.FC<{ k: string; v: string }> = ({ k, v }) => (
  <div style={styles.kv}>
    <span style={styles.k}>{k}</span>
    <span style={styles.v}>{v}</span>
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
  kv: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  k: { fontSize: 11, color: '#777' },
  v: { fontSize: 11, fontFamily: 'monospace', color: '#ccc', textAlign: 'right' as const },
  sim3Text: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#666',
    whiteSpace: 'pre' as const,
    lineHeight: '1.55',
    paddingTop: 6,
  },
};
