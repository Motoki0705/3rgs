/** AdjacentCourtEditor — direction dropdown + gap slider. */

import React from 'react';
import type { Sim3DraftState } from '@/hooks/useSim3Draft';

interface AdjacentCourtEditorProps {
  draft: Sim3DraftState;
  onChange: (partial: Partial<Sim3DraftState>) => void;
}

export const AdjacentCourtEditor: React.FC<AdjacentCourtEditorProps> = ({
  draft,
  onChange,
}) => (
  <div style={styles.section}>
    <div style={styles.sectionTitle}>隣接コート (Adjacent Court)</div>
    <div style={{ marginBottom: 8 }}>
      <div style={styles.dropdownLabel}>adjacent_direction</div>
      <select
        value={draft.adjacentDirection}
        onChange={(e) => onChange({ adjacentDirection: e.target.value })}
        style={styles.select}
      >
        <option value="+x">+X</option>
        <option value="-x">-X</option>
        <option value="+y">+Y</option>
        <option value="-y">-Y</option>
      </select>
    </div>
    <div style={styles.row}>
      <label style={styles.label}>Gap</label>
      <input
        type="range"
        min={0}
        max={30}
        step={0.05}
        value={draft.adjacentGap}
        onChange={(e) => onChange({ adjacentGap: parseFloat(e.target.value) })}
        style={styles.slider}
      />
      <span style={styles.val}>{draft.adjacentGap.toFixed(2)} m</span>
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
    marginBottom: 10,
  },
  dropdownLabel: {
    fontSize: 11,
    color: '#888',
    marginBottom: 4,
  },
  select: {
    background: '#1a1a1a',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: 4,
    padding: '4px 8px',
    fontSize: 12,
    width: '100%',
    cursor: 'pointer',
    outline: 'none',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '54px 1fr 58px',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  label: { fontSize: 11, color: '#888' },
  slider: {
    width: '100%',
    accentColor: '#4a90e2',
    cursor: 'pointer',
  },
  val: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: '#ddd',
    textAlign: 'right' as const,
  },
};
