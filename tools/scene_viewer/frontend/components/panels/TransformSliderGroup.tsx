/** TransformSliderGroup — tx/ty/tz/yaw/pitch/roll/logScale slider rows. */

import React from 'react';
import type { Sim3DraftState } from '@/hooks/useSim3Draft';

interface TransformSliderGroupProps {
  draft: Sim3DraftState;
  onChange: (partial: Partial<Sim3DraftState>) => void;
}

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  accentColor?: string;
  onChange: (v: number) => void;
}

const SliderRow: React.FC<SliderRowProps> = ({
  label,
  value,
  min,
  max,
  step,
  display,
  accentColor = '#4a90e2',
  onChange,
}) => (
  <div style={styles.row}>
    <label style={styles.label}>{label}</label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      style={{ ...styles.slider, accentColor }}
    />
    <span style={styles.val}>{display}</span>
  </div>
);

export const TransformSliderGroup: React.FC<TransformSliderGroupProps> = ({
  draft,
  onChange,
}) => {
  const scale = Math.pow(10, draft.logScale);

  // Dynamic slider ranges: widen if value is outside default
  const tRange = (v: number) => ({
    min: Math.min(-20, v - Math.abs(v) * 0.5),
    max: Math.max(20, v + Math.abs(v) * 0.5),
  });

  return (
    <>
      {/* Translation */}
      <Section title="平行移動 (Translation)">
        <SliderRow
          label="TX"
          value={draft.tx}
          {...tRange(draft.tx)}
          step={0.01}
          display={draft.tx.toFixed(3)}
          onChange={(v) => onChange({ tx: v })}
        />
        <SliderRow
          label="TY"
          value={draft.ty}
          {...tRange(draft.ty)}
          step={0.01}
          display={draft.ty.toFixed(3)}
          onChange={(v) => onChange({ ty: v })}
        />
        <SliderRow
          label="TZ"
          value={draft.tz}
          {...tRange(draft.tz)}
          step={0.01}
          display={draft.tz.toFixed(3)}
          onChange={(v) => onChange({ tz: v })}
        />
      </Section>

      {/* Rotation */}
      <Section title="回転 — オイラー角 Z-Y-X 内因性 (度)">
        <SliderRow
          label="Yaw Z"
          value={draft.yawDeg}
          min={-180}
          max={180}
          step={0.1}
          display={`${draft.yawDeg.toFixed(1)}°`}
          onChange={(v) => onChange({ yawDeg: v })}
        />
        <SliderRow
          label="Pitch Y"
          value={draft.pitchDeg}
          min={-180}
          max={180}
          step={0.1}
          display={`${draft.pitchDeg.toFixed(1)}°`}
          onChange={(v) => onChange({ pitchDeg: v })}
        />
        <SliderRow
          label="Roll X"
          value={draft.rollDeg}
          min={-180}
          max={180}
          step={0.1}
          display={`${draft.rollDeg.toFixed(1)}°`}
          onChange={(v) => onChange({ rollDeg: v })}
        />
      </Section>

      {/* Scale */}
      <Section title="スケール (Scale)">
        <SliderRow
          label="Scale"
          value={draft.logScale}
          min={Math.min(-2.0, draft.logScale - 0.5)}
          max={Math.max(1.5, draft.logScale + 0.5)}
          step={0.001}
          display={`${scale.toFixed(4)} ×`}
          accentColor="#e97c2d"
          onChange={(v) => onChange({ logScale: v })}
        />
        <div style={styles.hint}>スライダー値 = log₁₀(scale) | 10^v = 実スケール</div>
      </Section>
    </>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div style={styles.section}>
    <div style={styles.sectionTitle}>{title}</div>
    {children}
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
    cursor: 'pointer',
  },
  val: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: '#ddd',
    textAlign: 'right' as const,
  },
  hint: {
    fontSize: 10,
    color: '#444',
    marginTop: 2,
  },
};
