/** SaveActions — reset / save / load status for court-init. */

import React, { useState, useCallback } from 'react';
import type { Sim3Model } from '@/data/models';
import { saveSim3 } from '@/data/useSceneQuery';

interface SaveActionsProps {
  sim3: Sim3Model;
  onReset: () => void;
}

export const SaveActions: React.FC<SaveActionsProps> = ({ sim3, onReset }) => {
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null);

  const handleSave = useCallback(async () => {
    try {
      const result = await saveSim3(sim3);
      if (result.ok) {
        const short = result.path?.split('/').slice(-3).join('/') ?? '';
        setStatus({ msg: `保存しました → ${short}`, ok: true });
      } else {
        setStatus({ msg: `エラー: ${result.error}`, ok: false });
      }
    } catch (e: any) {
      setStatus({ msg: `通信エラー: ${e.message}`, ok: false });
    }
    setTimeout(() => setStatus(null), 3500);
  }, [sim3]);

  return (
    <>
      <div style={styles.btnRow}>
        <button style={styles.resetBtn} onClick={onReset}>
          ↺ 自動初期値に戻す
        </button>
        <button style={styles.saveBtn} onClick={handleSave}>
          💾 init_sim3.json を保存
        </button>
      </div>
      {status && (
        <div
          style={{
            ...styles.status,
            ...(status.ok ? styles.statusOk : styles.statusErr),
          }}
        >
          {status.msg}
        </div>
      )}
    </>
  );
};

const styles: Record<string, React.CSSProperties> = {
  btnRow: {
    display: 'flex',
    gap: 8,
    padding: '12px 14px',
    borderBottom: '1px solid #1e1e1e',
  },
  resetBtn: {
    flex: 1,
    padding: '8px 0',
    border: 'none',
    borderRadius: 5,
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    background: '#2a2a2a',
    color: '#aaa',
  },
  saveBtn: {
    flex: 1,
    padding: '8px 0',
    border: 'none',
    borderRadius: 5,
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    background: '#4a90e2',
    color: '#fff',
  },
  status: {
    margin: '0 14px 10px',
    fontSize: 11,
    textAlign: 'center' as const,
    padding: '5px 8px',
    borderRadius: 4,
  },
  statusOk: {
    background: 'rgba(46,160,67,.25)',
    color: '#56d364',
  },
  statusErr: {
    background: 'rgba(220,53,53,.25)',
    color: '#f85149',
  },
};
