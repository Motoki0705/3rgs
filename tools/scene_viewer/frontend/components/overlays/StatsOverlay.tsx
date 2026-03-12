/** StatsOverlay — point count, camera count, FPS. */

import React, { useState, useEffect, useRef } from 'react';

interface StatsOverlayProps {
  pointCount: number;
  cameraCount: number;
  trainCount?: number;
  testCount?: number;
}

export const StatsOverlay: React.FC<StatsOverlayProps> = ({
  pointCount,
  cameraCount,
  trainCount,
  testCount,
}) => {
  const [fps, setFps] = useState(0);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());

  useEffect(() => {
    let raf: number;
    function tick() {
      frameCount.current++;
      const now = performance.now();
      if (now - lastTime.current >= 1000) {
        setFps(frameCount.current);
        frameCount.current = 0;
        lastTime.current = now;
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const camLabel =
    testCount !== undefined && testCount > 0
      ? `カメラ: ${trainCount ?? cameraCount}学習 + ${testCount}テスト`
      : `カメラ: ${cameraCount}`;

  return (
    <div style={styles.container}>
      <span>点群: {pointCount.toLocaleString()}点</span>
      <span>{camLabel}</span>
      <span>FPS: {fps}</span>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'fixed',
    top: 12,
    left: 12,
    background: 'rgba(14,14,14,0.8)',
    border: '1px solid #2a2a2a',
    borderRadius: 6,
    padding: '5px 11px',
    fontSize: 10,
    color: '#888',
    zIndex: 10,
    display: 'flex',
    gap: 12,
  },
};
