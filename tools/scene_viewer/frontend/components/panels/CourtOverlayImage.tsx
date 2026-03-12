/**
 * CourtOverlayImage — camera image with projected court lines drawn on a
 * Canvas overlay.  Used in court-init and court-result panels to visually
 * verify that the Sim(3) alignment / optimised cameras are correct.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { CameraModel } from '@/data/models';
import { projectCourtToImage } from '@/utils/projectCourt';
import { CSS_COURT1, CSS_COURT2 } from '@/config/viewerTheme';

export interface CourtOverlayProps {
  camera: CameraModel;
  /** World-coordinate keypoints for court 1 (typically 20 points). */
  courtWorldKp1: number[][];
  /** World-coordinate keypoints for court 2 (adjacent court). */
  courtWorldKp2: number[][];
  /** Skeleton connectivity. */
  skeleton: [number, number][];
  /** Use c2wRefined instead of c2w for projection (court-result). */
  useRefined?: boolean;
}

export const CourtOverlayImage: React.FC<CourtOverlayProps> = ({
  camera,
  courtWorldKp1,
  courtWorldKp2,
  skeleton,
  useRefined = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const [imgLoaded, setImgLoaded] = useState(false);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgLoaded) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution to match the image's actual pixel dimensions
    const imgW = camera.width;
    const imgH = camera.height;
    canvas.width = imgW;
    canvas.height = imgH;

    ctx.clearRect(0, 0, imgW, imgH);

    // Draw court 1
    const proj1 = projectCourtToImage(courtWorldKp1, skeleton, camera, useRefined);
    ctx.strokeStyle = CSS_COURT1;
    ctx.lineWidth = Math.max(2, Math.round(imgW / 250));
    ctx.globalAlpha = 0.85;
    for (const line of proj1.lines) {
      ctx.beginPath();
      ctx.moveTo(line.from.u, line.from.v);
      ctx.lineTo(line.to.u, line.to.v);
      ctx.stroke();
    }

    // Draw court 2
    const proj2 = projectCourtToImage(courtWorldKp2, skeleton, camera, useRefined);
    ctx.strokeStyle = CSS_COURT2;
    for (const line of proj2.lines) {
      ctx.beginPath();
      ctx.moveTo(line.from.u, line.from.v);
      ctx.lineTo(line.to.u, line.to.v);
      ctx.stroke();
    }

    // Draw keypoint dots for court 1
    ctx.globalAlpha = 0.7;
    const dotR = Math.max(3, Math.round(imgW / 180));
    ctx.fillStyle = CSS_COURT1;
    for (const kp of proj1.keypoints) {
      if (!kp.valid) continue;
      if (kp.u < 0 || kp.u > imgW || kp.v < 0 || kp.v > imgH) continue;
      ctx.beginPath();
      ctx.arc(kp.u, kp.v, dotR, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = CSS_COURT2;
    for (const kp of proj2.keypoints) {
      if (!kp.valid) continue;
      if (kp.u < 0 || kp.u > imgW || kp.v < 0 || kp.v > imgH) continue;
      ctx.beginPath();
      ctx.arc(kp.u, kp.v, dotR, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1.0;
  }, [camera, courtWorldKp1, courtWorldKp2, skeleton, useRefined, imgLoaded]);

  // Re-draw whenever inputs change
  useEffect(() => {
    if (overlayEnabled) draw();
  }, [draw, overlayEnabled]);

  const handleImgLoad = useCallback(() => {
    setImgLoaded(true);
  }, []);

  return (
    <div ref={containerRef} style={styles.container}>
      <img
        ref={imgRef}
        src={`/api/image/${camera.globalIdx}`}
        alt="camera"
        style={styles.img}
        onLoad={handleImgLoad}
        onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
      />
      <canvas
        ref={canvasRef}
        style={{
          ...styles.canvas,
          display: overlayEnabled ? 'block' : 'none',
        }}
      />
      <button
        style={{
          ...styles.toggleBtn,
          ...(overlayEnabled ? styles.toggleBtnOn : styles.toggleBtnOff),
        }}
        onClick={() => setOverlayEnabled((v) => !v)}
        title={overlayEnabled ? 'コートライン非表示' : 'コートライン表示'}
      >
        {overlayEnabled ? '🏟️ ON' : '🏟️ OFF'}
      </button>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'relative',
    background: '#111',
    overflow: 'hidden',
  },
  img: {
    width: '100%',
    display: 'block',
    objectFit: 'contain' as const,
    background: '#111',
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
  },
  toggleBtn: {
    position: 'absolute',
    top: 4,
    right: 4,
    padding: '2px 7px',
    borderRadius: 4,
    border: '1px solid #444',
    fontSize: 10,
    fontWeight: 600,
    cursor: 'pointer',
    zIndex: 1,
    lineHeight: '18px',
  },
  toggleBtnOn: {
    background: 'rgba(68,204,68,0.25)',
    color: '#6be06b',
    borderColor: '#44cc44',
  },
  toggleBtnOff: {
    background: 'rgba(50,50,50,0.7)',
    color: '#888',
    borderColor: '#555',
  },
};
