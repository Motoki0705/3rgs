/** CanvasViewport — wraps R3F Canvas and provides camera + renderer setup. */

import React from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

interface CanvasViewportProps {
  /** Reduce width for a right-side panel. */
  panelWidth?: number;
  /** Use Z-up instead of Y-up. */
  zUp?: boolean;
  children: React.ReactNode;
}

export const CanvasViewport: React.FC<CanvasViewportProps> = ({
  panelWidth = 0,
  zUp = false,
  children,
}) => {
  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: panelWidth,
        bottom: 0,
      }}
    >
      <Canvas
        gl={{
          antialias: true,
        }}
        dpr={window.devicePixelRatio}
        camera={{
          fov: zUp ? 60 : 65,
          near: 0.001,
          far: 2000,
          position: zUp ? [0, -15, 8] : [0, 2, 5],
          up: zUp ? [0, 0, 1] : [0, 1, 0],
        }}
        onCreated={({ gl, camera }) => {
          gl.setClearColor(new THREE.Color(0x0f0f0f));
          if (!zUp) {
            camera.rotation.order = 'YXZ';
          }
        }}
        style={{ cursor: 'crosshair' }}
      >
        {children}
      </Canvas>
    </div>
  );
};
