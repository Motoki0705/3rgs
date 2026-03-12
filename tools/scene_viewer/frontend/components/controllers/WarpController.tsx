/** WarpController — R3F component for the warp animation hook. */

import React, { useImperativeHandle, forwardRef } from 'react';
import { useThree } from '@react-three/fiber';
import { useWarpAnimation } from '@/hooks/useWarpAnimation';

export interface WarpControllerRef {
  warpTo: (c2w: number[][], afterDone?: () => void) => void;
  isWarping: () => boolean;
}

interface WarpControllerProps {
  /** Called after warp completes (e.g. to sync Euler). */
  onWarpDone?: () => void;
}

export const WarpController = forwardRef<WarpControllerRef, WarpControllerProps>(
  ({ onWarpDone }, ref) => {
    const { camera } = useThree();
    const { warpTo, isWarping } = useWarpAnimation(onWarpDone);

    useImperativeHandle(ref, () => ({
      warpTo: (c2w: number[][], afterDone?: () => void) => warpTo(c2w, camera, afterDone),
      isWarping,
    }), [camera, warpTo, isWarping]);

    return null;
  },
);

WarpController.displayName = 'WarpController';
