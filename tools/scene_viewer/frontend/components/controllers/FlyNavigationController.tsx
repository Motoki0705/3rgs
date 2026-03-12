/** FlyNavigationController — R3F component wrapper for useFlyNavigation. */

import React, { useEffect } from 'react';
import { useFlyNavigation } from '@/hooks/useFlyNavigation';

interface FlyNavigationControllerProps {
  zUp?: boolean;
  enabled?: boolean;
  sceneRadius?: number;
  /** Expose syncFromCamera to parent. */
  onReady?: (ctrl: { syncFromCamera: () => void }) => void;
}

export const FlyNavigationController: React.FC<FlyNavigationControllerProps> = ({
  zUp = false,
  enabled = true,
  sceneRadius,
  onReady,
}) => {
  const { syncFromCamera, setSceneRadius } = useFlyNavigation({ zUp, enabled });

  useEffect(() => {
    if (sceneRadius !== undefined) setSceneRadius(sceneRadius);
  }, [sceneRadius, setSceneRadius]);

  useEffect(() => {
    onReady?.({ syncFromCamera });
  }, [onReady, syncFromCamera]);

  return null;
};
