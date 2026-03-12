/** useVisibilityState — toggle visibility of scene layers. */

import { useState, useCallback } from 'react';

export interface VisibilityFlags {
  pointCloud: boolean;
  origCameras: boolean;
  refinedCameras: boolean;
  deltaLines: boolean;
  court: boolean;
}

const DEFAULT_FLAGS: VisibilityFlags = {
  pointCloud: true,
  origCameras: true,
  refinedCameras: true,
  deltaLines: true,
  court: true,
};

export function useVisibilityState(initial?: Partial<VisibilityFlags>) {
  const [flags, setFlags] = useState<VisibilityFlags>({ ...DEFAULT_FLAGS, ...initial });

  const toggle = useCallback((key: keyof VisibilityFlags) => {
    setFlags((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  return { flags, toggle };
}
