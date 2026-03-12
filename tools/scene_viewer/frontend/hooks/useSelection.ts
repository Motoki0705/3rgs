/** useSelection — raycasted camera selection state. */

import { useState, useCallback } from 'react';

export function useSelection() {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const select = useCallback((idx: number) => {
    setSelectedIdx(idx);
  }, []);

  const deselect = useCallback(() => {
    setSelectedIdx(null);
  }, []);

  return { selectedIdx, select, deselect };
}
