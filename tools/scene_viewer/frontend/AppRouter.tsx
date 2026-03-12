/** AppRouter — selects the correct page component based on URL path. */

import React from 'react';
import { SceneModePage } from '@/pages/SceneModePage';
import { CourtInitModePage } from '@/pages/CourtInitModePage';
import { CourtResultModePage } from '@/pages/CourtResultModePage';

function detectMode(): 'scene' | 'court-init' | 'court-result' {
  const path = window.location.pathname.toLowerCase();
  if (path.includes('court_init') || path.includes('court-init')) return 'court-init';
  if (path.includes('court_result') || path.includes('court-result')) return 'court-result';
  return 'scene';
}

export const AppRouter: React.FC = () => {
  const mode = detectMode();

  switch (mode) {
    case 'court-init':
      return <CourtInitModePage />;
    case 'court-result':
      return <CourtResultModePage />;
    default:
      return <SceneModePage />;
  }
};
