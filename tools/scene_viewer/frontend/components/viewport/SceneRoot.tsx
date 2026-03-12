/** SceneRoot — scene graph parent grouping all layers. */

import React from 'react';

interface SceneRootProps {
  children: React.ReactNode;
}

export const SceneRoot: React.FC<SceneRootProps> = ({ children }) => {
  return <group>{children}</group>;
};
