/** SelectionController — raycast selection via R3F event system. 
 *  This is a logical‐only component; the actual click targets are in CameraLayer.
 */

import React from 'react';

/**
 * SelectionController is intentionally thin — R3F's built-in
 * onClick on meshes handles raycasting automatically.
 * This component exists as a placeholder to document the pattern
 * and provide any future selection-related side effects.
 */
export const SelectionController: React.FC = () => {
  return null;
};
