/** API data-fetching hooks using @tanstack/react-query. */

import { useQuery } from '@tanstack/react-query';
import { sceneAdapter, courtInitAdapter, courtResultAdapter } from './adapters';
import type { SceneViewModel, CourtInitViewModel, CourtResultViewModel, Sim3Model } from './models';

// ── Scene mode ──────────────────────────────────────────────────────────────

async function fetchScene(): Promise<SceneViewModel> {
  const r = await fetch('/api/scene');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  return sceneAdapter(data);
}

export function useSceneQuery() {
  return useQuery<SceneViewModel>({
    queryKey: ['scene'],
    queryFn: fetchScene,
  });
}

// ── Court-init mode ─────────────────────────────────────────────────────────

async function fetchCourtInit(): Promise<CourtInitViewModel> {
  const r = await fetch('/api/court_scene');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  return courtInitAdapter(data);
}

export function useCourtInitQuery() {
  return useQuery<CourtInitViewModel>({
    queryKey: ['court-init'],
    queryFn: fetchCourtInit,
  });
}

// ── Court-result mode ───────────────────────────────────────────────────────

async function fetchCourtResult(): Promise<CourtResultViewModel> {
  const r = await fetch('/api/court_result_scene');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  return courtResultAdapter(data);
}

export function useCourtResultQuery() {
  return useQuery<CourtResultViewModel>({
    queryKey: ['court-result'],
    queryFn: fetchCourtResult,
  });
}

// ── Sim3 save / load ────────────────────────────────────────────────────────

export async function saveSim3(sim3: Sim3Model): Promise<{ ok: boolean; path?: string; error?: string }> {
  const body = {
    scale: sim3.scale,
    rotation: sim3.rotation,
    translation: sim3.translation,
    adjacent_gap: sim3.adjacentGap,
    adjacent_direction: sim3.adjacentDirection,
  };
  const r = await fetch('/api/save_sim3', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

export async function loadSim3(): Promise<{ ok: boolean; sim3?: any; source?: string }> {
  const r = await fetch('/api/load_sim3');
  return r.json();
}
