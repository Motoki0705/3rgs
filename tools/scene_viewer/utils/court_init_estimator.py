#!/usr/bin/env python3
"""Initial Sim(3) estimator for tennis court alignment.

Given MASt3R reconstruction data (point cloud PLY + camera poses),
estimates an initial court-to-world Sim(3) transform by:

1. Fitting a ground plane to the dense point cloud via RANSAC
2. Determining court orientation (X/Y axes) from PCA of plane-projected points
3. Estimating scale from the bounding box vs known ITF court dimensions
4. Computing the translation from the projected point centroid

Outputs a dict in the same format as court transform init_sim3.json:
{
    "scale": float,
    "rotation": [[3x3]],
    "translation": [tx, ty, tz],
    "adjacent_gap": float,
    "adjacent_direction": str,
}

Usage as a module::

    from tools.scene_viewer.utils.court_init_estimator import CourtInitEstimator
    est = CourtInitEstimator("/path/to/data/tennis_court/mast3r")
    sim3 = est.estimate()

Usage from CLI::

    python tools/scene_viewer/utils/court_init_estimator.py --mast3r-dir data/tennis_court/mast3r
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─── Repo / src path setup ───────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.court_scheme import (  # noqa: E402
    COURT_SKELETON,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    BASELINE_CLEAR,
    SIDELINE_CLEAR,
    court_keypoints_3d,
)

# ─── Known ITF dimensions used for scale estimation ──────────────────────────
# Full playable + run-off area extent (used to estimate scene scale).
# These are rough bounds; actual visible extent may vary.
_FULL_Y_EXTENT = COURT_LENGTH + 2.0 * BASELINE_CLEAR   # ~36.6 m  (long axis)
_FULL_X_EXTENT = DOUBLES_WIDTH + 2.0 * SIDELINE_CLEAR  # ~18.3 m  (short axis)


class CourtInitEstimator:
    """Estimate an initial Sim(3) (court → MASt3R world) from a PLY point cloud.

    Args:
        mast3r_dir: Path to the ``mast3r/`` sub-directory that contains
            ``pointcloud.ply`` and ``camera_poses.npy``.
    """

    def __init__(self, mast3r_dir: str | Path) -> None:
        self.mast3r_dir = Path(mast3r_dir).resolve()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def estimate(
        self,
        n_sample: int = 50_000,
        ransac_iters: int = 2_000,
        ransac_thr: float = 0.1,
        adjacent_direction: str = "+x",
        init_gap: float = 3.0,
    ) -> dict[str, Any]:
        """Run the full estimation pipeline.

        Strategy:
        1. Load PLY sub-sample without pre-filtering (RANSAC handles outliers).
        2. RANSAC plane fitting.
        3. Project *camera positions* onto the plane for PCA + scale estimation.
        4. Near-plane PLY inliers give the robust ground centroid for translation.

        Returns a dict with keys:
            scale, rotation (list[list[float]], 3x3),
            translation (list[float], 3), adjacent_gap, adjacent_direction,
            point_cloud, cameras, image_dir, court_keypoints_court, court_skeleton.
        """
        pts_xyz, pts_rgb = self._load_point_cloud(n_sample)
        camera_poses = self._load_camera_poses()
        cam_cents = camera_poses[:, :3, 3]

        # Fit ground plane (RANSAC on the full unfiltered sub-sample)
        normal, plane_d = self._fit_plane_ransac(pts_xyz, ransac_iters, ransac_thr)

        # Orient normal toward camera centroid
        cam_centroid = cam_cents.mean(axis=0)
        signed_dist = float(np.dot(cam_centroid, normal) - plane_d)
        if signed_dist < 0:
            normal = -normal
            plane_d = -plane_d

        # Court axes and scale from camera-position PCA
        # Project camera positions onto the fitted plane
        cam_proj = cam_cents - (cam_cents @ normal - plane_d)[:, None] * normal
        e_y, e_x_cand = self._pca_axes_from_points(cam_proj, normal)

        # Compute extents along both candidate axes
        cam_proj_y = cam_proj @ e_y
        cam_proj_x = cam_proj @ e_x_cand
        ext_y = float(np.percentile(cam_proj_y, 95) - np.percentile(cam_proj_y, 5))
        ext_x = float(np.percentile(cam_proj_x, 95) - np.percentile(cam_proj_x, 5))

        # Ensure e_y is the longer axis
        if ext_y < ext_x:
            e_y, e_x_cand = e_x_cand, e_y
            ext_y, ext_x = ext_x, ext_y

        # Scale: camera layout span vs expected court+runoff extent
        scale_y = ext_y / _FULL_Y_EXTENT if _FULL_Y_EXTENT > 0 else 1.0
        scale_x = ext_x / _FULL_X_EXTENT if _FULL_X_EXTENT > 0 else 1.0
        scale = float(np.sqrt(max(scale_y, 1e-6) * max(scale_x, 1e-6)))
        scale = float(np.clip(scale, 1e-3, 100.0))

        # Build rotation R = [e_x | e_y | e_z]
        e_z = normal
        e_y_orth = e_y - np.dot(e_y, e_z) * e_z
        norm_ey = np.linalg.norm(e_y_orth)
        if norm_ey < 1e-9:
            e_y_orth = np.array([1.0, 0.0, 0.0])
            e_y_orth -= np.dot(e_y_orth, e_z) * e_z
            e_y_orth /= np.linalg.norm(e_y_orth)
        else:
            e_y_orth /= norm_ey
        e_x = np.cross(e_y_orth, e_z)
        e_x /= np.linalg.norm(e_x)
        rotation = np.column_stack([e_x, e_y_orth, e_z])  # (3, 3)

        # Translation from near-plane PLY inlier median
        plane_dists = np.abs(pts_xyz @ normal - plane_d)
        thr = float(np.clip(scale * 2.0, 0.05, 5.0))
        near_plane_mask = plane_dists < thr
        if near_plane_mask.sum() >= 10:
            inlier_pts = pts_xyz[near_plane_mask]
        else:
            inlier_pts = cam_proj  # fall back to camera projections

        cx = float(np.median(inlier_pts @ e_x))
        cy = float(np.median(inlier_pts @ e_y_orth))
        translation = cx * e_x + cy * e_y_orth + plane_d * e_z

        # Build outputs
        image_names, image_dir = self._load_image_metadata()
        cameras = self._build_camera_list(camera_poses, image_names)
        kp_court = court_keypoints_3d().cpu().numpy().tolist()

        result: dict[str, Any] = {
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "adjacent_gap": float(init_gap),
            "adjacent_direction": adjacent_direction,
            "plane_normal": normal.tolist(),
            "plane_d": float(plane_d),
            "point_cloud": {
                "xyz": pts_xyz.tolist(),
                "rgb": pts_rgb.tolist() if pts_rgb is not None else [],
            },
            "cameras": cameras,
            "image_dir": str(image_dir) if image_dir is not None else "",
            "court_keypoints_court": kp_court,
            "court_skeleton": COURT_SKELETON,
        }
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_point_cloud(
        self, n_sample: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load PLY point cloud, sub-sampling to at most *n_sample* points.

        For ASCII PLY, uses reservoir sampling to draw a uniform random subset
        without loading the full file into memory.

        Returns:
            xyz: (N, 3) float32
            rgb: (N, 3) float32 in [0, 1], or None
        """
        ply_path = self.mast3r_dir / "pointcloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {ply_path}")

        # Parse PLY header to get n_vertex and field layout
        header_lines, n_vertex, has_rgb = self._parse_ply_header(ply_path)

        # For binary PLY, fall back to plyfile
        with open(ply_path, "rb") as f:
            hdr_bytes = f.read(200)
        if b"format binary" in hdr_bytes:
            return self._load_ply_binary(ply_path, n_sample)

        actual_sample = min(n_sample, n_vertex)
        if actual_sample <= 0:
            empty_xyz = np.empty((0, 3), dtype=np.float32)
            empty_rgb = np.empty((0, 3), dtype=np.float32) if has_rgb else None
            return empty_xyz, empty_rgb

        if has_rgb:
            usecols = (0, 1, 2, 3, 4, 5)
        else:
            usecols = (0, 1, 2)
        n_cols = len(usecols)
        data = np.empty((actual_sample, n_cols), dtype=np.float32)
        rng = np.random.default_rng()

        with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(header_lines):
                next(f)

            seen = 0
            for line in f:
                parts = line.split()
                if len(parts) < n_cols:
                    continue
                row = np.asarray(parts[:n_cols], dtype=np.float32)
                if seen < actual_sample:
                    data[seen] = row
                else:
                    idx = rng.integers(seen + 1)
                    if idx < actual_sample:
                        data[idx] = row
                seen += 1

        if seen < actual_sample:
            data = data[:seen]

        xyz = data[:, :3].astype(np.float32)
        if has_rgb:
            rgb = (data[:, 3:6] / 255.0).astype(np.float32)
        else:
            rgb = None
        return xyz, rgb

    @staticmethod
    def _parse_ply_header(ply_path: Path) -> tuple[int, int, bool]:
        """Return (header_line_count, n_vertex, has_rgb)."""
        n_vertex = 0
        has_rgb = False
        header_lines = 0
        with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line.startswith("element vertex"):
                    n_vertex = int(line.split()[-1])
                if "red" in line or "green" in line:
                    has_rgb = True
                if line == "end_header":
                    header_lines = i + 1
                    break
        return header_lines, n_vertex, has_rgb

    def _load_ply_binary(
        self, ply_path: Path, n_sample: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fallback binary PLY loader using plyfile."""
        from plyfile import PlyData  # type: ignore
        ply = PlyData.read(str(ply_path))
        vertex = ply["vertex"]
        x = np.asarray(vertex["x"], dtype=np.float32)
        y = np.asarray(vertex["y"], dtype=np.float32)
        z = np.asarray(vertex["z"], dtype=np.float32)
        xyz = np.column_stack([x, y, z])
        rgb: np.ndarray | None = None
        if "red" in vertex.data.dtype.names:
            r = np.asarray(vertex["red"], dtype=np.float32) / 255.0
            g = np.asarray(vertex["green"], dtype=np.float32) / 255.0
            b = np.asarray(vertex["blue"], dtype=np.float32) / 255.0
            rgb = np.column_stack([r, g, b])
        n_total = len(xyz)
        if n_total > n_sample:
            idx = np.random.choice(n_total, n_sample, replace=False)
            xyz = xyz[idx]
            if rgb is not None:
                rgb = rgb[idx]
        return xyz, rgb

    def _load_camera_poses(self) -> np.ndarray:
        """Load (N, 4, 4) camera-to-world matrices."""
        poses_path = self.mast3r_dir / "camera_poses.npy"
        if not poses_path.exists():
            raise FileNotFoundError(f"Camera poses not found: {poses_path}")
        return np.load(str(poses_path)).astype(np.float64)

    def _load_image_metadata(self) -> tuple[list[str], Path | None]:
        """Return ordered train image names and the image directory if available."""
        scene_dir = self.mast3r_dir.parent
        names_path = scene_dir / "images_train.txt"
        images_dir = scene_dir / "images"

        if not names_path.exists():
            return [], images_dir if images_dir.exists() else None

        stems = [
            line.strip()
            for line in names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ext = ".jpg"
        if images_dir.exists():
            image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
            if image_files:
                ext = image_files[0].suffix
        image_names = [f"{stem}{ext}" for stem in stems]
        return image_names, images_dir if images_dir.exists() else None

    def _fit_plane_ransac(
        self,
        pts: np.ndarray,
        n_iter: int = 1_000,
        thr: float = 0.05,
    ) -> tuple[np.ndarray, float]:
        """RANSAC plane fitting.

        Returns:
            normal: unit normal vector (3,), float64
            d: plane offset s.t. normal · x = d  for points x on the plane
        """
        rng = np.random.default_rng(0)
        n = len(pts)
        best_inlier_count = -1
        best_normal = np.array([0.0, 0.0, 1.0])
        best_d = 0.0

        for _ in range(n_iter):
            idx = rng.choice(n, 3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            v1 = p1 - p0
            v2 = p2 - p0
            normal_cand = np.cross(v1, v2)
            norm_len = float(np.linalg.norm(normal_cand))
            if norm_len < 1e-10:
                continue
            normal_cand /= norm_len
            d_cand = float(np.dot(normal_cand, p0))
            distances = np.abs(pts @ normal_cand - d_cand)
            inlier_count = int(np.sum(distances < thr))
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_normal = normal_cand
                best_d = d_cand

        # Refit using all inliers (least-squares plane)
        dists = np.abs(pts @ best_normal - best_d)
        inliers = pts[dists < thr]
        if len(inliers) >= 3:
            # Least-squares plane fit: minimise ||A n - d||
            centroid = inliers.mean(axis=0)
            _, _, Vt = np.linalg.svd(inliers - centroid)
            best_normal = Vt[-1]  # smallest singular value = normal
            best_normal /= np.linalg.norm(best_normal)
            best_d = float(np.dot(best_normal, centroid))

        return best_normal.astype(np.float64), float(best_d)

    def _pca_axes_from_points(
        self,
        pts: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two dominant in-plane PCA axes of *pts* as 3-D unit vectors.

        Args:
            pts: (N, 3) array of points lying approximately on the plane.
            normal: plane normal (unit vector).

        Returns:
            (axis1, axis2) where axis1 has the larger variance.
        """
        centred = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        u1, u2 = Vt[0], Vt[1]
        # Project out normal component and re-normalise
        u1 = u1 - np.dot(u1, normal) * normal
        n1 = np.linalg.norm(u1)
        u1 = u1 / n1 if n1 > 1e-9 else np.array([1.0, 0.0, 0.0])
        u2 = u2 - np.dot(u2, normal) * normal
        u2 -= np.dot(u2, u1) * u1
        n2 = np.linalg.norm(u2)
        u2 = u2 / n2 if n2 > 1e-9 else np.cross(normal, u1)
        return u1.astype(np.float64), u2.astype(np.float64)

    def _build_camera_list(self, camera_poses: np.ndarray, image_names: list[str]) -> list[dict]:
        """Build a JSON-serialisable list of camera dicts for visualisation.

        Each entry has:
            c2w: 4×4 camera-to-world  (row-major list of lists)
        """
        cameras = []
        intrinsics_path = self.mast3r_dir / "camera_intrinsics.npy"
        intrinsics = None
        if intrinsics_path.exists():
            intrinsics = np.load(str(intrinsics_path))

        for i, c2w in enumerate(camera_poses):
            entry: dict[str, Any] = {
                "train_idx": i,
                "global_idx": i,
                "is_train": True,
                "image_name": image_names[i] if i < len(image_names) else f"{i:06d}.jpg",
                "c2w": c2w.tolist(),
            }
            if intrinsics is not None and i < len(intrinsics):
                K = intrinsics[i]
                entry["fx"] = float(K[0, 0])
                entry["fy"] = float(K[1, 1])
                entry["cx"] = float(K[0, 2])
                entry["cy"] = float(K[1, 2])
                entry["width"] = int(round(entry["cx"] * 2))
                entry["height"] = int(round(entry["cy"] * 2))
            cameras.append(entry)
        return cameras


# ─── Save / load helpers (for use by server.py) ──────────────────────────────

def sim3_to_json(result: dict[str, Any]) -> dict[str, Any]:
    """Extract Sim(3) parameters from estimator result into a saveable dict."""
    return {
        "scale": result["scale"],
        "rotation": result["rotation"],
        "translation": result["translation"],
        "adjacent_gap": result["adjacent_gap"],
        "adjacent_direction": result["adjacent_direction"],
        "plane_normal": result["plane_normal"],
        "plane_d": result["plane_d"],
    }


def save_sim3(sim3: dict[str, Any], output_path: Path | str) -> None:
    """Save Sim(3) dict to JSON (creating parent directories if necessary)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sim3, indent=2), encoding="utf-8")


def load_sim3(path: Path | str) -> dict[str, Any] | None:
    """Load a saved Sim(3) JSON, returning None if file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mast3r-dir",
        type=Path,
        default=Path("data/tennis_court/mast3r"),
        help="Path to mast3r/ sub-directory containing pointcloud.ply",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <mast3r-dir>/../court/transform/init_sim3.json",
    )
    parser.add_argument("--n-sample", type=int, default=50_000)
    parser.add_argument("--adjacent-direction", default="+x",
                        choices=("+x", "-x", "+y", "-y"))
    parser.add_argument("--init-gap", type=float, default=3.0)
    parser.add_argument("--ransac-iters", type=int, default=1_000)
    parser.add_argument("--ransac-thr", type=float, default=0.05)
    args = parser.parse_args()

    mast3r_dir = args.mast3r_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else mast3r_dir.parent / "court" / "transform" / "init_sim3.json"
    )

    est = CourtInitEstimator(mast3r_dir)
    print(f"Estimating initial Sim(3) from {mast3r_dir} …", flush=True)
    result = est.estimate(
        n_sample=args.n_sample,
        ransac_iters=args.ransac_iters,
        ransac_thr=args.ransac_thr,
        adjacent_direction=args.adjacent_direction,
        init_gap=args.init_gap,
    )

    sim3 = sim3_to_json(result)
    save_sim3(sim3, output)
    print(f"Saved to {output}")
    print(f"  scale={sim3['scale']:.4f}")
    print(f"  translation={[f'{v:.3f}' for v in sim3['translation']]}")
    print(f"  adjacent_gap={sim3['adjacent_gap']:.3f}m  direction={sim3['adjacent_direction']}")


if __name__ == "__main__":
    main()
