"""3D shape reconstruction visualizer for 9DTact.

Implements the O3DVisualizer-based TactileMeshVisualizer (see
docs/superpowers/specs/2026-07-28-visualizer-redesign-design.md)
and the pure-function helpers used by it.

Pure helpers (testable without Open3D):
- compute_readings(height_map, pixel_per_mm, contact_threshold)
- jet_vertex_colors(height_map, depth_max)
- jet_uint8_image(height_map, depth_max)
- require_gui()

GUI class:
- TactileMeshVisualizer
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Open3D is imported lazily so the helpers can be tested in headless envs.
try:
    import open3d  # noqa: F401
    from open3d.visualization import O3DVisualizer as _O3DVisualizer
    from open3d.visualization import gui as _gui
    O3DVisualizer = _O3DVisualizer
    gui = _gui
except Exception:  # pragma: no cover - exercised in environments without open3d
    O3DVisualizer = None
    gui = None


# ----- Pure-function helpers ---------------------------------------------------

@dataclass(frozen=True)
class Readings:
    max_depth: float
    contact_area_mm2: float
    contact_center_mm: Tuple[Optional[float], Optional[float]]


def compute_readings(
    height_map: np.ndarray,
    pixel_per_mm: float,
    contact_threshold: float = 0.05,
) -> Readings:
    """Aggregate the four sidebar readings from a height map in mm."""
    h = np.asarray(height_map, dtype=np.float32)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    max_depth = float(h.max())
    contact_mask = h > contact_threshold
    area_mm2 = float(contact_mask.sum()) * (pixel_per_mm ** 2)
    if contact_mask.any():
        ys, xs = np.where(contact_mask)
        cx = float(xs.mean() * pixel_per_mm)
        cy = float(ys.mean() * pixel_per_mm)
        center = (cx, cy)
    else:
        center = (None, None)
    return Readings(
        max_depth=max_depth,
        contact_area_mm2=area_mm2,
        contact_center_mm=center,
    )


# Jet LUT sampled at 256 stops (approximation of matplotlib's "jet",
# used to color the 3D mesh and 2D depth map consistently).
def _build_jet_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        if t < 0.125:
            r, g, b = 0.0, 0.0, 0.5 + 4.0 * t
        elif t < 0.375:
            r, g, b = 0.0, t * 4.0 - 0.5, 1.0
        elif t < 0.625:
            r, g, b = t * 4.0 - 1.5, 1.0, -t * 4.0 + 3.5
        elif t < 0.875:
            r, g, b = 1.0, -t * 4.0 + 4.5, 0.0
        else:
            r, g, b = -t * 4.0 + 4.5, 0.0, 0.0
        lut[i] = (max(0.0, min(1.0, r)),
                  max(0.0, min(1.0, g)),
                  max(0.0, min(1.0, b)))
    return lut


_JET_LUT = _build_jet_lut()


def _jet_lookup(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    idx = (t * 255.0).astype(np.int32)
    return _JET_LUT[idx]


def jet_vertex_colors(
    height_map: np.ndarray, depth_max: float
) -> np.ndarray:
    """Return float32 (N, 3) jet colors per vertex of the flattened grid."""
    h = np.asarray(height_map, dtype=np.float32).ravel()
    if depth_max <= 0:
        t = np.zeros_like(h)
    else:
        t = h / float(depth_max)
    return _jet_lookup(t)


def jet_uint8_image(height_map: np.ndarray, depth_max: float) -> np.ndarray:
    """Return a uint8 (H, W, 3) RGB image for gui.ImageWidget."""
    h = np.asarray(height_map, dtype=np.float32)
    if depth_max <= 0:
        t = np.zeros_like(h)
    else:
        t = h / float(depth_max)
    rgb = (_jet_lookup(t) * 255.0).astype(np.uint8)
    return rgb


def require_gui() -> None:
    """Raise an actionable error if open3d's GUI module is unavailable."""
    if O3DVisualizer is None or gui is None:
        raise RuntimeError(
            "This feature requires open3d >= 0.16 (O3DVisualizer + gui). "
            "Run `pip install --upgrade open3d` in your 9dtact conda env."
        )


# ----- MeshUpdate: the cross-thread payload ------------------------------------

@dataclass
class MeshUpdate:
    height_map: np.ndarray       # (nx, ny) float32 mm, copied at construction
    readings: Readings
    depth_max: float


__all__ = [
    "TactileMeshVisualizer",  # imported lazily below
    "Visualizer",              # legacy point-cloud visualizer (kept for backwards compat)
    "MeshUpdate",
    "Readings",
    "compute_readings",
    "jet_vertex_colors",
    "jet_uint8_image",
    "require_gui",
]


# ----- Legacy Visualizer (point-cloud) ----------------------------------------
# Preserved for callers that still use it. The new TactileMeshVisualizer
# supersedes it for 3D shape reconstruction.

class Visualizer:
    """Legacy point-cloud visualizer (pre-redesign).

    Renders the raw contact points with a hand-rolled red-green colormap.
    Kept so older scripts (and the force-visualizer's smoke path) keep
    working without modification.
    """

    def __init__(self, points):
        import open3d  # noqa: F401
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="9DTact-Shape_Reconstruction", width=2000, height=1600
        )

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.vis.add_geometry(self.pcd)

        self.colors = np.zeros([points.shape[0], 3])

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-25)
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.7)
        self.ctr.rotate(0, -40)
        self.ctr.set_front([0.4, -0.8, -1])
        self.ctr.set_lookat([5, -10, -5])
        self.ctr.set_up([0, 0, -1])
        self.vis.update_renderer()

    def update(self, points, gradients):
        np_colors = points[:, 2]
        if abs(np_colors.max()) > 0:
            np_colors = (np_colors - np_colors.min()) / (
                np_colors.max() - np_colors.min()
            )
        np_colors = np.ndarray.flatten(np_colors)
        np_colors[points[:, 2] <= 0.05] = 0
        t = np.clip(np_colors, 0, 1)
        self.colors[:, 0] = np.minimum(1, 2 * t)
        self.colors[:, 1] = np.minimum(1, 2 * (1 - t))
        self.colors[:, 2] = 0.2
        points_show = points.copy()
        points_show[:, 2] *= -1
        self.pcd.points = open3d.utility.Vector3dVector(points_show)
        self.pcd.colors = open3d.utility.Vector3dVector(self.colors)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        return self.pcd


# ----- TactileMeshVisualizer (GUI) --------------------------------------------
# Defined in a sibling module so headless tests can ignore it.
if O3DVisualizer is not None:
    from .visualizer_gui import TactileMeshVisualizer  # noqa: E402


# ----- Replay-mode entry point ------------------------------------------------

def _replay_main(frames_dir: str, fps: int = 30) -> int:
    """Drive the visualizer from a directory of frame_*.png BGR images."""
    import glob
    import os
    import time
    import cv2

    from sensor import Sensor  # type: ignore

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {frames_dir!r}")

    with open("shape_reconstruction/shape_config.yaml", encoding='utf-8') as f:
        cfg = __import__("yaml").safe_load(f)

    # Construct sensor WITHOUT opening the camera; use first frame as ref.
    ref = cv2.imread(frames[0])
    sensor = Sensor(cfg, calibrated=True, ref=ref)

    vis = TactileMeshVisualizer(sensor, cfg)
    interval = 1.0 / fps
    for path in frames:
        if vis.is_closed():
            break
        bgr = cv2.imread(path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = sensor.raw_image_2_height_map(gray)
        h = sensor.expand_image(h)
        vis.post_height_map(h)
        time.sleep(interval)
    return len(frames)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="9DTact visualizer")
    parser.add_argument("--replay", metavar="DIR", help="Replay frame_*.png images")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.replay:
        n = _replay_main(args.replay, fps=args.fps)
        print(f"replay: {n} frames")
    else:
        raise SystemExit("Run _3_Shape_Reconstruction.py for live mode.")