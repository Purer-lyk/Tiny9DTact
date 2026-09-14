"""3D shape reconstruction visualizer for 9DTact.

This module provides:

1. Legacy `Visualizer` class (point-cloud, preservation of the original
   9DTact open3d.visualization.Visualizer behaviour).
2. Pure-function helpers used by all new visualizers:
   - compute_readings(height_map, pixel_per_mm, contact_threshold)
   - jet_vertex_colors(height_map, depth_max)
   - jet_uint8_image(height_map, depth_max)
   - require_gui()
3. Cross-thread payload dataclass `MeshUpdate` plus the `Readings`
   dataclass.

Backends:
  - The legacy point-cloud `Visualizer` uses open3d 0.19's
    `open3d.visualization.Visualizer` (legacy API, stable).
  - New visualizers (e.g. VisPy-based) live in sibling modules like
    `shape_reconstruction.visualizer_vispy` and reuse the helpers
    defined here.

The jet colormap is used throughout the project for both 3D mesh vertex
colors and 2D depth maps, with a deep-blue base per the project's
visual style preference.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Legacy Visualizer still depends on open3d.
try:
    import open3d  # noqa: F401
except Exception:  # pragma: no cover
    open3d = None

# Vispy is imported at module level so tests can monkeypatch it.
try:
    import vispy  # noqa: F401
except Exception:  # pragma: no cover
    vispy = None

# PyQtGraph is the default backend used by the live visualizer.
try:
    import pyqtgraph  # noqa: F401
except Exception:  # pragma: no cover
    pyqtgraph = None


# ----- Legacy Visualizer (point cloud) ----------------------------------------

class Visualizer:
    """Legacy point-cloud visualizer (unchanged from the original 9DTact).

    Preserved for backwards compatibility with any callers that still
    use the original `_3_Shape_Reconstruction.py` flow. The new
    vispy-based UI lives in `visualizer_vispy.py`.
    """

    def __init__(self, points):
        if open3d is None:
            raise RuntimeError(
                "Legacy Visualizer requires open3d. Install open3d."
            )
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='9DTact-Shape_Reconstruction',
            width=2000, height=1600
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


# ----- Pure-function helpers --------------------------------------------------

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


# Jet LUT sampled at 256 stops (approximation of matplotlib's "jet").
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
    """Return a uint8 (H, W, 3) RGB image for display."""
    h = np.asarray(height_map, dtype=np.float32)
    if depth_max <= 0:
        t = np.zeros_like(h)
    else:
        t = h / float(depth_max)
    rgb = (_jet_lookup(t) * 255.0).astype(np.uint8)
    return rgb


def require_gui() -> None:
    """Raise an actionable error if no GUI backend is available.

    The default backend is PyQtGraph + PyQt5 (see visualizer_pg.py).
    """
    if vispy is None and pyqtgraph is None:
        raise RuntimeError(
            "A GUI backend is required. Install one of:\n"
            "  pip install pyqtgraph==0.13.4 PyQt5==5.15.10\n"
            "  pip install vispy==0.14.3 PyQt5==5.15.10"
        )


# ----- Cross-thread payload ---------------------------------------------------

@dataclass
class MeshUpdate:
    height_map: np.ndarray       # (nx, ny) float32 mm, copied at construction
    readings: Readings
    depth_max: float


__all__ = [
    "Visualizer",
    "MeshUpdate",
    "Readings",
    "compute_readings",
    "jet_vertex_colors",
    "jet_uint8_image",
    "require_gui",
]
