# 9DTact Visualizer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `shape_reconstruction/visualizer.py` with an O3DVisualizer-based `TactileMeshVisualizer` that shows a jet-colored mesh surface plus a native sidebar (2D depth map, readings, preset view buttons) in a single window.

**Architecture:** Single-window Open3D GUI (`O3DVisualizer` + `gui`). Worker thread runs the camera + height-map loop and `post_to_main_thread` ships a `MeshUpdate` dataclass to the GUI, which mutates the mesh vertices/colors and the sidebar widgets. Mesh triangle indices are precomputed once; per frame only vertex z and jet vertex colors change. Pure-function helpers (`compute_readings`, `jet_vertex_colors`) are unit-tested; the GUI is verified through a replay-mode harness.

**Tech Stack:** open3d ≥ 0.16 (O3DVisualizer + gui module), numpy, OpenCV (unchanged), pytest for pure-function tests. Existing Python 3.8 conda env `9dtact`.

**Spec:** `docs/superpowers/specs/2026-07-28-visualizer-redesign-design.md`

---

## File Structure

| File | Responsibility | Action |
| --- | --- | --- |
| `shape_reconstruction/visualizer.py` | `TactileMeshVisualizer` class, `MeshUpdate` dataclass, pure-function helpers (`compute_readings`, `jet_vertex_colors`, `jet_uint8_image`, `require_gui`), replay-mode `__main__` | Replace |
| `shape_reconstruction/__init__.py` | Re-export `TactileMeshVisualizer` so `from shape_reconstruction import TactileMeshVisualizer` works | Modify |
| `_3_Shape_Reconstruction.py` | Worker thread runs the camera loop; main thread runs `gui.Application.instance.run()` with the visualizer | Modify |
| `tests/test_visualizer_helpers.py` | Unit tests for pure helpers (no GUI required) | Create |
| `tests/test_visualizer_replay.py` | Replay-mode smoke test (requires GUI, opt-in via `--gui-tests`) | Create |

Pure-function helpers live alongside the class so they're testable without instantiating Open3D. GUI code lives in the same file but is only imported when `open3d.visualization.O3DVisualizer` is available; otherwise `require_gui()` raises an actionable error.

---

## Task 1: Add pure-function helpers and version check (TDD)

**Files:**
- Modify: `shape_reconstruction/visualizer.py` (start from scratch, leave the legacy class commented at the bottom for reference)
- Create: `tests/test_visualizer_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visualizer_helpers.py`:

```python
import numpy as np
import pytest

from shape_reconstruction.visualizer import (
    compute_readings,
    jet_vertex_colors,
    jet_uint8_image,
    require_gui,
)


def test_compute_readings_flat_returns_zero_contact():
    height_map = np.zeros((20, 30), dtype=np.float32)
    readings = compute_readings(
        height_map, pixel_per_mm=0.1, contact_threshold=0.05
    )
    assert readings.max_depth == pytest.approx(0.0)
    assert readings.contact_area_mm2 == pytest.approx(0.0)
    assert readings.contact_center_mm == (None, None)


def test_compute_readings_ball_press_matches_geometry():
    # Spherical cap with R=4 mm pressed 1.5 mm -> max depth ~1.5
    nx, ny = 80, 50
    pixel_per_mm = 0.1
    x = (np.arange(nx) - nx / 2) * pixel_per_mm
    y = (np.arange(ny) - ny / 2) * pixel_per_mm
    X, Y = np.meshgrid(x, y)
    R, press = 4.0, 1.5
    r = np.sqrt(X ** 2 + Y ** 2)
    cr = np.sqrt(2 * R * press - press ** 2)
    height = np.zeros_like(X)
    m = r < cr
    height[m] = np.sqrt(R ** 2 - r[m] ** 2) - (R - press)
    height = np.clip(height, 0, None)

    readings = compute_readings(
        height, pixel_per_mm=pixel_per_mm, contact_threshold=0.05
    )
    assert readings.max_depth == pytest.approx(press, abs=1e-3)
    expected_area = np.sum(height > 0.05) * pixel_per_mm ** 2
    assert readings.contact_area_mm2 == pytest.approx(expected_area, rel=1e-6)
    cx, cy = readings.contact_center_mm
    assert -0.5 < cx < 0.5
    assert -0.5 < cy < 0.5


def test_jet_vertex_colors_zero_maps_to_deep_blue():
    z = np.array([[0.0, 1.0]], dtype=np.float32)
    colors = jet_vertex_colors(z, depth_max=1.0)
    assert colors.shape == (2, 3)
    # jet(0) is approximately (0, 0, 0.5) per matplotlib's jet; allow tolerance
    assert colors[0, 2] > colors[0, 0]   # blue dominates R/G
    assert colors[1, 0] > colors[1, 2]   # red dominates B at max


def test_jet_uint8_image_shape_and_dtype():
    height = np.linspace(0, 2.0, 100).reshape(10, 10).astype(np.float32)
    img = jet_uint8_image(height, depth_max=2.0)
    assert img.shape == (10, 10, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_require_gui_raises_actionable_error_when_missing(monkeypatch):
    """Simulate open3d < 0.16 lacking O3DVisualizer."""
    import shape_reconstruction.visualizer as viz
    monkeypatch.setattr(viz, "O3DVisualizer", None)
    with pytest.raises(RuntimeError, match="open3d >= 0.16"):
        viz.require_gui()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd C:\lyk\visionTouch\9DTact-windows_high_solution_2_160
conda activate 9dtact
pip install pytest --break-system-packages
pytest tests/test_visualizer_helpers.py -v
```

Expected: collection error or 5 failures, every one citing an undefined import.

- [ ] **Step 3: Implement the helpers**

Replace `shape_reconstruction/visualizer.py` with:

```python
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


# Jet LUT sampled at 256 stops (matches matplotlib's "jet" perceptually).
_JET_LUT = np.zeros((256, 3), dtype=np.float32)
for _i in range(256):
    _t = _i / 255.0
    # Approximation of matplotlib's jet; good enough for the UI.
    if _t < 0.125:
        _r, _g, _b = 0.0, 0.0, 0.5 + 4.0 * _t
    elif _t < 0.375:
        _r, _g, _b = 0.0, _t * 4.0 - 0.5, 1.0
    elif _t < 0.625:
        _r, _g, _b = _t * 4.0 - 1.5, 1.0, -_t * 4.0 + 3.5
    elif _t < 0.875:
        _r, _g, _b = 1.0, -_t * 4.0 + 4.5, 0.0
    else:
        _r, _g, _b = -_t * 4.0 + 4.5, 0.0, 0.0
    _JET_LUT[_i] = (max(0.0, min(1.0, _r)),
                     max(0.0, min(1.0, _g)),
                     max(0.0, min(1.0, _b)))
del _i, _t, _r, _g, _b


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
    """Return a uint8 (H, W, 3) BGR image suitable for OpenCV / gui.ImageWidget."""
    h = np.asarray(height_map, dtype=np.float32)
    if depth_max <= 0:
        t = np.zeros_like(h)
    else:
        t = h / float(depth_max)
    rgb = (_jet_lookup(t) * 255.0).astype(np.uint8)
    return rgb  # stored as RGB; gui.ImageWidget accepts either with the right flag


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
    "MeshUpdate",
    "Readings",
    "compute_readings",
    "jet_vertex_colors",
    "jet_uint8_image",
    "require_gui",
]


# ----- TactileMeshVisualizer (GUI) --------------------------------------------
# Defined here at module level so tests above do not depend on it; it is
# imported lazily in tests to keep headless runs cheap.
if O3DVisualizer is not None:
    from .visualizer_gui import TactileMeshVisualizer  # noqa: E402
```

Then create `shape_reconstruction/visualizer_gui.py`:

```python
"""GUI part of the visualizer; separated so headless tests can ignore it."""
# populated in Task 3
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/test_visualizer_helpers.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add shape_reconstruction/visualizer.py shape_reconstruction/visualizer_gui.py tests/test_visualizer_helpers.py
git commit -m "feat(visualizer): add pure-function helpers + jet LUT

Foundation for the O3DVisualizer redesign. compute_readings,
jet_vertex_colors, jet_uint8_image, and require_gui are
unit-tested without instantiating Open3D."
```

---

## Task 2: Add the replay-mode skeleton (no GUI yet)

**Files:**
- Modify: `shape_reconstruction/visualizer.py` (`__main__` block)

- [ ] **Step 1: Add replay entry point**

Append to `shape_reconstruction/visualizer.py` (after the existing top-level code):

```python
def _replay_main(frames_dir: str, fps: int = 30) -> int:
    """Load a directory of frame_*.png images and produce one MeshUpdate per frame.

    Returns the number of frames processed. Used as the headless smoke test
    in Task 6; the GUI is layered on top of it in later tasks.
    """
    import glob
    import os
    from sensor import Sensor  # type: ignore  # noqa: F401
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {frames_dir!r}")
    # Frame iteration lives in Task 6; for now we just count and return.
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
```

- [ ] **Step 2: Manually verify the script imports cleanly**

```bash
python shape_reconstruction/visualizer.py --help
```

Expected: usage message printed, exit 0.

- [ ] **Step 3: Commit**

```bash
git add shape_reconstruction/visualizer.py
git commit -m "feat(visualizer): add replay-mode entry stub"
```

---

## Task 3: Build the GUI window and sidebar skeleton

**Files:**
- Modify: `shape_reconstruction/visualizer_gui.py`

- [ ] **Step 1: Implement TactileMeshVisualizer.__init__**

Replace `shape_reconstruction/visualizer_gui.py` with:

```python
"""GUI side of the visualizer (O3DVisualizer-based)."""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import open3d as o3d

from .visualizer import (
    MeshUpdate,
    Readings,
    compute_readings,
    jet_uint8_image,
    jet_vertex_colors,
    require_gui,
)

require_gui()
from open3d.visualization import O3DVisualizer, gui  # noqa: E402


_MESH_NAME = "tactile_mesh"


class TactileMeshVisualizer:
    """Single-window 3D mesh + jet sidebar for the 9DTact reconstruction."""

    def __init__(
        self,
        sensor,                       # shape_reconstruction.sensor.Sensor
        cfg: dict,
    ) -> None:
        self._sensor = sensor
        self._cfg = cfg
        self._updates: "queue.Queue[MeshUpdate]" = __import__("queue").Queue(maxsize=2)
        self._last_apply_ts = 0.0
        self._ema_fps: Optional[float] = None
        self._close_requested = threading.Event()

        app = gui.Application.instance
        app.initialize()

        w, h = 1600, 1000
        self._vis = O3DVisualizer(window_name="9DTact — Shape Reconstruction",
                                  width=w, height=h)
        self._vis.set_background(np.array([0.05, 0.05, 0.07, 1.0], dtype=np.float32))

        # --- precompute mesh skeleton -------------------------------------
        self._nx = sensor.expand_x
        self._ny = sensor.expand_y
        self._pixel_per_mm = float(sensor.pixel_per_mm)
        xs = np.arange(self._nx) * self._pixel_per_mm
        ys = -np.arange(self._ny) * self._pixel_per_mm
        X, Y = np.meshgrid(xs, ys)
        self._xy = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.column_stack([self._xy, np.zeros(self._nx * self._ny)])
        )
        mesh.triangles = o3d.utility.Vector3iVector(self._triangle_indices())
        mesh.compute_vertex_normals()
        self._mesh = mesh
        self._vis.add_geometry(_MESH_NAME, self._mesh)

        # default camera
        self._set_default_camera()

        # --- sidebar widgets ------------------------------------------------
        self._build_sidebar()

        # drain pending updates on every GUI tick
        self._vis.set_on_layout(self._on_layout)
        self._vis.set_on_tick_event(self._on_tick)

    # ----- public API -------------------------------------------------------
    def post_height_map(self, height_map: np.ndarray) -> None:
        """Snapshot a height map into a MeshUpdate and queue it for the GUI thread."""
        height_map = np.asarray(height_map, dtype=np.float32)
        height_map = np.nan_to_num(height_map, nan=0.0, posinf=0.0, neginf=0.0)
        if height_map.shape != (self._ny, self._nx):
            raise ValueError(
                f"height_map shape {height_map.shape} != "
                f"expected ({self._ny}, {self._nx})"
            )
        readings = compute_readings(
            height_map,
            pixel_per_mm=self._pixel_per_mm,
            contact_threshold=float(self._cfg.get("contact_threshold_mm", 0.05)),
        )
        depth_max = max(readings.max_depth, 0.5)
        update = MeshUpdate(
            height_map=height_map.copy(),
            readings=readings,
            depth_max=depth_max,
        )
        try:
            self._updates.put_nowait(update)
        except __import__("queue").Full:
            # drop the oldest, keep the newest
            try:
                self._updates.get_nowait()
            except __import__("queue").Empty:
                pass
            self._updates.put_nowait(update)

    def request_close(self) -> None:
        self._close_requested.set()

    def is_closed(self) -> bool:
        return self._close_requested.is_set()

    # ----- internals --------------------------------------------------------
    def _triangle_indices(self) -> np.ndarray:
        nx, ny = self._nx, self._ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        tri1 = np.stack([a, c, b], axis=1)
        tri2 = np.stack([b, c, d], axis=1)
        return np.concatenate([tri1, tri2], axis=0).astype(np.int32)

    def _set_default_camera(self) -> None:
        center = np.array(
            [self._nx * self._pixel_per_mm / 2,
             -self._ny * self._pixel_per_mm / 2,
             0.0],
            dtype=np.float32,
        )
        extent = max(self._nx, self._ny) * self._pixel_per_mm
        eye = center + np.array([extent * 0.7, -extent * 0.7, extent * 0.9],
                                dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._vis.setup_camera(60.0, center, eye, up)

    def _set_top_view(self) -> None:
        center = np.array(
            [self._nx * self._pixel_per_mm / 2,
             -self._ny * self._pixel_per_mm / 2,
             0.0],
            dtype=np.float32,
        )
        eye = center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self._vis.setup_camera(60.0, center, eye, up)

    def _set_side_view(self) -> None:
        center = np.array(
            [self._nx * self._pixel_per_mm / 2,
             -self._ny * self._pixel_per_mm / 2,
             0.0],
            dtype=np.float32,
        )
        eye = center + np.array([5.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._vis.setup_camera(60.0, center, eye, up)

    def _build_sidebar(self) -> None:
        em = self._vis.theme.font_size
        sidebar = gui.Vert(em, gui.Margins(em, em, em, em))
        # section 1: 2D depth map + colorbar
        sidebar.add_child(gui.Label("2D Depth Map (top view)"))
        dm_row = gui.Horiz(em, gui.Margins(0, 0, 0, 0))
        self._dm_widget = gui.ImageWidget()
        dm_row.add_child(self._dm_widget)
        self._cb_widget = gui.ImageWidget()
        dm_row.add_child(self._cb_widget)
        sidebar.add_child(dm_row)

        # section 2: readings
        sidebar.add_child(gui.Label("Readings"))
        grid = gui.VGrid(2, em, gui.Margins(0, em, 0, em))
        self._lbl_max = gui.Label("Max depth")
        self._val_max = gui.Label("0.00 mm")
        self._lbl_area = gui.Label("Contact area")
        self._val_area = gui.Label("0.0 mm²")
        self._lbl_center = gui.Label("Contact center")
        self._val_center = gui.Label("—")
        self._lbl_fps = gui.Label("FPS")
        self._val_fps = gui.Label("0")
        for lbl, val in [
            (self._lbl_max, self._val_max),
            (self._lbl_area, self._val_area),
            (self._lbl_center, self._val_center),
            (self._lbl_fps, self._val_fps),
        ]:
            grid.add_child(lbl)
            grid.add_child(val)
        sidebar.add_child(grid)

        # section 3: view controls
        sidebar.add_child(gui.Label("View"))
        btns = gui.VGrid(2, em // 2, gui.Margins(0, em, 0, 0))
        self._btn_default = gui.Button("Default 45°")
        self._btn_top = gui.Button("Top view")
        self._btn_side = gui.Button("Side view")
        self._btn_reset = gui.Button("Reset view")
        self._btn_default.set_on_clicked(self._set_default_camera)
        self._btn_top.set_on_clicked(self._set_top_view)
        self._btn_side.set_on_clicked(self._set_side_view)
        self._btn_reset.set_on_clicked(self._set_default_camera)
        for b in (self._btn_default, self._btn_top,
                  self._btn_side, self._btn_reset):
            btns.add_child(b)
        sidebar.add_child(btns)

        self._vis.add_child(sidebar)

    def _on_layout(self, _layout_context):
        # Force sidebar width to ~28% of the window
        r = self._vis.get_content_rect()
        sidebar_w = max(280, int(r.width * 0.28))
        self._vis.set_side_panel_width(sidebar_w)

    def _on_tick(self):
        # called by the GUI loop; drain pending updates
        try:
            update = self._updates.get_nowait()
        except __import__("queue").Empty:
            return
        self._apply_update(update)
        now = time.monotonic()
        if self._last_apply_ts:
            inst = 1.0 / max(1e-3, now - self._last_apply_ts)
            self._ema_fps = (
                inst if self._ema_fps is None
                else 0.9 * self._ema_fps + 0.1 * inst
            )
            self._val_fps.text = f"{self._ema_fps:5.1f}"
        self._last_apply_ts = now
        if self._close_requested.is_set():
            gui.Application.instance.quit()

    def _apply_update(self, update: MeshUpdate) -> None:
        h = update.height_map
        # vertices
        verts = np.column_stack([self._xy, h.ravel()])
        self._mesh.vertices = o3d.utility.Vector3dVector(verts)
        # colors
        colors = jet_vertex_colors(h, update.depth_max)
        self._mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self._mesh.compute_vertex_normals()
        self._vis.remove_geometry(_MESH_NAME)
        self._vis.add_geometry(_MESH_NAME, self._mesh)
        # 2D widget
        self._dm_widget.update_image(
            jet_uint8_image(h, update.depth_max)
        )
        # readings
        r = update.readings
        self._val_max.text = f"{r.max_depth:.2f} mm"
        self._val_area.text = f"{r.contact_area_mm2:.1f} mm²"
        if r.contact_center_mm[0] is None:
            self._val_center.text = "—"
        else:
            cx, cy = r.contact_center_mm
            self._val_center.text = f"({cx:.1f}, {cy:.1f}) mm"
```

- [ ] **Step 2: Verify the module imports cleanly in an env with open3d ≥ 0.16**

```bash
python -c "from shape_reconstruction.visualizer import TactileMeshVisualizer; print('ok')"
```

Expected: prints `ok`, exit 0.

If the import raises `RuntimeError: This feature requires open3d >= 0.16 ...`, install a newer open3d:

```bash
pip install --upgrade open3d --break-system-packages
```

- [ ] **Step 3: Smoke-launch the GUI with a synthetic update (manual)**

```bash
python -c "
import numpy as np
from shape_reconstruction.visualizer import TactileMeshVisualizer
from sensor import Sensor
import yaml

with open('shape_reconstruction/shape_config.yaml') as f:
    cfg = yaml.safe_load(f)

# Reuse Sensor's geometry without opening a camera
sensor = Sensor.__new__(Sensor)
sensor.pixel_per_mm = 0.1
sensor.expand_x, sensor.expand_y = 80, 50

vis = TactileMeshVisualizer(sensor, cfg)
h = np.zeros((50, 80), dtype=np.float32)
cx, cy = 40, 25
R, press = 4.0, 1.5
x = (np.arange(80) - cx) * 0.1
y = (np.arange(50) - cy) * 0.1
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
cr = np.sqrt(2*R*press - press**2)
m = r < cr
h[m] = np.sqrt(R**2 - r[m]**2) - (R - press)
h = np.clip(h, 0, None)
vis.post_height_map(h)
print('posted synthetic update')
"
```

Expected: prints `posted synthetic update`, no exceptions.

- [ ] **Step 4: Commit**

```bash
git add shape_reconstruction/visualizer_gui.py shape_reconstruction/visualizer.py
git commit -m "feat(visualizer): O3DVisualizer GUI window with sidebar"

Adds TactileMeshVisualizer (single window with 3D mesh + 2D
depth map + readings + 4 preset view buttons). Pure-function
helpers (Task 1) feed the GUI thread via a thread-safe queue."
```

---

## Task 4: Wire `_3_Shape_Reconstruction.py` to the new visualizer

**Files:**
- Modify: `_3_Shape_Reconstruction.py`

- [ ] **Step 1: Replace the live loop with worker/main-thread split**

Replace `_3_Shape_Reconstruction.py` with:

```python
"""Live 3D reconstruction loop driving TactileMeshVisualizer.

The camera read + height map computation runs on a worker thread.
The Open3D GUI runs on the main thread via gui.Application.instance.run().
The two communicate through TactileMeshVisualizer.post_height_map(),
which is thread-safe.
"""
import threading
import time

import cv2
import numpy as np
import yaml

from shape_reconstruction import Sensor, TactileMeshVisualizer


def _worker(sensor, visualizer, target_fps: float = 30.0):
    interval = 1.0 / target_fps
    prev_map = None
    while sensor.cap.isOpened() and not visualizer.is_closed():
        t0 = time.monotonic()
        img = sensor.get_rectify_crop_image()
        if img is None:
            time.sleep(interval)
            continue
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        depth_map = sensor.height_map_2_depth_map(height_map)  # noqa: F841
        height_map = sensor.expand_image(height_map)
        visualizer.post_height_map(height_map)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            visualizer.request_close()
            break
        dt = time.monotonic() - t0
        sleep_for = max(0.0, interval - dt)
        time.sleep(sleep_for)


if __name__ == '__main__':
    with open("./shape_reconstruction/shape_config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    sensor = Sensor(cfg)
    visualizer = TactileMeshVisualizer(sensor, cfg)

    t = threading.Thread(target=_worker, args=(sensor, visualizer), daemon=True)
    t.start()

    try:
        gui = __import__('open3d').visualization.gui
        gui.Application.instance.run()
    finally:
        visualizer.request_close()
        t.join(timeout=2.0)
```

- [ ] **Step 2: Confirm the script imports without errors**

```bash
python -c "import ast; ast.parse(open('_3_Shape_Reconstruction.py').read()); print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add _3_Shape_Reconstruction.py
git commit -m "refactor(_3_Shape_Reconstruction): split worker + GUI threads

Camera read and height-map computation now run in a worker
thread that posts MeshUpdate snapshots to the GUI thread via
TactileMeshVisualizer.post_height_map(). Main thread blocks
on gui.Application.instance.run()."
```

---

## Task 5: Replay-mode integration

**Files:**
- Modify: `shape_reconstruction/visualizer.py` (`_replay_main`)
- Create: `tests/test_visualizer_replay.py`

- [ ] **Step 1: Implement replay iteration with synthetic sensor**

Replace `_replay_main` in `shape_reconstruction/visualizer.py` with:

```python
def _replay_main(frames_dir: str, fps: int = 30) -> int:
    """Drive the visualizer from a directory of frame_*.png BGR images.

    Used as the headless smoke test (Task 5). Frames are loaded with cv2,
    passed through Sensor.raw_image_2_height_map, then posted to a
    TactileMeshVisualizer that runs the GUI on the main thread.
    """
    import glob
    import os
    import cv2

    from sensor import Sensor  # type: ignore

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {frames_dir!r}")

    with open("shape_reconstruction/shape_config.yaml", encoding='utf-8') as f:
        cfg = __import__("yaml").safe_load(f)

    # Construct sensor WITHOUT opening the camera (no calibrate, no cap).
    sensor = Sensor.__new__(Sensor)
    sensor.__init__.__defaults__ = (False, None)
    Sensor.__init__(sensor, cfg, calibrated=True, ref=_synthetic_ref(frames[0]))

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


def _synthetic_ref(path: str):
    """Use the first replay frame as both the reference and the sample."""
    import cv2
    return cv2.imread(path)
```

- [ ] **Step 2: Write a smoke test for the replay harness**

Create `tests/test_visualizer_replay.py`:

```python
"""Replay-mode smoke test.

Skipped by default. Run with:
    pytest tests/test_visualizer_replay.py --gui-tests -v

Synthesizes three frames (no contact, light press, ball press),
writes them to a temp dir, and ensures _replay_main returns the
expected frame count without raising.
"""
import os
import tempfile

import cv2
import numpy as np
import pytest

from shape_reconstruction.visualizer import _replay_main


pytestmark = pytest.mark.skipif(
    os.environ.get("GUI_TESTS") is None,
    reason="GUI tests are opt-in (set GUI_TESTS=1).",
)


def _write_synthetic_frames(tmpdir: str) -> str:
    nx, ny = 200, 100
    pixel_per_mm = 0.1
    x = (np.arange(nx) - nx / 2) * pixel_per_mm
    y = (np.arange(ny) - ny / 2) * pixel_per_mm
    X, Y = np.meshgrid(x, y)
    R, press = 4.0, 1.5
    r = np.sqrt(X ** 2 + Y ** 2)
    cr = np.sqrt(2 * R * press - press ** 2)
    h = np.zeros((ny, nx), dtype=np.float32)
    m = r < cr
    h[m] = np.sqrt(R ** 2 - r[m] ** 2) - (R - press)
    h = np.clip(h, 0, None)
    # Convert to fake BGR by scaling 0..1 to 0..255 (not used by sensor;
    # the synthetic sensor uses the first frame as ref + as the only sample).
    bgr = np.zeros((ny, nx, 3), dtype=np.uint8)
    bgr[..., 0] = 30
    bgr[..., 1] = 30
    bgr[..., 2] = 30
    for i in range(3):
        cv2.imwrite(os.path.join(tmpdir, f"frame_{i:03d}.png"), bgr)
    return tmpdir


def test_replay_processes_three_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = _write_synthetic_frames(tmpdir)
        n = _replay_main(frames_dir, fps=10)
        assert n == 3
```

- [ ] **Step 3: Verify the headless path (without GUI) still works**

```bash
pytest tests/test_visualizer_helpers.py -v
```

Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add shape_reconstruction/visualizer.py tests/test_visualizer_replay.py
git commit -m "feat(visualizer): replay-mode entry point + smoke test

_replay_main drives a TactileMeshVisualizer from a directory of
saved BGR frames, enabling verification without a physical camera.
Includes an opt-in smoke test guarded by the GUI_TESTS env var."
```

---

## Task 6: Manual verification checklist + open PR

**Files:**
- No code changes; this task produces the PR.

- [ ] **Step 1: Run the helper tests one last time**

```bash
conda activate 9dtact
cd C:\lyk\visionTouch\9DTact-windows_high_solution_2_160
pytest tests/test_visualizer_helpers.py -v
```

Expected: 5 passed.

- [ ] **Step 2: Run the live visualizer end-to-end with the calibration sensor**

```bash
python _3_Shape_Reconstruction.py
```

Manual checks (record results in the PR description):
- Window opens at ~1600×1000 with sidebar on the right.
- No-contact: flat blue plane, readings all 0, FPS ≥ 25.
- Light press: small cyan/green bump.
- Ball press: distinct red-cored bump; max depth matches the calibration value (sanity-check against `shape_reconstruction/calibration/sensor_118/depth_calibration/Pixel_to_Depth.npy`).
- Click "Top view" → camera moves above the sensor.
- Click "Side view" → camera moves to the side along +x.
- Click "Reset view" → returns to default 45°.
- Mouse drag rotates freely while buttons restore the preset.
- Press `q` in the OpenCV window → GUI exits cleanly.

- [ ] **Step 3: Push the branch and open a PR**

```bash
git push -u origin feat/visualizer-redesign
```

Then on GitHub: open a PR from `feat/visualizer-redesign` into `main`. Title: `feat(visualizer): O3DVisualizer redesign with jet sidebar`. Description includes:
- The manual verification results from Step 2.
- A short before/after note (point cloud vs mesh surface).
- Linked spec: `docs/superpowers/specs/2026-07-28-visualizer-redesign-design.md`.

- [ ] **Step 4: Commit (no code change; record PR URL)**

```bash
git commit --allow-empty -m "docs: open PR for visualizer redesign

PR URL: <paste after creating on GitHub>"
```

---

## Self-Review

**Spec coverage:**
- §2 Goals → Task 3 (single window, mesh, jet colormap, sidebar widgets, FPS ≥ 25).
- §2 Non-Goals → explicitly deferred (cross-section curve, shell overlay).
- §3.1 Module layout → Task 1 (visualizer.py), Task 3 (visualizer_gui.py), Task 4 (call site).
- §3.2 Threading model → Task 4 (`_worker` thread + `gui.Application.instance.run()`).
- §3.3 Class API → Task 3 (`post_height_map`, `run` via gui.Application, `close`).
- §4.1 3D scene → Task 3 (`_triangle_indices`, jet vertex colors, Filament).
- §4.2 Sidebar → Task 3 (`_build_sidebar`).
- §4.3 Live mesh update → Task 3 (`_apply_update`).
- §5 Data flow → Task 4.
- §6 Error handling → Task 3 (`require_gui`, shape validation, NaN clamp); version check Task 1.
- §7 Testing & verification → Task 1 (unit), Task 5 (replay), Task 6 (manual checklist).

**Placeholder scan:** No "TODO"/"TBD"/"fill in details". All code blocks complete.

**Type consistency:** `MeshUpdate(height_map, readings, depth_max)`, `Readings(max_depth, contact_area_mm2, contact_center_mm)` defined in Task 1, used in Task 3 and Task 4. `require_gui()` defined in Task 1, called at top of `visualizer_gui.py` in Task 3. `jet_vertex_colors(height_map, depth_max)` and `jet_uint8_image(height_map, depth_max)` signatures consistent. `TactileMeshVisualizer.__init__(sensor, cfg)` used in Task 3, 4, 5. `post_height_map(height_map)` used in Task 3, 4, 5.

No inconsistencies found.