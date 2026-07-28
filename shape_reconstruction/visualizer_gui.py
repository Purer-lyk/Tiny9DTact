"""GUI side of the visualizer (O3DVisualizer-based).

Implements the TactileMeshVisualizer class. Separated from visualizer.py
so the pure-function helpers can be unit-tested without instantiating
Open3D.
"""
from __future__ import annotations

import queue
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
        self._updates: "queue.Queue[MeshUpdate]" = queue.Queue(maxsize=2)
        self._last_apply_ts = 0.0
        self._ema_fps: Optional[float] = None
        self._close_requested = threading.Event()

        app = gui.Application.instance
        app.initialize()

        w, h = 1600, 1000
        self._vis = O3DVisualizer(
            title="9DTact — Shape Reconstruction", width=w, height=h
        )
        self._vis.set_background(
            np.array([[0.05], [0.05], [0.07], [1.0]], dtype=np.float32)
        )

        # --- precompute mesh skeleton -------------------------------------
        self._nx = int(sensor.expand_x)
        self._ny = int(sensor.expand_y)
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

        self._set_default_camera()
        self._build_sidebar()

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
        except queue.Full:
            try:
                self._updates.get_nowait()
            except queue.Empty:
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

    def _center_xyz(self) -> np.ndarray:
        return np.array(
            [
                self._nx * self._pixel_per_mm / 2,
                -self._ny * self._pixel_per_mm / 2,
                0.0,
            ],
            dtype=np.float32,
        )

    def _set_default_camera(self) -> None:
        center = self._center_xyz()
        extent = max(self._nx, self._ny) * self._pixel_per_mm
        eye = center + np.array(
            [extent * 0.7, -extent * 0.7, extent * 0.9], dtype=np.float32
        )
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._vis.setup_camera(60.0, center, eye, up)

    def _set_top_view(self) -> None:
        center = self._center_xyz()
        eye = center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self._vis.setup_camera(60.0, center, eye, up)

    def _set_side_view(self) -> None:
        center = self._center_xyz()
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
        for b in (
            self._btn_default, self._btn_top, self._btn_side, self._btn_reset
        ):
            btns.add_child(b)
        sidebar.add_child(btns)

        self._vis.add_child(sidebar)

    def _on_tick(self):
        try:
            update = self._updates.get_nowait()
        except queue.Empty:
            if self._close_requested.is_set():
                gui.Application.instance.quit()
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
        verts = np.column_stack([self._xy, h.ravel()])
        self._mesh.vertices = o3d.utility.Vector3dVector(verts)
        colors = jet_vertex_colors(h, update.depth_max)
        self._mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self._mesh.compute_vertex_normals()
        # Re-adding geometry is the cheapest way to push vertex-color updates
        # through O3DVisualizer in open3d >= 0.16.
        self._vis.remove_geometry(_MESH_NAME)
        self._vis.add_geometry(_MESH_NAME, self._mesh)
        self._dm_widget.update_image(jet_uint8_image(h, update.depth_max))
        r = update.readings
        self._val_max.text = f"{r.max_depth:.2f} mm"
        self._val_area.text = f"{r.contact_area_mm2:.1f} mm²"
        if r.contact_center_mm[0] is None:
            self._val_center.text = "—"
        else:
            cx, cy = r.contact_center_mm
            self._val_center.text = f"({cx:.1f}, {cy:.1f}) mm"