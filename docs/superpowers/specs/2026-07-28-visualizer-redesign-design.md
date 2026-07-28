# Design: 9DTact 3D Shape Reconstruction Visualizer Redesign

- **Date:** 2026-07-28
- **Status:** Approved (pending PR)
- **Owner:** ClaudePartner (on behalf of Purer-lyk)
- **Branch:** `feat/visualizer-redesign`

## 1. Problem

The current `shape_reconstruction/visualizer.py` uses `open3d.visualization.Visualizer` (legacy API) to render a point cloud with a hand-rolled red-green colormap, a hardcoded camera, no depth visualization aside from raw points, and a non-resizable 2000×1600 window. The visual presentation is dated, information density is low, and switching views requires dragging the mouse without any way to return to a known good viewpoint.

The goal is a clean, vivid 3D reconstruction window where the **shape of the press is the primary subject** (3D mesh surface, jet colormap with a deep-blue base), and complementary information (2D depth map, live readings, preset views) lives in a native sidebar.

## 2. Goals & Non-Goals

**Goals**
- Single-window experience (no separate cv2 window).
- Smooth mesh surface built from the height-map grid, lit by Filament.
- Vivid jet colormap on both 3D mesh and 2D depth map, sharing one normalization range so colors are semantically consistent.
- Sidebar with: 2D depth map (with mm colorbar), live readings (max depth, contact area, contact center, FPS), preset view buttons (default 45° / top / side / reset).
- Mouse drag rotate + scroll zoom remain available at all times; buttons are quick-return helpers, not replacements.
- ≥25 FPS sustained on the current Windows test machine.

**Non-Goals (this iteration)**
- Depth profile cross-section curve (deferred to a later PR).
- Sensor shell .obj overlay (the current code already commented them out; keep it that way).
- Loading / saving contact sequences.
- Multi-touch (multiple disconnected contact regions) — already supported by the existing height map, no change required.

## 3. Architecture

### 3.1 Module layout

The change is contained to `shape_reconstruction/visualizer.py` and the call site `_3_Shape_Reconstruction.py`. No changes to `sensor.py`, `camera.py`, or any force-estimation code.

```
shape_reconstruction/
├── visualizer.py          # NEW: TactileMeshVisualizer (O3DVisualizer-based)
├── sensor.py              # unchanged
├── camera.py              # unchanged
└── …
_3_Shape_Reconstruction.py # CALL SITE: swap Visualizer(...) for TactileMeshVisualizer(...);
                           # the loop already passes height_map, so the change is a one-line
                           # argument and a one-line class name swap.
```

### 3.2 Threading model

Open3D GUI requires all GUI mutations on the main thread. The reconstruction loop is CPU-bound (height-map computation, mesh regeneration). We use Open3D's standard pattern:

- **Main thread:** `gui.Application.instance.run()` drives the event loop. A timer callback (every ~30 ms) drains a thread-safe queue of pending mesh/readings updates posted by the worker.
- **Worker thread:** runs the OpenCV/camera loop, computes `height_map`, builds a `MeshUpdate` dataclass, and `Application.instance.post_to_main_thread(...)` puts it on the queue.
- **`_3_Shape_Reconstruction.py`:** keeps the same `while sensor.cap.isOpened()` shape but defers the body to a worker function; the main thread spawns both the GUI and the worker.

We considered a single-thread model with `run_one_tick()` between camera reads, but the synchronous 30 ms GUI tick would dominate the camera read budget and risk missing frames. Worker + post_to_main_thread is the canonical Open3D pattern and is what the upstream Open3D examples use for live sensors.

### 3.3 Class API

```python
class TactileMeshVisualizer:
    def __init__(self, cfg, sensor):
        """Build the GUI window, sidebar widgets, mesh resource, and start the app loop."""
    def post_height_map(self, height_map):  # called from worker thread
        """Snapshot the height map into a MeshUpdate and post to the GUI thread."""
    def run(self):  # called from main thread
        """Block on gui.Application.instance.run()."""
    def close(self):  # called from worker on shutdown
        """Signal the GUI to exit cleanly."""
```

The dataclass `MeshUpdate(height_map, readings, depth_colormap_image)` is the unit of communication between threads. All numpy arrays are copied at construction so the worker can immediately overwrite the buffer.

## 4. Components

### 4.1 3D scene (`O3DVisualizer` widget)

- **Mesh:** `open3d.geometry.TriangleMesh` built once from a regular `nx × ny` grid. Triangle indices are precomputed (≈2(nx−1)(ny−1) triangles); per frame only vertex positions (z) and vertex colors are updated via `mesh.vertices = Vector3dVector(...)` and `mesh.colors = Vector3dVector(...)`.
- **Material:** Filament lit material with vertex colors enabled. No textures.
- **Vertex colors:** computed from z using jet colormap with the same normalization range as the 2D depth map. Non-contact vertices (z ≤ contact_threshold) are colored with `jet(0)` (deep blue), not black.
- **Background:** dark navy `#0d0d11` (Filament `scene.set_background`).
- **Lighting:** Filament default directional + ambient; sufficient for the matte mesh.
- **Axis triad:** Open3D's built-in `coordinate frame` widget, anchored at the origin corner of the sensor grid.

### 4.2 Sidebar (`gui.Vert` panel)

Three labeled sections separated by horizontal dividers, all widgets themed in Open3D's dark palette.

1. **2D Depth Map (top view)**
   - `gui.ImageWidget` showing the jet-colored height map, scaled to fit the available width.
   - Beside it, a vertical `ImageWidget` acting as a colorbar with a 4-tick label (0, 0.4, 0.8, max) in mm, rendered once at init and refreshed only when the normalization max changes by >10% (to avoid flicker).

2. **Readings** (4 rows, label / monospace value)
   - **Max depth** (mm) — `height_map.max()`
   - **Contact area** (mm²) — `np.sum(height_map > contact_threshold) * pixel_per_mm ** 2`
   - **Contact center** (mm, mm) — area-weighted centroid of contact pixels
   - **FPS** — exponential moving average of frame interval, refreshed every 10 frames

3. **View** (4 buttons in a 2×2 grid)
   - **Default 45°** — preset camera defined in code: look at the sensor center, 45° elevation, azimuth such that the gel plane tilts toward the viewer.
   - **Top view** — looking straight down at the sensor plane.
   - **Side view** — looking at the sensor from the side along the +x axis.
   - **Reset view** — restores the default 45° view AND clears the user's drag/zoom transform.

Buttons invoke `O3DVisualizer.setup_camera(...)` with the corresponding camera parameters.

### 4.3 Live mesh update

For each incoming `MeshUpdate`:

1. Copy the height-map z into the preallocated vertex buffer; non-contact pixels (z ≤ threshold) clamp to 0 (so the plane is visible at deep blue, not "punched through").
2. Compute jet vertex colors from the same `depth_max` used by the colorbar; assign via `mesh.colors`.
3. Update `ImageWidget` for the 2D map (encoded as a single-channel uint16 → jet LUT → RGB bytes).
4. Recompute readings and update the four value labels.
5. `mesh.has_vertex_normals()` is set false; Filament computes flat normals from the updated vertex z, giving the surface real shape under lighting.

## 5. Data flow

```
camera BGR frame
   │  sensor.get_rectify_crop_image()
   ▼
sensor.raw_image_2_height_map(img_GRAY)  → height_map (nx × ny float32, mm)
sensor.expand_image(height_map)           → height_map_expanded
   │
   ▼
worker thread:
   readings  = compute_readings(height_map_expanded)        # max/area/centroid
   depth_max = max(readings.max_depth, prev_depth_max * 0.95)
   mesh_upd  = MeshUpdate(height_map_expanded, readings, depth_max)
   visualizer.post_height_map(mesh_upd)                     # thread-safe queue
   │
   ▼ (posted to main thread)
main thread (gui timer):
   apply mesh update to O3DVisualizer scene + sidebar widgets
   ── user can drag/zoom freely; preset buttons restore the camera ──
```

The same `depth_max` drives both the 3D mesh vertex colors and the 2D depth map's colormap range, so a red spot in the 2D map and a red spot in the 3D mesh always correspond to the same depth value.

## 6. Error handling & edge cases

- **Camera disconnected (`cap.isOpened()` returns false on a later iteration):** worker stops posting updates; sidebar shows "No signal" label; 3D scene keeps the last frame. `close()` is called and the worker exits.
- **No contact (entire height map ≤ contact_threshold):** mesh is the flat sensor plane colored deep blue; readings show 0.00 mm / 0.0 mm² / "—" / live FPS.
- **open3d version check:** `import open3d` followed by `from open3d.visualization import O3DVisualizer, gui`. If the import fails (older open3d without the gui module), raise an actionable error: `"This redesign requires open3d >= 0.16 (with O3DVisualizer). Run: pip install --upgrade open3d"`. Detect at module import time so users get a clear message instead of a cryptic AttributeError mid-run.
- **Invalid height_map shape:** if the incoming `height_map` shape doesn't match `expand_x × expand_y`, log a warning and skip that frame.
- **NaN / inf in height_map:** clamped to 0 before further processing.

## 7. Testing & verification

UI cannot be automated. The implementation includes a **replay mode** (`visualizer.py --replay path/to/frames/`):

- Reads a sequence of saved BGR frames from disk (or any folder of PNGs matching `frame_*.png`).
- Feeds them through the same `Sensor.raw_image_2_height_map` pipeline.
- Renders at a configurable FPS into the GUI.

Test scenarios (recorded calibration data is available at `shape_reconstruction/calibration/sensor_118/`):

| Scenario | Input | Expected |
| --- | --- | --- |
| No contact | sequence with no press | flat blue plane, all readings 0, FPS ≥ 25 |
| Light contact | shallow press frames | small low-saturation bump, blue→cyan only |
| Ball press (4 mm) | calibration ball frames | distinct round bump, max depth matches `Pixel_to_Depth` value, contact area consistent with R=4 mm cap |
| Multi-bump | manual press across regions | multiple bumps visible in 2D map and 3D mesh simultaneously |

Manual checks per scenario:
- 3D mesh lit, no obvious triangulation artifacts at edges.
- 2D depth map matches the mesh when rotated to top view.
- All four preset buttons return the camera to the documented orientation.
- FPS counter in the sidebar matches a stopwatch within 10% over 30 seconds.

## 8. Out of scope (deferred)

- Cross-section depth profile plot.
- Multi-sensor display.
- Recording/replay of recorded presses with synchronized camera playback.
- Saving a snapshot of the current view.

These will get their own design docs when picked up.