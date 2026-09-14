# Design: 9DTact 3D Reconstruction — WebGL + Three.js Viewer

- **Date:** 2026-07-29
- **Status:** Approved (pending PR)
- **Owner:** ClaudePartner (on behalf of Purer-lyk)
- **Branch:** `feat/vispy-viewer` (later renamed to `feat/web-viewer`)

## 1. Problem

We need a real-time 3D visualization for the 9DTact shape
reconstruction. Two previous attempts (PyQt5 + Open3D 0.19
`O3DVisualizer`, PyQt5 + VisPy 0.14, PyQt5 + PyQtGraph 0.13) all hit
**GPU-side bottlenecks on the user's integrated GPU**:
- Open3D 0.19 `O3DVisualizer` is unstable when embedded in a
  `gui.Window` (silently crashes after a few frames).
- VisPy 0.14 re-uploads the full vertex buffer on every `Mesh.set_data()`
  call, which caps us at ~10 FPS at 18k vertices.
- PyQtGraph 0.13 requires `PyOpenGL_accelerate` (C extension) to be
  fast; building it from source needs MS Build Tools, which we don't
  have.

The project is stuck below 30 FPS, which is unacceptable for tactile
feedback work.

## 2. Goals & Non-Goals

**Goals**
- 30+ FPS on the user's integrated GPU.
- Cross-platform: Windows, macOS, Linux, plus any laptop with a
  Chrome / Edge / Firefox.
- A native GL pipeline (no Python GL bindings) so we don't depend
  on `PyOpenGL` or Cython acceleration.
- Vivid jet colormap on both 3D mesh and 2D depth map, sharing one
  normalization range.
- 2D top-view depth map + four live readings + preset view buttons
  in a sidebar.
- Mouse drag rotate + scroll zoom on the 3D view.
- A replay mode so we can validate the visualization without a
  physical sensor.

**Non-Goals (this iteration)**
- Saving snapshots or recordings.
- Multi-sensor display.
- Logging or analytics.
- Authentication / access control.

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (Three.js via CDN, no build step)              │
│                                                         │
│  index.html                                             │
│  ├── 3D canvas (WebGL)                                  │
│  │   └── THREE.Mesh: PlaneGeometry + jet vertex colors  │
│  ├── 2D depth-map canvas (top view)                     │
│  └── sidebar (HTML/CSS)                                 │
│       ├── 2D map                                        │
│       ├── readings (max depth / area / center / FPS)    │
│       └── 4 preset view buttons                         │
│                                                         │
│  WebSocket client (binary, ~25 KB/frame @ 30 FPS)        │
└────────────────────────────┬────────────────────────────┘
                             │ ws://localhost:8765
┌────────────────────────────▼────────────────────────────┐
│  Python                                                  │
│  _3_Shape_Reconstruction.py runs Sensor + viewer.        │
│  shape_reconstruction/visualizer_ws.py:                  │
│    - asyncio WebSocket server                           │
│    - receives {"op": "playback" | "live"}               │
│    - sends binary message per frame:                    │
│        [magic 4B=0x395D534D]                            │
│        [height uint16 HxW little-endian]                │
│        [readings: max_depth f32, area f32, cx f32,      │
│         cy f32, fps f32]                                │
│        [depth_max f32]                                  │
│        [shape: H u16, W u16, pixel_per_mm f32]          │
└─────────────────────────────────────────────────────────┘
```

## 4. Components

### 4.1 Python side

`shape_reconstruction/visualizer_ws.py` exports:
- `WebSocketVisualizer(sensor, cfg)` — a thin object that starts
  the asyncio WebSocket server in a background thread and dispatches
  each `height_map` as a binary frame.
- `pause()`, `resume()`, `close()`.
- `start_browser(url)` — opens the default browser to the viewer
  URL.

Pure functions (`compute_readings`, `jet_vertex_colors`,
`jet_uint8_image`, `require_gui`) continue to live in
`shape_reconstruction/visualizer.py` and are reused by tests.

### 4.2 Browser side

`web_viewer/index.html`:
- Three.js r160 via `https://cdn.jsdelivr.net/three@0.160.0/...`.
- WebGL renderer with antialias + logarithmic depth buffer.
- A `THREE.PlaneGeometry` with `nx × ny` segments; vertex z is
  updated in place each frame.
- Vertex colors computed from jet colormap (same formula as
  Python's `jet_vertex_colors`).
- `OrbitControls` for mouse drag rotate + scroll zoom.
- A `CanvasTexture` for the 2D depth map (top view) updated via
  `tex.image.data.set(...)` and `tex.needsUpdate = true`.
- A sidebar with the same readings and 4 preset view buttons.

### 4.3 Wire format

```
[Magic uint32 LE = 0x395D534D]
[frame_index uint32 LE]
[timestamp_ms float64 LE]
[height_map uint16 LE (H x W pixels, depth in mm * 100)]
[readings 7 x float32 LE: max_depth, area, cx, cy, fps, depth_max, _]
[H uint16 LE]
[W uint16 LE]
[pixel_per_mm float32 LE]
```

Total per frame: ~25 KB at 200×100. At 30 FPS that's 750 KB/s,
well within WebSocket throughput.

## 5. Data flow

1. Camera capture (worker thread).
2. `Sensor.raw_image_2_height_map` → `height_map`.
3. `Sensor.expand_image` → `height_map_expanded`.
4. `visualizer_ws.update(height_map, readings)` packages and
   enqueues the frame.
5. The WebSocket server thread dequeues and sends the binary frame
   to the browser.
6. The browser decodes the frame, updates `mesh.geometry.attributes.position.array[z]`,
   and updates the depth-map texture.
7. The browser repaints.

## 6. Error handling & edge cases

- **No browser connection**: The server keeps running; the latest
  frame is overwritten (no buffering).
- **Sensor disconnected**: server stays up; the client shows the
  last frame and a "No signal" banner.
- **WebSocket disconnect**: the browser auto-reconnects.
- **Frame > 1 Mbit**: skip the frame (drop rate limiting).
- **Numerical issues**: depth values are clamped to uint16 with
  `(depth * 100).astype(np.uint16)`.

## 7. Testing & verification

UI cannot be automated. We provide:

- `tests/test_web_protocol.py` — round-trip encode/decode the binary
  message.
- A replay mode (``python _3_Shape_Reconstruction.py --replay PATH``)
  that reads BGR frames from disk and feeds them through the same
  pipeline.
- Manual checks: 30+ FPS, no GC stutters, mouse drag smooth,
  sidebar reactive.

## 8. Out of scope (deferred)

- Snapshot / video recording.
- Cross-browser fingerprinting of the viewer.
- Multi-tab sync.

These will get their own design docs when picked up.
