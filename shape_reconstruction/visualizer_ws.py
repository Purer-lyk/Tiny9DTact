"""WebSocket-based visualizer for 9DTact 3D shape reconstruction.

Runs an asyncio WebSocket server in a background thread that pushes
each height_map to the browser-based Three.js viewer. The server
encodes frames as compact binary messages (~25 KB at 200x100) and
relies on WebSocket's ordering to keep the client in sync.

The previous Python-based visualizers (PyQt5 + Open3D/VisPy/PyQtGraph)
all hit GPU bottlenecks on the user's integrated GPU. The Three.js
viewer avoids PyOpenGL entirely and keeps the GL pipeline in the
browser.
"""
from __future__ import annotations

import asyncio
import struct
import threading
import time
import webbrowser
from typing import Optional

import numpy as np

# Lazy: only required when actually used by the live visualizer.
websockets = None


MAGIC = 0x395D534D


# ----- Binary frame encoding --------------------------------------------------

def _encode_frame(
    height_map: np.ndarray,
    readings: dict,
    fps: float,
    depth_max: float,
    frame_index: int,
    status: int = None,
) -> bytes:
    """Serialize a single frame to bytes.

    Frame layout (all little-endian):
        uint32   magic 0x395D534D
        uint32   frame_index
        float64  timestamp_ms
        uint16   H
        uint16   W
        float32  pixel_per_mm
        float32  max_depth
        float32  area_mm2
        float32  cx (or -1 if no contact)
        float32  cy (or -1 if no contact)
        float32  fps
        float32  depth_max
        uint8    status (0 live, 1 camera unplugged, 2 reconnecting)
        uint8    pad (keeps the height array 2-byte aligned)
        uint16[] height_map (H x W, depth_mm * 100)
    """
    h, w = height_map.shape
    # Status can be passed explicitly or carried inside readings;
    # WebSocketVisualizer.update() does the latter.
    if status is None:
        status = int(readings.get("status", 0))
    # depth in mm * 100, clipped to uint16
    h_int = np.clip(height_map * 100.0, 0, 65535).astype(np.uint16)
    cx, cy = readings.get("contact_center_mm", (None, None))
    # Use a sentinel value for "no contact" so we can distinguish from
    # a legitimate (0.0, 0.0) center.
    cx = cx if cx is not None else -1.0
    cy = cy if cy is not None else -1.0
    header = struct.pack(
        "<IIdHHfffffffBx",
        MAGIC,
        frame_index,
        time.time() * 1000.0,
        h, w,
        float(readings.get("pixel_per_mm", 0.1)),
        float(readings["max_depth"]),
        float(readings["contact_area_mm2"]),
        float(cx),
        float(cy),
        float(fps),
        float(depth_max),
        int(status),
    )
    return header + h_int.tobytes()


def _decode_frame(data: bytes) -> dict:
    """Parse a frame encoded by _encode_frame. Used by tests and
    by the browser JS (mirrored there)."""
    if len(data) < 50:
        raise ValueError("Frame too short")
    magic, frame_index, ts_ms, h, w, ppm, max_d, area, cx, cy, fps, dmax, status = \
        struct.unpack("<IIdHHfffffffBx", data[:50])
    if magic != MAGIC:
        raise ValueError(f"Bad magic 0x{magic:08x}")
    h_int = np.frombuffer(data[50:], dtype=np.uint16).reshape(h, w).copy()
    return {
        "frame_index": frame_index,
        "timestamp_ms": ts_ms,
        "H": h,
        "W": w,
        "pixel_per_mm": ppm,
        "max_depth": max_d,
        "contact_area_mm2": area,
        "contact_center_mm": (cx if cx >= 0 else None, cy if cy >= 0 else None),
        "fps": fps,
        "depth_max": dmax,
        "status": status,
        "height_map": h_int.astype(np.float32) / 100.0,
    }


# ----- WebSocket server -------------------------------------------------------

class WebSocketVisualizer:
    """Asyncio WebSocket server that streams height_map frames to a
    browser-based Three.js viewer.

    The server is launched in a background thread so the main thread
    can keep running the OpenCV capture loop. The HTTP "push" model
    uses overwrite-on-back-pressure: if the connected client is
    slow, we drop the oldest queued frame and keep going.
    """

    def __init__(
        self,
        sensor,                       # shape_reconstruction.sensor.Sensor
        cfg: dict,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        global websockets
        if websockets is None:
            try:
                import websockets as _ws  # type: ignore
                websockets = _ws
            except ImportError as exc:
                raise RuntimeError(
                    "WebSocketVisualizer requires the 'websockets' package. "
                    "Install with: pip install websockets==12.0"
                ) from exc

        self._sensor = sensor
        self._cfg = cfg
        self._host = host
        self._port = port
        self._frame_index = 0
        self._last_fps_ts = 0.0
        self._ema_fps = 0.0
        self._close = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._clients: set = set()
        self._pending: Optional[bytes] = None
        self._pending_lock = threading.Lock()

    # ----- public API ---------------------------------------------------------

    def get_url(self) -> str:
        return f"ws://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def open_browser(self, html_path: str) -> None:
        """Open the local HTML viewer in the default browser."""
        url = "file://" + html_path.replace("\\", "/")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def update(
        self,
        height_map: np.ndarray,
        readings: dict,
    ) -> None:
        """Queue a frame for the next available WebSocket send."""
        if self._ema_fps:
            self._ema_fps = 0.9 * self._ema_fps + 0.1 * (1.0 / max(1e-3, time.monotonic() - self._last_fps_ts))
        else:
            self._ema_fps = 30.0
        self._last_fps_ts = time.monotonic()

        msg = _encode_frame(
            height_map,
            readings,
            fps=self._ema_fps,
            depth_max=max(readings["max_depth"], 0.5),
            frame_index=self._frame_index,
            status=int(readings.get("status", 0)),
        )
        self._frame_index += 1

        # Debug: log the first few frames so we can sanity-check the
        # wire encoding end-to-end.
        if self._frame_index <= 3:
            cx, cy = readings.get("contact_center_mm", (None, None))
            print(
                f"[ws] frame {self._frame_index - 1}: "
                f"max_depth={readings['max_depth']:.3f}, "
                f"area={readings['contact_area_mm2']:.2f}, "
                f"center=({cx}, {cy}), "
                f"shape={height_map.shape}, "
                f"fps={self._ema_fps:.1f}",
                flush=True,
            )

        # Overwrite-on-back-pressure: keep only the most recent frame.
        with self._pending_lock:
            self._pending = msg

        # Schedule dispatch on the event loop if it's running.
        if self._loop is not None and self._clients:
            asyncio.run_coroutine_threadsafe(self._flush(), self._loop)

    def close(self) -> None:
        self._close.set()
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._stop_server(), self._loop)

    # ----- internals ----------------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        async def main():
            # The height_map frame is H*W uint16 + ~48 bytes header.
            # A 696x1254 frame is ~1.7 MB which exceeds websockets'
            # default max_size of 1 MiB, so we raise it to 8 MiB.
            self._server = await websockets.serve(
                self._handler, self._host, self._port,
                max_size=8 * 1024 * 1024,
            )
            try:
                await self._server.wait_closed()
            except asyncio.CancelledError:
                pass
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            # After handshake, push the latest frame immediately.
            await self._flush()
            async for _msg in ws:
                # We don't process client messages right now; the
                # server is push-only. Keep the loop alive until the
                # client disconnects.
                pass
        finally:
            self._clients.discard(ws)

    async def _flush(self) -> None:
        if not self._clients:
            return
        with self._pending_lock:
            msg = self._pending
            self._pending = None
        if msg is None:
            return
        # Send to all connected clients; close stragglers on error.
        dead = []
        for ws in list(self._clients):
            try:
                # Back-pressure: if the previous frame has not fully
                # left the transport yet (slow client, e.g. a software
                # -rendered WebView), DROP this frame for that client
                # instead of queueing it. Queueing would otherwise
                # build up unbounded end-to-end delay.
                transport = ws.transport
                if transport is not None and \
                        transport.get_write_buffer_size() > 0:
                    continue
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _stop_server(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


# ----- launch helper ---------------------------------------------------------

def open_viewer() -> None:
    """Open the bundled web_viewer/index.html in the user's default browser."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    html = os.path.normpath(os.path.join(here, "..", "web_viewer", "index.html"))
    if not os.path.exists(html):
        raise FileNotFoundError(f"Web viewer not found at {html}")
    webbrowser.open("file://" + html.replace("\\", "/"))


__all__ = [
    "WebSocketVisualizer",
    "open_viewer",
    "_encode_frame",
    "_decode_frame",
]
