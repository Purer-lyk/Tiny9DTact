"""Live 3D reconstruction loop driving the WebSocket viewer.

Architecture:
  - Sensor (OpenCV capture + height-map compute) feeds the WebSocket
    server (visualizer_ws.py), which pushes binary height_map frames
    to the Three.js client.
  - Desktop mode (default): the bundled web_viewer/index.html is
    shown inside a native window via pywebview (Edge WebView2 on
    Windows — Chromium engine, full WebGL support). webview.start()
    runs on the main thread; the capture loop runs in a worker
    thread. Closing the window stops the program.
  - Browser mode (--browser): opens the system browser instead and
    runs the capture loop on the main thread (the old behaviour).

The WebSocket viewer is the recommended backend because it leaves
OpenGL handling to the embedded browser engine, sidestepping the
PyOpenGL / VisPy 0.14 / PyQtGraph 0.13 GPU bottleneck we hit on
integrated GPUs.
"""
import argparse
import os
import queue
import sys
import threading
import time

# Mute the MSMF warning storm OpenCV prints while the USB camera is
# unplugged (cap_msmf.cpp OnReadSample error spam). Must be set
# before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import yaml

from shape_reconstruction import Sensor
from shape_reconstruction.camera import probe_capture_channels
from shape_reconstruction.cammatch import is_same_reference
from shape_reconstruction.visualizer import compute_readings

# How many capture device indices to scan when looking for the sensor
# again (channel ordering on Windows/MSMF can change after unplug, and
# laptops with a built-in webcam may map the sensor to any index).
MAX_SCAN_CHANNELS = 6

# Placeholder frame used for status pushes while no sensor exists yet
# (viewer opens before the camera does). Any small even shape works;
# the real frame size takes over as soon as the sensor streams.
_BLANK_SHAPE = (8, 12)  # (H, W)


class CameraControl:
    """Thread-safe command channel from the viewer UI to the capture
    loop ('select camera N' / 'force reconnect'), plus the currently
    active channel reported back for the UI."""

    def __init__(self) -> None:
        self._queue: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self.active_channel = None

    def request_select(self, channel: int) -> None:
        self._queue.put(("select", int(channel)))

    def request_reconnect(self) -> None:
        self._queue.put(("reconnect", None))

    def poll(self, timeout: float = 0.0):
        """Return the most recent command (older ones are dropped), or
        None when the queue is empty. Never blocks longer than
        `timeout` -- also usable as a poll-with-sleep."""
        try:
            cmd = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                return cmd

    def set_active(self, channel) -> None:
        with self._lock:
            self.active_channel = channel


def _process_and_push(sensor, vws, img):
    """Full pipeline for one BGR frame: height map -> readings -> ws."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = sensor.raw_image_2_height_map(img_gray)
    h = sensor.expand_image(h)
    readings = compute_readings(
        h, pixel_per_mm=float(sensor.pixel_per_mm), contact_threshold=0.05
    )
    vws.update(
        h,
        {
            "max_depth": readings.max_depth,
            "contact_area_mm2": readings.contact_area_mm2,
            "contact_center_mm": readings.contact_center_mm,
            "pixel_per_mm": float(sensor.pixel_per_mm),
        },
    )


def _push_status_frame(sensor, vws, status):
    """Blank the viewer: a zero height map tagged with a status code
    (1 = unplugged, 2 = reconnecting). The page blacks out the 2D/3D
    views and shows the state instead of the jet colormap."""
    blank = np.zeros(sensor.ref_GRAY.shape[:2], dtype=np.float32)
    vws.update(
        blank,
        {
            "max_depth": 0.0,
            "contact_area_mm2": 0.0,
            "contact_center_mm": (None, None),
            "pixel_per_mm": float(sensor.pixel_per_mm),
            "status": status,
        },
    )


def _push_blank_status(vws, status):
    """Status push for the pre-sensor phase (window open, camera not
    attached yet): there is no reference frame to size the blank from,
    so a tiny placeholder shape is used."""
    vws.update(
        np.zeros(_BLANK_SHAPE, dtype=np.float32),
        {
            "max_depth": 0.0,
            "contact_area_mm2": 0.0,
            "contact_center_mm": (None, None),
            "pixel_per_mm": 0.1,
            "status": status,
        },
    )


def _acquire_sensor(cfg, vws, stop, control):
    """Construct the Sensor (open camera + capture reference) while
    keeping the viewer alive: the window opens BEFORE any camera is
    required. Pushes status=1/2 frames so the UI shows UNPLUGGED /
    RECONNECTING, retries every second until the device yields frames,
    and honours a viewer-side channel override (session-only: cfg is
    mutated in memory, never written back to shape_config.yaml).

    Returns None once `stop` fires (window closed)."""
    print("[startup] waiting for the sensor camera ...")
    _push_blank_status(vws, status=1)
    last_blank = time.monotonic()
    while stop is None or not stop.is_set():
        if control is not None:
            cmd = control.poll(timeout=1.0)
            if cmd is not None and cmd[0] == "select":
                cfg["camera_setting"]["camera_channel"] = cmd[1]
                print(f"[startup] camera channel set to {cmd[1]} "
                      f"(viewer request; applies to this run only)")
        _push_blank_status(vws, status=2)  # opening / capturing ref
        try:
            sensor = Sensor(cfg)
            if control is not None:
                control.set_active(sensor._camera_channel)
            print("[startup] sensor ready")
            return sensor
        except RuntimeError as exc:
            # Camera absent or still initialising -- keep waiting.
            if time.monotonic() - last_blank > 2.0:
                _push_blank_status(vws, status=1)
                last_blank = time.monotonic()
            print(f"[startup] camera not ready ({exc}); retrying ...")
    return None


def _candidate_channels(preferred: int):
    """Scan order: the last-working channel first (the common replug
    case), then every other index (laptops re-enumerate devices)."""
    return [preferred] + [i for i in range(MAX_SCAN_CHANNELS) if i != preferred]


def _recover_connection(sensor, vws, stop, control,
                        requested=None, poll_s=1.0, verify_identity=True):
    """Handle a lost/switched camera: blank the viewer, then find the
    *same physical device* again before resuming.

    On multi-camera machines (laptops with a built-in webcam) simply
    reopening the configured channel often grabs the WRONG camera --
    it opens fine and then streams an unrelated view forever. So every
    candidate channel is quick-probed and compared structurally
    (cammatch NCC) against the last known-good reference; only the
    matching device is accepted. A viewer-side manual selection
    (control queue: ('select', N)) bypasses the identity check -- the
    user's explicit choice wins, e.g. when hooking up a new sensor.
    `verify_identity=False` is the first-acquisition mode (program
    started before the camera was attached): only the configured
    channel is accepted, since no trusted anchor exists to compare
    against.

    Every per-channel attempt is fully guarded: a device that vanishes
    mid-probe or throws MSMF errors is skipped, never fatal.

    Returns False only when the stop event fires (window closed)."""
    print("[capture] waiting for a usable camera - blanking viewer ...")
    prev_ref_GRAY = sensor.ref_GRAY.copy()  # identity anchor
    _push_status_frame(sensor, vws, status=1)
    try:
        sensor.cap.release()
    except Exception:
        pass
    last_blank = time.monotonic()
    while stop is None or not stop.is_set():
        manual_channel = None
        if requested is not None and requested[0] == "select":
            manual_channel = requested[1]
        if manual_channel is not None:
            candidates = [manual_channel]
        elif not verify_identity:
            # First acquisition: no trusted anchor exists yet, so only
            # the configured channel (default 0) is accepted -- the
            # viewer dropdown picks anything else. Keeps "default
            # connects 0" predictable instead of grabbing whichever
            # camera answers first (e.g. a built-in webcam).
            candidates = [sensor._camera_channel]
        else:
            candidates = _candidate_channels(sensor._camera_channel)
        for ch in candidates:
            if stop is not None and stop.is_set():
                return False
            try:
                if not sensor.open_capture(ch):
                    continue
                if manual_channel is not None:
                    print(f"[capture] switching to camera channel {ch} (user request)")
                elif not verify_identity:
                    print(f"[capture] acquiring first reference from channel {ch} ...")
                else:
                    probe = sensor.quick_reference_probe()
                    matched, score = is_same_reference(prev_ref_GRAY, probe)
                    if not matched:
                        print(f"[capture] channel {ch}: different camera "
                              f"(ncc={score:.2f}); skipping")
                        sensor.cap.release()
                        continue
                    print(f"[capture] channel {ch}: sensor identified "
                          f"(ncc={score:.2f})")
                _push_status_frame(sensor, vws, status=2)
                sensor.ref = sensor.get_stable_rectify_crop_avg_image()
                sensor.ref_GRAY = cv2.cvtColor(sensor.ref, cv2.COLOR_BGR2GRAY)
                if control is not None:
                    control.set_active(sensor._camera_channel)
                print("[capture] reference updated - resuming live stream")
                return True
            except Exception as exc:
                # Device vanished mid-probe, MSMF tear-down error, bad
                # frame from a foreign camera ... skip to the next one.
                print(f"[capture] channel {ch}: {type(exc).__name__}: "
                      f"{exc}; skipping")
                try:
                    sensor.cap.release()
                except Exception:
                    pass
        # Nothing accepted this round (device still unplugged, or a
        # manual pick that would not open). Fall back to the automatic
        # scan and keep the viewer blanked; late-joining clients also
        # see the state via the periodic status frame.
        requested = None
        if time.monotonic() - last_blank > 2.0:
            _push_status_frame(sensor, vws, status=1)
            last_blank = time.monotonic()
        # Poll doubles as the wait; a UI command takes effect at once.
        cmd = control.poll(timeout=poll_s) if control is not None else None
        if cmd is not None:
            requested = cmd
    return False


def _drive_live(sensor, vws, target_fps: float, stop: threading.Event = None,
                control: CameraControl = None) -> None:
    interval = 1.0 / target_fps
    none_streak = 0
    err_streak = 0
    # NOTE: not `while cap.isOpened()` -- MSMF flips that to False on
    # unplug, which used to end the loop silently (and any escaping
    # exception used to kill the window). The streak counters below
    # own all dead-device handling instead.
    while stop is None or not stop.is_set():
        # First run without a camera attached: the sensor was built
        # with a placeholder reference (see _make_sensor). Acquire the
        # real reference from the configured channel before streaming;
        # the dropdown stays available to pick a different device.
        if getattr(sensor, "ref_pending", False):
            if not _recover_connection(sensor, vws, stop, control,
                                       verify_identity=False):
                break
            sensor.ref_pending = False
            none_streak = 0
            err_streak = 0
            continue
        # Viewer-side camera commands (manual switch / force
        # reconnect) are honoured mid-stream.
        if control is not None:
            cmd = control.poll()
            if cmd is not None:
                if not _recover_connection(sensor, vws, stop, control,
                                           requested=cmd):
                    break
                none_streak = 0
                err_streak = 0
                continue
        t0 = time.monotonic()
        try:
            img = sensor.get_rectify_crop_image()
        except Exception:
            # cap.read() returned no frame (None flows into the rectify
            # index and raises) or MSMF raised while the device was
            # being torn down -- either way this frame is simply gone.
            img = None
        if img is None:
            # An occasional dropped frame is normal; a sustained streak
            # means the camera was unplugged -> blank + wait for replug.
            none_streak += 1
            if none_streak >= 10:
                if not _recover_connection(sensor, vws, stop, control):
                    break
                none_streak = 0
                err_streak = 0
                continue
            time.sleep(interval)
            continue
        none_streak = 0
        try:
            _process_and_push(sensor, vws, img)
        except Exception as exc:
            # Processing hiccup (bad frame, encode issue): drop the
            # frame; a sustained streak is treated like a camera loss.
            err_streak += 1
            if err_streak == 1:
                print(f"[capture] frame error: {type(exc).__name__}: {exc}")
            if err_streak >= 10:
                if not _recover_connection(sensor, vws, stop, control):
                    break
                none_streak = 0
                err_streak = 0
                continue
        else:
            err_streak = 0
        cv2.waitKey(1)
        dt = time.monotonic() - t0
        time.sleep(max(0.0, interval - dt))


def _drive_replay(sensor, vws, frames_dir: str, fps: int,
                  stop: threading.Event = None) -> None:
    """Replay a directory of frame_*.png BGR images through the
    same pipeline. Used for development without a physical sensor."""
    import glob
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {frames_dir!r}")
    interval = 1.0 / fps
    for path in frames:
        if stop is not None and stop.is_set():
            break
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        _process_and_push(sensor, vws, bgr)
        time.sleep(interval)


def _base_dir() -> str:
    """Directory holding the user-editable data folders.

    In a PyInstaller build the Python runtime lives inside _internal/,
    while shape_config.yaml / calibration / web_viewer are shipped NEXT
    TO the exe (see build_release.bat) so they stay editable without
    rebuilding. In a source checkout the data sits next to this file.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _viewer_html_path() -> str:
    html = os.path.normpath(
        os.path.join(_base_dir(), "web_viewer", "index.html"))
    if not os.path.exists(html):
        raise FileNotFoundError(f"Web viewer not found at {html}")
    return html


def _camera_inventory(sensor, max_channels: int = MAX_SCAN_CHANNELS) -> dict:
    """List currently openable capture channels for the viewer UI.

    The actively held channel is excluded from the probe (opening it
    twice would compete for the device) and reported as `active`.
    Runs in a worker thread when invoked from the WebSocket server.
    """
    active = sensor._camera_channel
    try:
        others = sensor.probe_channels(max_channels=max_channels,
                                       exclude={active})
    except Exception:
        others = []
    return {"active": active, "list": sorted({active} | set(others))}


def _make_sensor(cfg):
    """Build the Sensor, tolerating a not-yet-attached camera.

    Normally Sensor() captures the reference image during __init__.
    With no camera attached that raises RuntimeError after a short
    grace period; in that case rebuild it with a black placeholder
    reference and flag `ref_pending`, so the viewer window opens
    right away and the live loop acquires the real reference once the
    device shows up (configured channel first, dropdown to override).
    """
    try:
        return Sensor(cfg)
    except RuntimeError as exc:
        print(f"[startup] no camera yet ({exc}); "
              f"opening the viewer and waiting for the device ...")
        crop = cfg["camera_calibration"]["crop_size"]
        sensor = Sensor(cfg, ref=np.zeros((crop[0], crop[1], 3), np.uint8))
        sensor.ref_pending = True
        return sensor


def _run_desktop(vws, sensor, args, control: CameraControl = None) -> None:
    """Native window via pywebview; blocks until the window closes."""
    import webview
    from shape_reconstruction.visualizer_ws import open_viewer

    # WebView2 performance flags. Without these the embedded engine
    # can fall back to software WebGL (SwiftShader) on integrated-GPU
    # machines, which adds visible latency. Must be set BEFORE the
    # WebView2 environment is created.
    os.environ.setdefault(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--ignore-gpu-blocklist "
        "--enable-gpu-rasterization "
        "--disable-background-timer-throttling "
        "--disable-features=msWebView2BrowserProcessEfficiencyMode",
    )

    stop = threading.Event()

    def worker():
        if args.replay:
            try:
                _drive_replay(sensor, vws, args.replay, fps=args.fps, stop=stop)
            except Exception as exc:
                print(f"[capture] replay stopped: {exc}")
            return
        # Live capture: guarded restart loop. A camera unplug (or any
        # unexpected error) must NEVER take the window down -- the
        # viewer blanks and the loop restarts after a backoff. The
        # window closes only when the user closes it (stop event).
        backoff = 1.0
        while not stop.is_set():
            try:
                _drive_live(sensor, vws, target_fps=float(args.fps),
                            stop=stop, control=control)
                break  # a normal return means stop was requested
            except Exception as exc:
                import traceback
                print(f"[capture] unexpected error: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                try:
                    _push_status_frame(sensor, vws, status=1)
                except Exception:
                    pass
                if stop.wait(backoff):
                    break
                backoff = min(backoff * 2.0, 10.0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    webview.create_window(
        "9DTact Shape Reconstruction",
        _viewer_html_path(),
        width=1280,
        height=820,
        min_size=(980, 620),
    )
    # private_mode=False keeps a persistent profile so the three.js
    # CDN download is cached across runs (much faster startup).
    # Set 9DTACT_DEBUG=1 to open with DevTools enabled.
    debug = os.environ.get("9DTACT_DEBUG") == "1"
    try:
        webview.start(gui="edgechromium", private_mode=False, debug=debug)
    except Exception as exc:
        print(f"Desktop window failed to start ({exc}).")
        print("Install the Microsoft WebView2 Runtime if this persists.")
        print("Falling back to the system browser ...")
        stop.set()
        t.join(timeout=2.0)
        open_viewer()
        if args.replay:
            _drive_replay(sensor, vws, args.replay, fps=args.fps)
        else:
            _drive_live(sensor, vws, target_fps=float(args.fps), control=control)
        return
    # Window closed: stop the capture loop.
    stop.set()
    t.join(timeout=2.0)


def _run_browser(vws, sensor, args, control: CameraControl = None) -> None:
    """Classic behaviour: open the system browser, loop on main thread."""
    from shape_reconstruction.visualizer_ws import open_viewer
    open_viewer()
    if args.replay:
        _drive_replay(sensor, vws, args.replay, fps=args.fps)
    else:
        _drive_live(sensor, vws, target_fps=float(args.fps), control=control)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="9DTact Web Viewer")
    parser.add_argument("--replay", metavar="DIR", help="Replay frame_*.png images")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--browser", action="store_true",
                        help="Open in the system browser instead of a desktop window")
    args = parser.parse_args()

    # PyInstaller build: anchor the working directory at the exe folder
    # so the relative data paths (shape_reconstruction/, web_viewer/)
    # resolve no matter where the exe is launched from. A relative
    # --replay path is resolved against the ORIGINAL cwd first.
    replay_abs = args.replay
    if getattr(sys, "frozen", False):
        if replay_abs and not os.path.isabs(replay_abs):
            replay_abs = os.path.abspath(replay_abs)
        os.chdir(_base_dir())
        args.replay = replay_abs

    with open("./shape_reconstruction/shape_config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Default: connect the configured channel (camera_channel: 0).
    # The window opens even when the camera is not attached yet -- it
    # shows the UNPLUGGED state and picks the device up once it
    # appears. Config/calibration errors still fail loudly.
    sensor = _make_sensor(cfg)
    if getattr(sensor, "ref_pending", False) and args.replay:
        raise SystemExit(
            "[replay] no camera attached: replay still needs a live "
            "reference image from the sensor camera.")

    from shape_reconstruction.visualizer_ws import WebSocketVisualizer
    vws = WebSocketVisualizer(sensor, cfg)
    control = CameraControl()
    control.set_active(sensor._camera_channel)
    vws.set_camera_control(control)
    vws.set_camera_inventory_fn(lambda: _camera_inventory(sensor))
    vws.start()
    print("WebSocket viewer listening on", vws.get_url())

    use_desktop = not args.browser
    if use_desktop:
        try:
            import webview  # noqa: F401
        except ImportError:
            print("pywebview is not installed; falling back to the browser.")
            print("Install it with: pip install pywebview")
            use_desktop = False

    try:
        if use_desktop:
            _run_desktop(vws, sensor, args, control=control)
        else:
            _run_browser(vws, sensor, args, control=control)
    except KeyboardInterrupt:
        pass
    finally:
        vws.close()
        try:
            sensor.cap.release()
        except Exception:
            pass
