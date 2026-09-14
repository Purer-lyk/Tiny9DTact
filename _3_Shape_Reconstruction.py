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
from shape_reconstruction.visualizer import compute_readings


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


def _recover_connection(sensor, vws, stop, poll_s=1.0):
    """Handle an unplugged camera: keep the viewer blanked, wait for
    the device to be re-attached, then recapture the reference image
    (auto-exposure needs to settle again) and resume streaming.

    Returns False only when the stop event fires (window closed)."""
    print("[capture] camera lost - blanking viewer, waiting for replug ...")
    _push_status_frame(sensor, vws, status=1)
    sensor.cap.release()
    last_blank = time.monotonic()
    while stop is None or not stop.is_set():
        if sensor.open_capture():
            print("[capture] camera re-opened; capturing fresh reference ...")
            _push_status_frame(sensor, vws, status=2)
            try:
                sensor.ref = sensor.get_stable_rectify_crop_avg_image()
                sensor.ref_GRAY = cv2.cvtColor(sensor.ref, cv2.COLOR_BGR2GRAY)
                print("[capture] reference updated - resuming live stream")
                return True
            except (RuntimeError, cv2.error) as exc:
                print(f"[capture] reference capture failed ({exc}); retrying ...")
                sensor.cap.release()
        # Re-push the blank frame occasionally so a viewer that
        # (re)connects mid-outage also sees the disconnected state.
        if time.monotonic() - last_blank > 2.0:
            _push_status_frame(sensor, vws, status=1)
            last_blank = time.monotonic()
        time.sleep(poll_s)
    return False


def _drive_live(sensor, vws, target_fps: float, stop: threading.Event = None) -> None:
    interval = 1.0 / target_fps
    none_streak = 0
    while sensor.cap.isOpened():
        if stop is not None and stop.is_set():
            break
        t0 = time.monotonic()
        try:
            img = sensor.get_rectify_crop_image()
        except (TypeError, IndexError, cv2.error):
            # cap.read() returned no frame; get_raw_image() then feeds
            # None into the rectify index and raises before returning.
            img = None
        if img is None:
            # An occasional dropped frame is normal; a sustained streak
            # means the camera was unplugged -> blank + wait for replug.
            none_streak += 1
            if none_streak >= 10:
                if not _recover_connection(sensor, vws, stop):
                    break
                none_streak = 0
                continue
            time.sleep(interval)
            continue
        none_streak = 0
        _process_and_push(sensor, vws, img)
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


def _viewer_html_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    html = os.path.normpath(os.path.join(here, "web_viewer", "index.html"))
    if not os.path.exists(html):
        raise FileNotFoundError(f"Web viewer not found at {html}")
    return html


def _run_desktop(vws, sensor, args) -> None:
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
        try:
            if args.replay:
                _drive_replay(sensor, vws, args.replay, fps=args.fps, stop=stop)
            else:
                _drive_live(sensor, vws, target_fps=float(args.fps), stop=stop)
        except Exception as exc:
            print(f"[capture] loop stopped: {exc}")
            # Make sure the window closes if the capture dies.
            try:
                for w in webview.windows:
                    w.destroy()
            except Exception:
                pass

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
            _drive_live(sensor, vws, target_fps=float(args.fps))
        return
    # Window closed: stop the capture loop.
    stop.set()
    t.join(timeout=2.0)


def _run_browser(vws, sensor, args) -> None:
    """Classic behaviour: open the system browser, loop on main thread."""
    from shape_reconstruction.visualizer_ws import open_viewer
    open_viewer()
    if args.replay:
        _drive_replay(sensor, vws, args.replay, fps=args.fps)
    else:
        _drive_live(sensor, vws, target_fps=float(args.fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="9DTact Web Viewer")
    parser.add_argument("--replay", metavar="DIR", help="Replay frame_*.png images")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--browser", action="store_true",
                        help="Open in the system browser instead of a desktop window")
    args = parser.parse_args()

    with open("./shape_reconstruction/shape_config.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    sensor = Sensor(cfg)

    from shape_reconstruction.visualizer_ws import WebSocketVisualizer
    vws = WebSocketVisualizer(sensor, cfg)
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
            _run_desktop(vws, sensor, args)
        else:
            _run_browser(vws, sensor, args)
    except KeyboardInterrupt:
        pass
    finally:
        vws.close()
        try:
            sensor.cap.release()
        except Exception:
            pass
