"""Live 3D reconstruction loop driving TactileMeshVisualizer.

The camera read + height map computation runs on a worker thread.
The Open3D GUI runs on the main thread via gui.Application.instance.run().
The two communicate through TactileMeshVisualizer.post_height_map(),
which is thread-safe.
"""
import threading
import time

import cv2
import yaml

from shape_reconstruction import Sensor, TactileMeshVisualizer


def _worker(sensor, visualizer, target_fps: float = 30.0):
    interval = 1.0 / target_fps
    while sensor.cap.isOpened() and not visualizer.is_closed():
        t0 = time.monotonic()
        img = sensor.get_rectify_crop_image()
        if img is None:
            time.sleep(interval)
            continue
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
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
        import open3d
        open3d.visualization.gui.Application.instance.run()
    finally:
        visualizer.request_close()
        t.join(timeout=2.0)