import time

import numpy as np
import cv2
import yaml


def probe_capture_channels(max_channels=6, exclude=()):
    """Return the indices of currently openable capture devices.

    Module-level so the viewer's camera picker can enumerate devices
    even before a Sensor exists. Opens each candidate briefly with a
    throw-away VideoCapture; never raises.
    """
    found = []
    for i in range(int(max_channels)):
        if i in exclude:
            continue
        try:
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened()
            cap.release()
        except Exception:
            ok = False
        if ok:
            found.append(i)
    return found


class Camera:
    def __init__(self, cfg, calibrated=True):
        sensor_id = cfg['sensor_id']
        camera_setting = cfg['camera_setting']
        camera_channel = camera_setting['camera_channel']
        self.raw_img_width = camera_setting['resolution'][0]
        self.raw_img_height = camera_setting['resolution'][1]
        fps = camera_setting['fps']
        self._camera_channel = camera_channel
        self._camera_fps = fps
        self.cap = cv2.VideoCapture()  # placeholder; opened below
        # Default: the configured channel only (shape_config.yaml
        # camera_channel: 0). No fallback scan here -- on a laptop that
        # would happily open the built-in webcam instead. When the
        # device is not present yet, the caller retries until it
        # appears (see _3_Shape_Reconstruction.py).
        if self.open_capture():
            print('------Camera is open--------')
        else:
            print('------Camera failed to open--------')

        calibration_root_dir = cfg['calibration_root_dir']
        self.calibration_sensor_dir = calibration_root_dir + '/sensor_' + str(sensor_id)
        camera_calibration = cfg['camera_calibration']
        self.camera_calibration_dir = self.calibration_sensor_dir + camera_calibration['camera_calibration_dir']
        self.row_index_path = self.camera_calibration_dir + camera_calibration['row_index_path']
        self.col_index_path = self.camera_calibration_dir + camera_calibration['col_index_path']
        self.position_scale_path = self.camera_calibration_dir + camera_calibration['position_scale_path']

        self.crop_img_height = camera_calibration['crop_size'][0]
        self.crop_img_width = camera_calibration['crop_size'][1]

        if calibrated:
            self.row_index = np.load(self.row_index_path)
            self.col_index = np.load(self.col_index_path)
            position_scale = np.load(self.position_scale_path)
            center_position = position_scale[0:2]
            self.pixel_per_mm = position_scale[2]
            self.row_points = camera_calibration['row_points']
            self.col_points = camera_calibration['col_points']
            # self.height_begin = int(center_position[0] - self.crop_img_height / 2)
            # self.height_end = int(center_position[0] + self.crop_img_height / 2)
            # self.width_begin = int(center_position[1] - self.crop_img_width / 2)
            # self.width_end = int(center_position[1] + self.crop_img_width / 2)

            self.height_begin = int(center_position[0] - self.crop_img_height / 2) if self.row_points % 2 == 1 else \
                        int(center_position[0] - self.crop_img_height / (self.row_points-1) * (self.row_points//2))
            self.height_end = int(center_position[0] + self.crop_img_height / 2) if self.row_points % 2 == 1 else \
                        int(center_position[0] + self.crop_img_height / (self.row_points-1) * (self.row_points//2 - 1))
            self.width_begin = int(center_position[1] - self.crop_img_width / 2) if self.col_points % 2 == 1 else \
                        int(center_position[1] - self.crop_img_width / (self.col_points-1) * (self.col_points//2))
            self.width_end = int(center_position[1] + self.crop_img_width / 2) if self.col_points % 2 == 1 else \
                        int(center_position[1] + self.crop_img_width / (self.col_points-1) * (self.col_points//2 - 1))

    def open_capture(self, channel=None):
        """(Re)open the capture device with the configured settings.

        Used once at construction time and again to recover after the
        USB camera is unplugged and re-attached. `channel` overrides
        the configured camera_channel (used when scanning for the
        right device on multi-camera machines). Returns True when the
        device opened successfully. Never raises: MSMF can throw
        cv2.error while a device is being torn down, and callers
        (especially the recovery loop) rely on a clean False.
        """
        if channel is not None:
            self._camera_channel = int(channel)
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cap = cv2.VideoCapture(self._camera_channel)
        except Exception:
            cap = None
        self.cap = cap if cap is not None else cv2.VideoCapture()
        if not self.cap.isOpened():
            return False
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, self.raw_img_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.raw_img_height),
            (cv2.CAP_PROP_FPS, self._camera_fps),
            (cv2.CAP_PROP_EXPOSURE, -7),
        ):
            try:
                self.cap.set(prop, value)
            except Exception:
                pass  # unsupported property on this device/backend
        return True

    def probe_channels(self, max_channels=6, exclude=()):
        """Return the indices of currently openable capture devices.

        Thin wrapper around probe_capture_channels(); pass the index
        currently held by self.cap in `exclude` to avoid competing for
        it. Never raises.
        """
        return probe_capture_channels(max_channels=max_channels, exclude=exclude)

    def quick_reference_probe(self, skip_frames=8, avg_frames=4, max_misses=4):
        """Fast gray probe frame for camera identity verification.

        Unlike get_stable_rectify_crop_avg_image() there is no
        stability wait -- we only need *a* frame good enough to tell
        cameras apart (cammatch compares structure, not exact pixels).
        Raises RuntimeError when the device yields no usable frames.
        """
        got = []
        skipped = 0
        misses = 0
        while len(got) < avg_frames:
            try:
                img = self.get_rectify_crop_image()
            except (TypeError, IndexError, cv2.error):
                # cap.read() returned nothing (unplugged mid-probe or
                # the device is held elsewhere). Occasional drops are
                # tolerated; a sustained streak aborts the probe.
                misses += 1
                if misses > max_misses:
                    raise RuntimeError('device yields no frames')
                time.sleep(0.02)
                continue
            misses = 0
            if skipped < skip_frames:
                skipped += 1  # discard auto-exposure warm-up frames
                continue
            got.append(img.astype(np.float64))
        avg = np.mean(got, axis=0).astype(np.uint8)
        return cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)

    def get_raw_image(self):
        src = self.cap.read()[1]
        return src
        # return src

    def rectify_image(self, img):
        img_rectify = img[self.row_index, self.col_index]
        return img_rectify

    def crop_image(self, img):
        return img[self.height_begin:self.height_end, self.width_begin:self.width_end]

    def rectify_crop_image(self, img):
        img = self.crop_image(self.rectify_image(img))
        return img

    def get_rectify_image(self):
        img = self.rectify_image(self.get_raw_image())
        return img

    def get_rectify_crop_image(self):
        img = self.crop_image(self.get_rectify_image())
        return img

    def get_raw_avg_image(self):
        global img
        while True:
            img = self.cap.read()[1]
            img = img
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.cap.read()[1]
            raw_image = raw_image
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_rectify_avg_image(self):
        global img
        while True:
            img = self.get_rectify_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_image()
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def get_stable_rectify_crop_avg_image(
        self,
        skip_frames=10,
        stable_frames=15,
        diff_threshold=2.0,
        avg_frames=10,
        timeout_s=15.0,
    ):
        """Automatically capture the reference image once the video
        has stabilised (auto-exposure settled, no residual motion).

        Streams rectified+cropped frames and measures the mean
        absolute difference between consecutive downsampled gray
        frames. The first `skip_frames` frames are always discarded
        (camera warm-up). Once the difference stays below
        `diff_threshold` for `stable_frames` consecutive frames, the
        following `avg_frames` frames are averaged into the
        reference. If the image never stabilises within `timeout_s`
        seconds, a plain average is captured anyway so the program
        never blocks forever.
        """
        print('Waiting for the image to stabilise ...')
        t0 = time.monotonic()
        prev_small = None
        stable_count = 0
        n_read = 0
        fail_streak = 0
        img = None
        while True:
            try:
                img = self.get_rectify_crop_image()
            except (TypeError, IndexError, cv2.error):
                # cap.read() returned nothing (unplugged mid-capture) or
                # MSMF raised while tearing the device down. An
                # occasional dropped frame is normal; a sustained streak
                # means the camera is gone -> abort instead of spinning
                # forever (the `continue` used to skip the timeout
                # check below).
                fail_streak += 1
                if fail_streak > 150:
                    raise RuntimeError('No frames from the camera '
                                       '(unplugged or occupied by another app)')
                time.sleep(0.01)
                continue
            fail_streak = 0
            n_read += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(
                gray, (0, 0), fx=0.25, fy=0.25,
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            if prev_small is not None and n_read > skip_frames:
                diff = float(np.mean(np.abs(small - prev_small)))
                if diff < diff_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= stable_frames:
                    print(
                        f'Image stable after {n_read} frames '
                        f'({time.monotonic() - t0:.1f} s); '
                        f'capturing reference.'
                    )
                    break
            prev_small = small
            if time.monotonic() - t0 > timeout_s:
                print('Stability timeout reached; capturing anyway.')
                break
        img_add = np.zeros(img.shape, np.float64)
        n_used = 0
        for _ in range(avg_frames):
            try:
                f = self.get_rectify_crop_image()
            except (TypeError, IndexError, cv2.error):
                time.sleep(0.01)
                continue
            img_add += f
            n_used += 1
        if n_used == 0:
            raise RuntimeError('Could not read frames for the reference image')
        img_avg = (img_add / n_used).astype(np.uint8)
        return img_avg

    def get_rectify_crop_avg_image(self):
        global img
        while True:
            img = self.get_rectify_crop_image()
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('y'):
                cv2.destroyWindow('img')
                break
            if key == ord('q'):
                quit()
        img_add = np.zeros_like(img, float)
        img_number = 10
        for i in range(img_number):
            raw_image = self.get_rectify_crop_image()
            img_add += raw_image
        img_avg = img_add / img_number
        img_avg = img_avg.astype(np.uint8)
        return img_avg

    def img_list_avg_rectify(self, img_list):
        img_1 = cv2.imread(img_list[0])
        img_add = np.zeros_like(img_1, float)
        for img_path in img_list:
            img = cv2.imread(img_path)
            img_add += img
        img_avg = img_add / len(img_list)
        img_avg = img_avg.astype(np.uint8)
        ref_img_avg = self.rectify_image(img_avg)
        return ref_img_avg


if __name__ == '__main__':
    f = open("shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    camera = Camera(cfg)
    while True:
        raw_img = camera.get_raw_image()
        cv2.imshow('raw_img', raw_img)
        raw_img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('raw_img_GRAY', raw_img_GRAY)
        rectify_img = camera.get_rectify_image()
        cv2.imshow('rectify_img', rectify_img)
        rectify_crop_img = camera.get_rectify_crop_image()
        cv2.imshow('rectify_crop_img', rectify_crop_img)
        rectify_crop_GRAY = cv2.cvtColor(rectify_crop_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('rectify_crop_GRAY', rectify_crop_GRAY)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
