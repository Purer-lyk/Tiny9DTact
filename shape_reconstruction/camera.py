import time

import numpy as np
import cv2
import yaml


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

    def open_capture(self):
        """(Re)open the capture device with the configured settings.

        Used once at construction time and again to recover after the
        USB camera is unplugged and re-attached. Returns True when the
        device opened successfully.
        """
        self.cap.release()
        self.cap = cv2.VideoCapture(self._camera_channel)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.raw_img_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.raw_img_height)
        self.cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        return True

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
        img = None
        while True:
            try:
                img = self.get_rectify_crop_image()
            except (TypeError, IndexError):
                time.sleep(0.01)
                continue
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
            except (TypeError, IndexError):
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
