# collect images and calibrate the sensor
import numpy as np
import cv2
import yaml
import os
import argparse
from shape_reconstruction import Sensor
from shape_reconstruction.depth_preprocessing import darker_gray_difference


class SensorCalibration:
    def __init__(self, cfg_path, deblur=False):
        f = open(cfg_path, 'r+', encoding='utf-8')
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if deblur:
            cfg['depth_deblur']['enabled'] = True
        self.sensor = Sensor(cfg, calibrated=False)
        if not os.path.exists(self.sensor.depth_calibration_dir):
            os.makedirs(self.sensor.depth_calibration_dir)
        depth_calibration = cfg['depth_calibration']
        self.BallRad = depth_calibration['BallRad']
        self.circle_detection_gray = depth_calibration['circle_detect_gray']
        self.show_circle_detection = depth_calibration['show_circle_detection']
        self.capture_dir = self.sensor.depth_calibration_dir
        if self.sensor.depth_deblur is not None:
            self.capture_dir = os.path.join(self.capture_dir, self.sensor.depth_deblur.tag)
            os.makedirs(self.capture_dir, exist_ok=True)

    def run(self):
        print("DON'T touch the sensor surface!!!!!")
        print('Please press "y" to save the reference image!')
        ref = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.capture_dir + '/ref.png', ref)
        rc_ref = self.sensor.rectify_crop_image(ref)
        cv2.imwrite(self.capture_dir + '/rectify_crop_ref.png', rc_ref)
        print('Reference image saved!')
        print('Please press the ball on the sensor and press "y" to save the sample image!')
        sample = self.sensor.get_raw_avg_image()
        cv2.imwrite(self.capture_dir + '/sample.png', sample)
        rc_sample = self.sensor.rectify_crop_image(sample)
        cv2.imwrite(self.capture_dir + '/rectify_crop_sample.png', rc_sample)
        print('Sample image saved!')

        contact_diff = (darker_gray_difference(rc_ref, rc_sample)
                        if self.sensor.depth_deblur is not None else None)
        gray_list, depth_list = self.mapping_data_collection(
            self.sensor.prepare_depth_image(rc_sample), self.sensor.prepare_depth_image(rc_ref),
            contact_diff=contact_diff)
        gray_list = np.array(gray_list)
        depth_list = np.array(depth_list)
        Pixel_to_Depth = self.get_list(gray_list, depth_list)
        print(Pixel_to_Depth)
        np.save(self.sensor.Pixel_to_Depth_path, Pixel_to_Depth)

    def circle_detection(self, diff):
        diff_gray = diff
        contact_mask = (diff_gray > self.circle_detection_gray).astype(np.uint8)
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        if len(sorted_areas):
            cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            print(center)
            print(radius)
            if self.show_circle_detection:
                key = -1
                print('If the detected circle is suitable, press the key "q" to continue!')
                while key != ord('q'):
                    center = (int(x), int(y))
                    radius = int(radius)
                    circle_show = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    circle_show = cv2.circle(circle_show, center, radius, (0, 0, 255), 1)
                    circle_show[int(y), int(x)] = [255, 255, 255]
                    cv2.imshow('contact', circle_show.astype(np.uint8))
                    key = cv2.waitKey(0)
                    if key == ord('w'):
                        y -= 1
                    elif key == ord('s'):
                        y += 1
                    elif key == ord('a'):
                        x -= 1
                    elif key == ord('d'):
                        x += 1
                    elif key == ord('m'):
                        radius += 1
                    elif key == ord('n'):
                        radius -= 1
                cv2.destroyWindow('contact')
            return center, radius
        else:
            return (0, 0), 0

    def mapping_data_collection(self, img, ref, *, contact_diff=None):
        gray_list = []
        depth_list = []
        diff = darker_gray_difference(ref, img)
        cv2.imshow('ref', ref)
        cv2.imshow('img', img)
        cv2.imshow('diff', diff.astype(np.uint8))
        center, detect_radius_p = self.circle_detection(diff if contact_diff is None else contact_diff)
        if detect_radius_p <= 0:
            raise ValueError('No valid ball contact detected; capture a new calibration sample.')
        if detect_radius_p * self.sensor.pixel_per_mm >= self.BallRad:
            raise ValueError('Contact radius must be smaller than BallRad; check the circle and pixel scale.')
        if detect_radius_p:
            x = np.linspace(0, diff.shape[0] - 1, diff.shape[0])
            y = np.linspace(0, diff.shape[1] - 1, diff.shape[1])
            xv, yv = np.meshgrid(y, x)
            xv = xv - center[0]
            yv = yv - center[1]
            rv = np.sqrt(xv ** 2 + yv ** 2)
            # print(rv)
            mask = (rv < detect_radius_p)
            # print(mask.shape)
            # print(mask)
            print(self.sensor.pixel_per_mm)
            temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.sensor.pixel_per_mm ** 2
            height_map = (np.sqrt(self.BallRad ** 2 - temp) * mask - np.sqrt(
                self.BallRad ** 2 - (detect_radius_p * self.sensor.pixel_per_mm) ** 2)) * mask
            # print("height_map:", height_map)
            height_map[np.isnan(height_map)] = 0
            diff_gray = diff
            # diff_gray = self.sensor.crop_image(diff_gray)
            # height_map = self.sensor.crop_image(height_map)
            count = 0
            for i in range(height_map.shape[0]):
                for j in range(height_map.shape[1]):
                    if height_map[i, j] > 0:
                        gray_list.append(diff_gray[i, j])
                        depth_list.append(height_map[i, j])
                        count += 1
            # print('Sample points number: {}'.format(count))
            return gray_list, depth_list

    def get_list(self, gray_list, depth_list):
        gray_list = np.asarray(gray_list, dtype=np.float32)
        depth_list = np.asarray(depth_list, dtype=np.float64)
        if gray_list.shape != depth_list.shape or gray_list.ndim != 1:
            raise ValueError('Calibration gray and depth samples must be matching 1D arrays.')
        valid = (np.isfinite(gray_list) & np.isfinite(depth_list)
                 & (gray_list >= 1) & (gray_list <= 255) & (depth_list > 0))
        if not np.any(valid):
            raise ValueError('No valid calibration samples; the depth lookup table was not saved.')
        bins = np.rint(gray_list[valid]).astype(np.intp)
        counts = np.bincount(bins)
        sums = np.bincount(bins, weights=depth_list[valid])
        known = np.flatnonzero(counts)
        # Anchor no intensity change at zero depth and interpolate ALL missing
        # gray levels, not just index zero. Keep the existing variable LUT size.
        return np.interp(np.arange(len(counts)), np.r_[0, known],
                         np.r_[0.0, sums[known] / counts[known]])

if __name__ == '__main__':
    config_path = './shape_reconstruction/shape_config.yaml'
    parser = argparse.ArgumentParser(description='9DTact depth calibration')
    parser.add_argument('--deblur', action='store_true', help='Calibrate with NAFNet enabled')
    depth_calibration = SensorCalibration(config_path, deblur=parser.parse_args().deblur)
    depth_calibration.run()
