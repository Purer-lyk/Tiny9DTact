"""Regression checks for calibration arithmetic, LUT generation and live lookup."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from _2_Sensor_Calibration import SensorCalibration
from shape_reconstruction.depth_preprocessing import darker_gray_difference
from shape_reconstruction.sensor import Sensor


def test_difference_does_not_wrap_or_sum_uint8_channels():
    reference = np.array([[[200, 200, 200], [100, 100, 100]]], dtype=np.uint8)
    image = np.array([[[100, 100, 100], [110, 110, 110]]], dtype=np.uint8)
    np.testing.assert_array_equal(darker_gray_difference(reference, image), [[100, 0]])


def test_colored_calibration_and_live_grayscale_inputs_agree():
    reference = np.array([[[240, 130, 90], [90, 240, 160]]], dtype=np.uint8)
    image = np.array([[[30, 100, 110], [200, 160, 70]]], dtype=np.uint8)
    expected = darker_gray_difference(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    np.testing.assert_array_equal(darker_gray_difference(reference, image), expected)


def test_lookup_averages_samples_and_interpolates_missing_levels():
    calibration = SensorCalibration.__new__(SensorCalibration)
    lut = calibration.get_list([2, 2, 4], [0.1, 0.3, 0.4])
    np.testing.assert_allclose(lut, [0, 0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize('gray,depth', [([], []), ([0], [1]), ([np.nan], [1])])
def test_invalid_samples_rejected(gray, depth):
    calibration = SensorCalibration.__new__(SensorCalibration)
    with pytest.raises(ValueError, match='No valid calibration samples'):
        calibration.get_list(gray, depth)


def test_live_lookup_keeps_large_differences_and_noise_floor():
    sensor = Sensor.__new__(Sensor)
    sensor.ref_GRAY = np.full((15, 15), 220, np.uint8)
    sensor.Pixel_to_Depth = np.arange(256) * 0.01
    sensor.max_index = 255
    sensor.lighting_threshold = 2
    sensor.kernel_list = [3, 3]
    for current, expected in [(220, 0), (230, 0), (219, 0), (120, 0.98), (20, 1.98)]:
        height = sensor.raw_image_2_height_map(np.full((15, 15), current, np.uint8))
        np.testing.assert_allclose(height, expected, atol=1e-6)


def test_ball_image_to_saved_lookup_without_camera(monkeypatch, tmp_path):
    calibration = SensorCalibration.__new__(SensorCalibration)
    calibration.BallRad = 2.9
    calibration.circle_detection_gray = 10
    calibration.show_circle_detection = False
    calibration.sensor = SimpleNamespace(pixel_per_mm=0.1)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    reference = np.full((61, 61, 3), 200, np.uint8)
    image = reference.copy()
    cv2.circle(image, (30, 30), 15, (100, 100, 100), -1)
    gray, depth = calibration.mapping_data_collection(image, reference)
    assert len(gray) > 0
    np.testing.assert_array_equal(gray, np.full(len(gray), 100))
    lut = calibration.get_list(gray, depth)
    path = tmp_path / 'Pixel_to_Depth.npy'
    np.save(path, lut)
    loaded = np.load(path)
    assert len(loaded) == 101
    assert loaded[0] == 0
    assert np.all(np.isfinite(loaded))
    assert np.all(loaded[1:] > 0)


def test_no_contact_rejected_before_lookup_generation(monkeypatch):
    calibration = SensorCalibration.__new__(SensorCalibration)
    calibration.circle_detection_gray = 10
    calibration.show_circle_detection = False
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    image = np.full((15, 15, 3), 100, np.uint8)
    with pytest.raises(ValueError, match='No valid ball contact'):
        calibration.mapping_data_collection(image, image)
