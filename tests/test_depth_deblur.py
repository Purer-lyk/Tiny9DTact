import numpy as np
import pytest

from shape_reconstruction import Camera, Sensor
from _2_Sensor_Calibration import SensorCalibration


def test_enhancement_is_shared_by_every_depth_frame_and_new_reference():
    sensor = Sensor.__new__(Sensor)
    calls = []

    def enhance(frame):
        calls.append(int(frame[0, 0, 0]))
        return frame * 2

    sensor.depth_deblur = enhance
    sensor.Pixel_to_Depth = np.arange(256) / 100
    sensor.max_index = 255
    sensor.lighting_threshold = 0
    sensor.kernel_list = []
    reference = np.full((10, 10, 3), 100, dtype=np.uint8)
    frame = reference - 10
    sensor.set_reference(reference)
    np.testing.assert_allclose(sensor.image_2_height_map(frame), 0.20)
    np.testing.assert_allclose(sensor.image_2_height_map(frame), 0.20)
    sensor.set_reference(frame)
    np.testing.assert_allclose(sensor.image_2_height_map(frame), 0)
    assert calls == [100, 90, 90, 90, 90]
    assert sensor.ref_GRAY[0, 0] == 90


def test_enhanced_mode_rejects_plain_lookup_table(monkeypatch, tmp_path):
    monkeypatch.setattr(Camera, '__init__', lambda camera, cfg: setattr(
        camera, 'calibration_sensor_dir', str(tmp_path)))
    monkeypatch.setattr('shape_reconstruction.sensor.DepthDeblur',
                        lambda checkpoint: type('Deblur', (), {'tag': 'test'})())
    np.save(tmp_path / 'Pixel_to_Depth.npy', [0, 1])
    cfg = {'depth_deblur': {'enabled': True, 'checkpoint': 'model.pth'},
           'depth_calibration': {'depth_calibration_dir': '',
                                 'Pixel_to_Depth_path': '/Pixel_to_Depth.npy'}}
    with pytest.raises(FileNotFoundError, match='Pixel_to_Depth-test.npy'):
        Sensor(cfg)
    sensor = Sensor(cfg, calibrated=False)
    assert sensor.Pixel_to_Depth_path == str(tmp_path / 'Pixel_to_Depth-test.npy')


def test_calibration_uses_plain_contact_circle_but_enhanced_gray_samples(monkeypatch):
    import cv2
    from types import SimpleNamespace

    calibration = SensorCalibration.__new__(SensorCalibration)
    calibration.BallRad = 2.9
    calibration.circle_detection_gray = 10
    calibration.show_circle_detection = False
    calibration.sensor = SimpleNamespace(pixel_per_mm=0.1)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    reference = np.full((61, 61, 3), 100, dtype=np.uint8)
    image = reference - 8
    contact = np.zeros((61, 61), dtype=np.float32)
    cv2.circle(contact, (30, 30), 15, 50, -1)
    gray, depth = calibration.mapping_data_collection(image, reference, contact_diff=contact)
    assert len(gray) > 500
    np.testing.assert_array_equal(gray, np.full(len(gray), 8))
    assert max(depth) > 0.4
