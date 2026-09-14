"""Unit tests for the pure-function helpers in shape_reconstruction.visualizer."""
import numpy as np
import pytest

from shape_reconstruction.visualizer import (
    compute_readings,
    jet_vertex_colors,
    jet_uint8_image,
    require_gui,
)


def test_compute_readings_flat_returns_zero_contact():
    height_map = np.zeros((20, 30), dtype=np.float32)
    readings = compute_readings(
        height_map, pixel_per_mm=0.1, contact_threshold=0.05
    )
    assert readings.max_depth == pytest.approx(0.0)
    assert readings.contact_area_mm2 == pytest.approx(0.0)
    assert readings.contact_center_mm == (None, None)


def test_compute_readings_ball_press_matches_geometry():
    nx, ny = 80, 50
    pixel_per_mm = 0.1
    x = (np.arange(nx) - nx / 2) * pixel_per_mm
    y = (np.arange(ny) - ny / 2) * pixel_per_mm
    X, Y = np.meshgrid(x, y)
    R, press = 4.0, 1.5
    r = np.sqrt(X ** 2 + Y ** 2)
    cr = np.sqrt(2 * R * press - press ** 2)
    height = np.zeros_like(X)
    m = r < cr
    height[m] = np.sqrt(R ** 2 - r[m] ** 2) - (R - press)
    height = np.clip(height, 0, None)

    readings = compute_readings(
        height, pixel_per_mm=pixel_per_mm, contact_threshold=0.05
    )
    assert readings.max_depth == pytest.approx(press, abs=1e-3)
    expected_area = np.sum(height > 0.05) * pixel_per_mm ** 2
    assert readings.contact_area_mm2 == pytest.approx(expected_area, rel=1e-6)
    cx, cy = readings.contact_center_mm
    assert abs(cx - (nx / 2) * pixel_per_mm) < pixel_per_mm
    assert abs(cy - (ny / 2) * pixel_per_mm) < pixel_per_mm


def test_jet_vertex_colors_zero_maps_to_deep_blue():
    z = np.array([[0.0, 1.0]], dtype=np.float32)
    colors = jet_vertex_colors(z, depth_max=1.0)
    assert colors.shape == (2, 3)
    # jet(0) is deep blue; jet(1) is deep red.
    assert colors[0, 2] > colors[0, 0]
    assert colors[1, 0] > colors[1, 2]


def test_jet_uint8_image_shape_and_dtype():
    height = np.linspace(0, 2.0, 100).reshape(10, 10).astype(np.float32)
    img = jet_uint8_image(height, depth_max=2.0)
    assert img.shape == (10, 10, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_require_gui_raises_actionable_error_when_missing(monkeypatch):
    import shape_reconstruction.visualizer as viz
    monkeypatch.setattr(viz, "vispy", None)
    monkeypatch.setattr(viz, "pyqtgraph", None)
    with pytest.raises(RuntimeError, match="GUI backend"):
        viz.require_gui()
