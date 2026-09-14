"""Round-trip test for the WebSocket frame encoding."""
import numpy as np
import pytest

from shape_reconstruction.visualizer_ws import _encode_frame, _decode_frame


def test_round_trip_flat():
    h = np.zeros((20, 30), dtype=np.float32)
    readings = {
        "max_depth": 0.0,
        "contact_area_mm2": 0.0,
        "contact_center_mm": (None, None),
        "pixel_per_mm": 0.1,
    }
    blob = _encode_frame(h, readings, fps=30.0, depth_max=0.5, frame_index=42)
    decoded = _decode_frame(blob)
    assert decoded["frame_index"] == 42
    assert decoded["H"] == 20
    assert decoded["W"] == 30
    assert decoded["pixel_per_mm"] == pytest.approx(0.1, abs=1e-5)
    assert decoded["max_depth"] == 0.0
    assert decoded["contact_center_mm"] == (None, None)
    assert decoded["height_map"].shape == (20, 30)  # H, W
    assert np.allclose(decoded["height_map"], 0.0)


def test_round_trip_with_press():
    H, W = 100, 200
    pixel_per_mm = 0.1
    x = (np.arange(W) - W / 2) * pixel_per_mm
    y = (np.arange(H) - H / 2) * pixel_per_mm
    X, Y = np.meshgrid(x, y)
    R, press = 4.0, 1.5
    r = np.sqrt(X ** 2 + Y ** 2)
    cr = np.sqrt(2 * R * press - press ** 2)
    h = np.zeros_like(X)
    m = r < cr
    h[m] = np.sqrt(R ** 2 - r[m] ** 2) - (R - press)
    h = np.clip(h, 0, None).astype(np.float32)

    cx, cy = 0.0, 0.0  # ball press is centered on the grid
    readings = {
        "max_depth": float(h.max()),
        "contact_area_mm2": float((h > 0.05).sum() * pixel_per_mm ** 2),
        "contact_center_mm": (cx, cy),
        "pixel_per_mm": pixel_per_mm,
    }
    blob = _encode_frame(h, readings, fps=29.7, depth_max=1.6, frame_index=7)
    decoded = _decode_frame(blob)
    assert decoded["frame_index"] == 7
    assert decoded["H"] == H
    assert decoded["W"] == W
    assert decoded["max_depth"] == pytest.approx(h.max(), abs=1e-3)
    assert decoded["contact_center_mm"] == (cx, cy)
    assert np.allclose(decoded["height_map"], h, atol=0.01)


def test_round_trip_no_contact_center():
    h = np.zeros((10, 10), dtype=np.float32)
    readings = {
        "max_depth": 0.0,
        "contact_area_mm2": 0.0,
        "contact_center_mm": (None, None),
        "pixel_per_mm": 0.1,
    }
    blob = _encode_frame(h, readings, fps=30.0, depth_max=0.5, frame_index=0)
    decoded = _decode_frame(blob)
    assert decoded["contact_center_mm"] == (None, None)


def test_round_trip_magic():
    blob = _encode_frame(
        np.zeros((5, 5), dtype=np.float32),
        {"max_depth": 0.0, "contact_area_mm2": 0.0,
         "contact_center_mm": (None, None), "pixel_per_mm": 0.1},
        fps=30.0, depth_max=0.5, frame_index=0,
    )
    # Corrupt magic
    bad = bytearray(blob)
    bad[0] = 0xFF
    with pytest.raises(ValueError, match="Bad magic"):
        _decode_frame(bytes(bad))
