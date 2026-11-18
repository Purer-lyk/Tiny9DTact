"""Tests for the camera identity matcher (shape_reconstruction.cammatch).

Covers the three cases that matter for reconnect-after-unplug on
multi-camera machines:
  1. same camera re-attached -> near-identical frame -> match
  2. built-in webcam -> different structure -> no match, even when
     both frames are similarly dark (mean-absolute-difference would
     false-positive here)
  3. featureless frames (lens cap / extreme exposure) -> never match
"""
import numpy as np

from shape_reconstruction.cammatch import (
    is_same_reference,
    reference_ncc,
    NCC_MATCH_THRESHOLD,
)

RNG = np.random.default_rng(42)


def make_sensor_like_frame(seed: int) -> np.ndarray:
    """A dark frame with structure: illumination falloff + scatter dots
    + sensor noise, roughly like a 9DTact no-contact reference."""
    h, w = 120, 200
    yy, xx = np.mgrid[0:h, 0:w]
    frame = 40.0 + 25.0 * np.exp(-(((xx - w / 2) / 80.0) ** 2
                                   + ((yy - h / 2) / 50.0) ** 2))
    rng = np.random.default_rng(seed)
    # scatter dot grid
    for r in range(10, h, 18):
        for c in range(15, w, 22):
            frame[r - 1:r + 2, c - 1:c + 2] += 15.0
    frame += rng.normal(0, 1.5, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def make_room_like_frame(seed: int) -> np.ndarray:
    """A differently-structured frame (bright-ish, gradients running the
    other way, different 'objects'), like a webcam view."""
    h, w = 120, 200
    yy, xx = np.mgrid[0:h, 0:w]
    frame = 110.0 + 30.0 * (xx / w) - 20.0 * (yy / h)
    rng = np.random.default_rng(seed)
    frame[40:60, 80:150] += 40.0  # a bright blob
    frame += rng.normal(0, 1.5, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_identical_frames_match_with_ncc_one():
    a = make_sensor_like_frame(1)
    matched, score = is_same_reference(a, a.copy())
    assert matched
    assert score > 0.99


def test_same_camera_with_noise_still_matches():
    """Replug noise/warmup differences must not break the match."""
    a = make_sensor_like_frame(1)
    b = make_sensor_like_frame(1)  # same structure, different noise draw
    matched, score = is_same_reference(a, b)
    assert matched, f"score={score}"
    assert score > 0.8


def test_different_camera_is_rejected():
    a = make_sensor_like_frame(1)
    b = make_room_like_frame(2)
    matched, score = is_same_reference(a, b)
    assert not matched
    assert score < NCC_MATCH_THRESHOLD


def test_two_dark_frames_do_not_false_positive():
    """The built-in-webcam-at-exposure--7 trap: both frames dark, but
    structure differs -> NCC (not mean-abs) must reject."""
    dark_a = (make_sensor_like_frame(1).astype(np.float32) * 0.15).astype(np.uint8)
    dark_b = (make_room_like_frame(2).astype(np.float32) * 0.15).astype(np.uint8)
    assert dark_a.mean() < 25 and dark_b.mean() < 25
    matched, score = is_same_reference(dark_a, dark_b)
    assert not matched, f"score={score}"


def test_featureless_frame_never_matches():
    a = make_sensor_like_frame(1)
    flat = np.full_like(a, 20)  # lens-capped webcam
    assert reference_ncc(a, flat) == 0.0
    assert reference_ncc(flat, flat) == 0.0  # even flat vs flat
    matched, _ = is_same_reference(a, flat)
    assert not matched


def test_threshold_between_same_and_different_populations():
    """The threshold must separate the two populations with margin."""
    same_scores = []
    diff_scores = []
    for seed in range(5):
        a = make_sensor_like_frame(seed)
        same_scores.append(reference_ncc(a, make_sensor_like_frame(seed + 100)))
        diff_scores.append(reference_ncc(a, make_room_like_frame(seed + 200)))
    assert min(same_scores) > NCC_MATCH_THRESHOLD
    assert max(diff_scores) < NCC_MATCH_THRESHOLD
