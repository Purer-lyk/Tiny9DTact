"""Camera identity matching.

After the sensor's USB camera is unplugged, Windows may re-enumerate
devices so that the configured channel index (e.g. 0) now points at a
different physical camera (typically a laptop's built-in webcam).
Reopening "channel 0" then succeeds, but it streams the wrong device
and the viewer never recovers the real sensor.

This module decides whether a candidate camera is *the same physical
device* as the one the reference frame was captured from, by comparing
the structural similarity (normalised cross-correlation of the mean-
subtracted, downsampled gray images) between the last known-good
reference and a quick probe frame from the candidate:

- Same camera re-attached -> identical gel texture, illumination
  falloff and internal scatter dots -> NCC close to 1.
- Built-in webcam (even a near-black frame at our low exposure) ->
  completely different structure -> NCC around 0.

Plain mean-absolute-difference is deliberately NOT used: both the
sensor's no-contact frame and a webcam at exposure -7 can be dark,
so absolute difference would false-positive on two black frames.

Pure numpy on purpose: importable (and testable) without OpenCV.
"""
from __future__ import annotations

import numpy as np

# NCC at or above this value counts as "same physical camera".
# Same device after replug scores ~0.9; unrelated cameras score <0.2.
NCC_MATCH_THRESHOLD = 0.45

# Below this gray std-dev an image is considered featureless (e.g. a
# webcam with the lens cap on / extreme low exposure) and can never be
# matched -- the sensor frame always shows gel structure.
MIN_STRUCTURE_STD = 2.0

# Both images are subsampled to roughly this grid before comparison.
SAMPLE_GRID = 64


def _subsample(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    sy = max(1, h // SAMPLE_GRID)
    sx = max(1, w // SAMPLE_GRID)
    return gray[::sy, ::sx]


def reference_ncc(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Normalised cross-correlation of two gray images after mean
    subtraction. Returns 0.0 when either image is featureless."""
    a = _subsample(a_gray).astype(np.float64).ravel()
    b = _subsample(b_gray).astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na = float(np.sqrt(np.dot(a, a)))
    nb = float(np.sqrt(np.dot(b, b)))
    if na < MIN_STRUCTURE_STD * a.size ** 0.5 or nb < MIN_STRUCTURE_STD * b.size ** 0.5:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_same_reference(
    a_gray: np.ndarray,
    b_gray: np.ndarray,
    threshold: float = NCC_MATCH_THRESHOLD,
) -> tuple[bool, float]:
    """Return (matched, score). `a_gray` is the anchor (last known-good
    reference), `b_gray` the probe frame from a candidate camera."""
    score = reference_ncc(a_gray, b_gray)
    return bool(score >= threshold), score


__all__ = ["reference_ncc", "is_same_reference",
           "NCC_MATCH_THRESHOLD", "MIN_STRUCTURE_STD"]
