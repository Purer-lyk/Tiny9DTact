"""Shared grayscale difference used by depth calibration and reconstruction."""

import cv2
import numpy as np


def darker_gray_difference(reference, image):
    """Return nonnegative reference-minus-image in gray levels, without wraparound.

    Convert BGR inputs exactly as the live grayscale capture path does, then
    promote to float BEFORE subtraction. Brighter pixels contribute zero.
    """
    def gray(value):
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
        return value.astype(np.float32)

    return np.maximum(gray(reference) - gray(image), 0.0)
