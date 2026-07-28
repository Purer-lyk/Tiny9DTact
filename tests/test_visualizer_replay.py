"""Replay-mode smoke test (opt-in).

Set GUI_TESTS=1 to enable. The replay harness opens an Open3D window,
so it cannot run headless.
"""
import os
import tempfile

import cv2
import numpy as np
import pytest

from shape_reconstruction.visualizer import _replay_main


pytestmark = pytest.mark.skipif(
    os.environ.get("GUI_TESTS") is None,
    reason="GUI tests are opt-in (set GUI_TESTS=1).",
)


def _write_synthetic_frames(tmpdir: str) -> str:
    nx, ny = 200, 100
    bgr = np.full((ny, nx, 3), 30, dtype=np.uint8)
    for i in range(3):
        cv2.imwrite(os.path.join(tmpdir, f"frame_{i:03d}.png"), bgr)
    return tmpdir


def test_replay_processes_three_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = _write_synthetic_frames(tmpdir)
        n = _replay_main(frames_dir, fps=10)
        assert n == 3