"""Same-resolution NAFNet GoPro width-32 inference for depth images."""

import hashlib
from pathlib import Path

import cv2
import numpy as np


class DepthDeblur:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Deblur weights not found: {checkpoint}. See README.md.")
        try:
            import torch
            from .nafnet import NAFNet
        except ImportError as exc:
            raise ImportError("Depth deblur requires PyTorch; see README.md.") from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        with checkpoint.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            precision = "fp16" if self.dtype == torch.float16 else "fp32"
            self.tag = f"nafnet-v1-{precision}-{digest.hexdigest()[:12]}"
            handle.seek(0)
            weights = torch.load(handle, map_location="cpu", weights_only=True)
        self.model = NAFNet().eval()
        self.model.load_state_dict(weights.get("params", weights), strict=True)
        self.model.to(device=self.device, dtype=self.dtype, memory_format=torch.channels_last)

    def __call__(self, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Depth deblur expects a uint8 BGR image.")
        torch = self.torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            frame = torch.from_numpy(rgb).to(device=self.device, dtype=self.dtype)
            frame = frame.permute(2, 0, 1).unsqueeze(0) / 255.0
            restored = self.model(frame)[0].permute(1, 2, 0)
            restored = restored.float().mul_(255).round_().clamp_(0, 255)
            restored = restored.to(torch.uint8).cpu().numpy()
        return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
