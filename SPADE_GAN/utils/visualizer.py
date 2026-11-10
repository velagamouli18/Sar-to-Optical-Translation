import os
from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
from PIL import Image

from .metrics import s2_to_rgb


def save_s2_geotiff(s2_tensor: torch.Tensor, output_path: str):
    """
    Saves a 12-channel Sentinel-2 tensor (in [-1.0, 1.0]) back as a 16-bit uint16 GeoTIFF.
    Reflectance = (s2 + 1.0) * 0.5 * 10000.
    """
    if s2_tensor.ndim == 4:
        s2_tensor = s2_tensor[0]

    # Convert [-1, 1] -> [0, 10000] uint16
    s2_np = s2_tensor.detach().float().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0).numpy()
    s2_uint16 = (s2_np * 10000.0).round().astype(np.uint16)
    # (C, H, W) -> (H, W, C)
    s2_hwc = np.transpose(s2_uint16, (1, 2, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, s2_hwc)


def save_rgb_preview(s2_tensor: torch.Tensor, output_path: str, false_color: bool = False):
    """
    Extracts RGB bands from 12-band Sentinel-2 tensor and saves as a PNG.
    """
    rgb_arr = s2_to_rgb(s2_tensor, false_color=false_color)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb_arr).save(output_path)


def save_sar_preview(s1_tensor: torch.Tensor, output_path: str):
    """
    Saves a 2-channel SAR tensor as an RGB preview using [VV, VH, VV-VH].
    """
    if s1_tensor.ndim == 4:
        s1_tensor = s1_tensor[0]

    # s1_tensor in [-1.0, 1.0]
    s1_01 = s1_tensor.detach().float().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0)
    ch_vv = s1_01[0]
    ch_vh = s1_01[1]
    ch_diff = (s1_01[0] - s1_01[1]).add(0.5).clamp(0.0, 1.0)

    rgb = torch.stack([ch_vv, ch_vh, ch_diff], dim=0)
    rgb_uint8 = rgb.mul(255).add(0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb_uint8).save(output_path)
