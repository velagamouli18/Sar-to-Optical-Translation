"""Shared Sentinel-1/2 normalization, RGB preview, and export helpers.

The full-S2 constants intentionally match ``src/training/folder_dataset.py`` in
the xBD-S12 damage-mapping code.  Keeping one numerical convention is crucial:
the generator is trained in [-1, 1], while exported 12-band TIFFs are converted
back to the original reflectance-value domain expected by the damage loader.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch


S2_BAND_NAMES: Tuple[str, ...] = (
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
)

# B4, B3, B2 in the xBD-S12 band order above.
S2_RGB_INDICES: Tuple[int, int, int] = (3, 2, 1)

S2_P1 = np.asarray(
    [1066, 1060, 1150, 1083, 1095, 1047, 1059, 1047, 1045, 1040, 1055, 1049],
    dtype=np.float32,
)
S2_P99 = np.asarray(
    [2385, 2816, 3321, 4243, 4613, 4859, 5339, 5703, 5705, 5576, 6007, 5419],
    dtype=np.float32,
)

S1_P1 = np.asarray([-23.85894, -38.00315], dtype=np.float32)
S1_P99 = np.asarray([-0.83129, -8.70669], dtype=np.float32)


def optical_channels(mode: str) -> int:
    if mode == "s2_tci":
        return 3
    if mode == "s2":
        return len(S2_BAND_NAMES)
    raise ValueError(f"Unsupported optical mode {mode!r}; choose 's2_tci' or 's2'.")


def rgb_indices(mode: str) -> Tuple[int, int, int]:
    return (0, 1, 2) if mode == "s2_tci" else S2_RGB_INDICES


def select_rgb(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Select RGB channels from BCHW/CHW normalized optical tensors."""
    indices = rgb_indices(mode)
    channel_dim = 1 if tensor.ndim == 4 else 0
    index = torch.as_tensor(indices, device=tensor.device)
    return torch.index_select(tensor, channel_dim, index)


def normalized_s2_to_reflectance(tensor: torch.Tensor) -> torch.Tensor:
    """Invert xBD-S12 percentile normalization to physical reflectance/10000."""
    if tensor.ndim not in (3, 4) or tensor.shape[-3] != 12:
        raise ValueError(f"Expected CHW/BCHW 12-band S2 tensor, got {tuple(tensor.shape)}")
    shape = (1, 12, 1, 1) if tensor.ndim == 4 else (12, 1, 1)
    low = torch.as_tensor(S2_P1, device=tensor.device, dtype=tensor.dtype).view(shape)
    high = torch.as_tensor(S2_P99, device=tensor.device, dtype=tensor.dtype).view(shape)
    raw = ((tensor.clamp(-1.0, 1.0) + 1.0) * 0.5) * (high - low) + low
    return raw / 10000.0


def optical_to_rgb01(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Return display/perceptual RGB in [0, 1] from normalized optical data."""
    return ((select_rgb(tensor, mode).clamp(-1.0, 1.0) + 1.0) * 0.5)


def optical_to_rgb_uint8(tensor: torch.Tensor, mode: str) -> np.ndarray:
    rgb = optical_to_rgb01(tensor.detach().cpu(), mode)
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise ValueError("Pass one sample at a time when converting to uint8.")
        rgb = rgb[0]
    return (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_optical_prediction(tensor: torch.Tensor, path: Path, mode: str) -> Path:
    """Save TCI as RGB PNG or full S2 as 12-band uint16 TIFF (H, W, C)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Pass one sample at a time when saving an optical image.")
        tensor = tensor[0]

    if mode == "s2_tci":
        from PIL import Image

        path = path.with_suffix(".png")
        Image.fromarray(optical_to_rgb_uint8(tensor, mode)).save(path)
        return path

    import rasterio

    path = path.with_suffix(".tif")
    low = torch.as_tensor(S2_P1, device=tensor.device, dtype=tensor.dtype).view(12, 1, 1)
    high = torch.as_tensor(S2_P99, device=tensor.device, dtype=tensor.dtype).view(12, 1, 1)
    raw = ((tensor.clamp(-1.0, 1.0) + 1.0) * 0.5) * (high - low) + low
    raw_chw = raw.round().clamp(0, np.iinfo(np.uint16).max).to(torch.uint16).cpu().numpy()
    # Write an explicit 12-band GeoTIFF layout.  A generic HWC TIFF can look
    # correct in tifffile yet be exposed as a single band by rasterio, which is
    # what the downstream xBD-S12 dataset uses.
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=raw_chw.shape[1],
        width=raw_chw.shape[2],
        count=raw_chw.shape[0],
        dtype="uint16",
    ) as destination:
        destination.write(raw_chw)
    return path