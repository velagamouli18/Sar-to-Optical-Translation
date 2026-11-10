import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB on tensors in [-1.0, 1.0].
    """
    p = pred.detach().clamp(-1.0, 1.0)
    t = target.detach().clamp(-1.0, 1.0)
    mse = (p - t).pow(2).flatten(start_dim=1).mean(dim=1)
    psnr = 10.0 * torch.log10(torch.tensor(4.0, device=mse.device) / torch.clamp(mse, min=1e-10))
    return float(psnr.mean().item())


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Mean Structural Similarity Index Measure (SSIM) across all channels.
    """
    # Convert from [-1, 1] to [0, 1] numpy array
    p = pred.detach().float().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0).numpy()
    t = target.detach().float().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0).numpy()

    # (B, C, H, W) -> average per sample
    scores = []
    for i in range(p.shape[0]):
        # channel_axis=0 for (C, H, W)
        score = ssim_fn(p[i], t[i], channel_axis=0, data_range=1.0)
        scores.append(score)
    return float(np.mean(scores))


def calculate_sam(pred: torch.Tensor, target: torch.Tensor, in_degrees: bool = True) -> float:
    """
    Computes the Spectral Angle Mapper (SAM) between 12-channel multispectral vectors.
    """
    p_ref = (pred.detach().clamp(-1.0, 1.0) + 1.0) * 0.5
    t_ref = (target.detach().clamp(-1.0, 1.0) + 1.0) * 0.5

    dot = (p_ref * t_ref).sum(dim=1)
    norm_p = torch.norm(p_ref, p=2, dim=1)
    norm_t = torch.norm(t_ref, p=2, dim=1)

    cos_theta = dot / (norm_p * norm_t + 1e-7)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

    sam_rad = torch.acos(cos_theta).mean().item()
    if in_degrees:
        return float(np.degrees(sam_rad))
    return float(sam_rad)


# Canonical Sentinel-2 12-band indices:
# 0: B1, 1: B2(Blue), 2: B3(Green), 3: B4(Red), 4: B5, 5: B6,
# 6: B7, 7: B8(NIR), 8: B8A, 9: B9, 10: B11, 11: B12
S2_RGB_BANDS = (3, 2, 1)          # Red (B4), Green (B3), Blue (B2)
S2_FALSE_COLOR_BANDS = (7, 3, 2)  # NIR (B8), Red (B4), Green (B3)


def s2_to_rgb(s2_tensor: torch.Tensor, false_color: bool = False) -> np.ndarray:
    """
    Extracts standard RGB visualization from 12-band Sentinel-2 tensor in [-1.0, 1.0].
    Returns uint8 numpy array (H, W, 3).
    """
    # Tensor shape (12, H, W) or (B, 12, H, W)
    if s2_tensor.ndim == 4:
        s2_tensor = s2_tensor[0]

    s2_01 = s2_tensor.detach().float().cpu().mul(0.5).add(0.5).clamp(0.0, 1.0)

    if s2_01.shape[0] == 3:
        rgb = s2_01
    else:
        bands = S2_FALSE_COLOR_BANDS if false_color else S2_RGB_BANDS
        rgb = torch.stack([s2_01[bands[0]], s2_01[bands[1]], s2_01[bands[2]]], dim=0)

    # Convert to uint8 (H, W, 3)
    rgb_uint8 = rgb.mul(255).add(0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return rgb_uint8
