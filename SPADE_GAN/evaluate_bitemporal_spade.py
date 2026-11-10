"""Evaluate/export 3-channel TCI or complete 12-band bitemporal SPADE output with building-aware metrics."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights, vgg19
from tqdm import tqdm
import yaml

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from datasets.bitemporal_s1_s2_dataset import BitemporalS1S2Dataset
from models.bitemporal_change_spade import BitemporalChangeGenerator
from utils.optical import (
    S2_BAND_NAMES,
    normalized_s2_to_reflectance,
    optical_channels,
    optical_to_rgb_uint8,
    save_optical_prediction,
    select_rgb,
)


class VGGLoss(nn.Module):
    """Fallback RGB perceptual distance when the LPIPS package is unavailable."""

    def __init__(self):
        super().__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            nn.Sequential(*[features[i] for i in range(0, 4)]),
            nn.Sequential(*[features[i] for i in range(4, 9)]),
            nn.Sequential(*[features[i] for i in range(9, 18)]),
            nn.Sequential(*[features[i] for i in range(18, 27)]),
        ]).eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x * 0.5 + 0.5 - mean) / std
        y = (y * 0.5 + 0.5 - mean) / std
        loss = x.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def calc_metrics(fake: torch.Tensor, real: torch.Tensor) -> Tuple[float, float, float, float]:
    """PSNR, SSIM, unit-range MAE/L1, and unit-range RMSE on CHW/BCHW data normalized to [-1, 1]."""
    if fake.ndim == 3:
        fake, real = fake.unsqueeze(0), real.unsqueeze(0)
    f = ((fake.clamp(-1.0, 1.0) + 1.0) * 0.5)
    r = ((real.clamp(-1.0, 1.0) + 1.0) * 0.5)
    mse = F.mse_loss(f, r).item()
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))
    mae = F.l1_loss(f, r).item()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_f = F.avg_pool2d(f, 11, stride=1, padding=5)
    mu_r = F.avg_pool2d(r, 11, stride=1, padding=5)
    var_f = F.avg_pool2d(f * f, 11, stride=1, padding=5) - mu_f.square()
    var_r = F.avg_pool2d(r * r, 11, stride=1, padding=5) - mu_r.square()
    cov = F.avg_pool2d(f * r, 11, stride=1, padding=5) - mu_f * mu_r
    ssim = (((2 * mu_f * mu_r + c1) * (2 * cov + c2)) /
            ((mu_f.square() + mu_r.square() + c1) * (var_f + var_r + c2) + 1e-8)).mean()
    return float(psnr), float(ssim.item()), float(mae), float(rmse)


def spectral_angle_degrees(fake: torch.Tensor, real: torch.Tensor) -> float:
    fake_ref = normalized_s2_to_reflectance(fake)
    real_ref = normalized_s2_to_reflectance(real)
    numerator = (fake_ref * real_ref).sum(dim=1)
    denominator = torch.linalg.vector_norm(fake_ref, dim=1) * torch.linalg.vector_norm(real_ref, dim=1)
    cosine = (numerator / denominator.clamp_min(1e-8)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return float(torch.rad2deg(torch.acos(cosine)).mean().item())


SENTINEL2_DOCS = """
================================================================================
 Sentinel-2 12-Band Multispectral Specifications & Domain Conventions
================================================================================
Band Order & Characteristics:
  0: B1   Coastal Aerosol (443 nm, 60m) - atmospheric scattering / aerosol retrieval
  1: B2   Blue (490 nm, 10m)            - RGB Blue channel
  2: B3   Green (560 nm, 10m)           - RGB Green channel
  3: B4   Red (665 nm, 10m)             - RGB Red channel
  4: B5   Red Edge 1 (705 nm, 20m)      - vegetation chlorophyll absorption edge
  5: B6   Red Edge 2 (740 nm, 20m)      - vegetation canopy structure
  6: B7   Red Edge 3 (783 nm, 20m)      - vegetation leaf area index
  7: B8   NIR broad (842 nm, 10m)       - biomass / water contrast
  8: B8A  NIR narrow (865 nm, 20m)      - narrow near-infrared plateau
  9: B9   Water vapour (945 nm, 60m)    - atmospheric water absorption
 10: B11  SWIR 1 (1610 nm, 20m)         - moisture, soil, debris / damage contrast
 11: B12  SWIR 2 (2190 nm, 20m)         - burn scars, minerals, structural damage

Normalization & Processing Pipeline:
  - GAN Tensor Range: [-1.0, 1.0] (hyperbolic tangent output space)
  - Unit Evaluation Range: [0.0, 1.0] via x_unit = (clamp(x, -1, 1) + 1.0) * 0.5
  - Physical Reflectance: [0.0, 1.0] reflectance domain obtained by inverting
    the xBD-S12 per-band P1/P99 percentile scaling:
      raw_dn = x_unit * (P99 - P1) + P1,  Reflectance = raw_dn / 10000.0
  - RGB Perceptual Metrics: Evaluated strictly on channels [3, 2, 1] (B4, B3, B2)
    mapped to [0.0, 1.0], never by treating 12-band tensors as RGB images.
  - Multispectral SAM & ERGAS: Computed strictly across all 12 valid spectral bands
    in physical reflectance, with epsilon guards protecting zero divisions.
================================================================================
"""

SENTINEL2_BAND_SPECS = [
    {"band": "B1", "desc": "Coastal aerosol", "central_wl": "443 nm", "res": "60m"},
    {"band": "B2", "desc": "Blue (RGB B)", "central_wl": "490 nm", "res": "10m"},
    {"band": "B3", "desc": "Green (RGB G)", "central_wl": "560 nm", "res": "10m"},
    {"band": "B4", "desc": "Red (RGB R)", "central_wl": "665 nm", "res": "10m"},
    {"band": "B5", "desc": "Vegetation Red Edge 1", "central_wl": "705 nm", "res": "20m"},
    {"band": "B6", "desc": "Vegetation Red Edge 2", "central_wl": "740 nm", "res": "20m"},
    {"band": "B7", "desc": "Vegetation Red Edge 3", "central_wl": "783 nm", "res": "20m"},
    {"band": "B8", "desc": "NIR broad", "central_wl": "842 nm", "res": "10m"},
    {"band": "B8A", "desc": "NIR narrow", "central_wl": "865 nm", "res": "20m"},
    {"band": "B9", "desc": "Water vapour", "central_wl": "945 nm", "res": "60m"},
    {"band": "B11", "desc": "SWIR 1", "central_wl": "1610 nm", "res": "20m"},
    {"band": "B12", "desc": "SWIR 2", "central_wl": "2190 nm", "res": "20m"},
]


def calc_per_band_metrics(
    fake: torch.Tensor, real: torch.Tensor, optical_mode: str
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Calculate per-band MAE, RMSE, PSNR, SSIM, Pearson correlation r, and ERGAS."""
    if fake.ndim == 3:
        fake, real = fake.unsqueeze(0), real.unsqueeze(0)

    f_u = ((fake.clamp(-1.0, 1.0) + 1.0) * 0.5)
    r_u = ((real.clamp(-1.0, 1.0) + 1.0) * 0.5)

    if optical_mode == "s2":
        f_eval = normalized_s2_to_reflectance(fake)
        r_eval = normalized_s2_to_reflectance(real)
        band_names = S2_BAND_NAMES
    else:
        f_eval = f_u
        r_eval = r_u
        band_names = ("R", "G", "B")

    num_bands = f_eval.shape[1]
    per_band_dict = {}
    sum_rel_mse = 0.0
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    for b in range(num_bands):
        fb = f_eval[:, b:b + 1]
        rb = r_eval[:, b:b + 1]
        fb_u = f_u[:, b:b + 1]
        rb_u = r_u[:, b:b + 1]

        # MAE and RMSE in reflectance units
        mae_b = float(F.l1_loss(fb, rb).item())
        mse_b = float(F.mse_loss(fb, rb).item())
        rmse_b = math.sqrt(max(mse_b, 0.0))

        # PSNR and SSIM on unit range [0, 1]
        mse_u = float(F.mse_loss(fb_u, rb_u).item())
        psnr_b = 10.0 * math.log10(1.0 / max(mse_u, 1e-8))

        mu_f = F.avg_pool2d(fb_u, 11, stride=1, padding=5)
        mu_r = F.avg_pool2d(rb_u, 11, stride=1, padding=5)
        var_f = F.avg_pool2d(fb_u * fb_u, 11, stride=1, padding=5) - mu_f.square()
        var_r = F.avg_pool2d(rb_u * rb_u, 11, stride=1, padding=5) - mu_r.square()
        cov = F.avg_pool2d(fb_u * rb_u, 11, stride=1, padding=5) - mu_f * mu_r
        ssim_b = float((((2 * mu_f * mu_r + c1) * (2 * cov + c2)) /
                        ((mu_f.square() + mu_r.square() + c1) * (var_f + var_r + c2) + 1e-8)).mean().item())

        # Pearson correlation coefficient r
        fb_flat = fb.reshape(-1)
        rb_flat = rb.reshape(-1)
        fb_cent = fb_flat - fb_flat.mean()
        rb_cent = rb_flat - rb_flat.mean()
        denom = torch.sqrt((fb_cent ** 2).sum() * (rb_cent ** 2).sum())
        if denom > 1e-8:
            corr_b = float(((fb_cent * rb_cent).sum() / denom).clamp(-1.0, 1.0).item())
        else:
            corr_b = 0.0

        # Mean of real reference band for relative error normalization (ERGAS)
        mean_r = float(rb.mean().item())
        if abs(mean_r) > 1e-6:
            sum_rel_mse += (rmse_b / mean_r) ** 2
        else:
            sum_rel_mse += (rmse_b / 1e-6) ** 2

        b_name = band_names[b]
        per_band_dict[b_name] = {
            "mae": mae_b,
            "rmse": rmse_b,
            "psnr": psnr_b,
            "ssim": ssim_b,
            "correlation": corr_b,
        }

    # ERGAS: 100 * (h/l) * sqrt((1/B) * sum((RMSE_b / mean_b)^2))
    ergas = float(100.0 * math.sqrt(max(sum_rel_mse / num_bands, 0.0)))
    return per_band_dict, ergas


def load_building_masks(
    mask_dir: Optional[Path], uid: str, target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load and downsample building and damaged-building masks for a sample UID.

    Uses nearest-neighbor interpolation to preserve categorical discrete labels.
    Decodes standard xBD color palettes or single-channel discrete masks:
      1: Intact building
      2: Minor damage
      3: Major damage
      4: Destroyed building
      (Classes 1-4 are all buildings; classes 2-4 are damaged buildings).

    Returns:
        b_mask (bool array HxW): True on any building pixel
        d_mask (bool array HxW): True on damaged-building pixel
        n_b (int): count of building pixels
        n_d (int): count of damaged building pixels
    """
    if mask_dir is None or not mask_dir.is_dir():
        zeros = np.zeros(target_size, dtype=bool)
        return zeros, zeros, 0, 0

    candidates = [
        mask_dir / f"{uid}_mask.png",
        mask_dir / f"{uid}_mask.tif",
        mask_dir / f"{uid}_mask.tiff",
    ]
    mask_path = next((c for c in candidates if c.exists()), None)
    if mask_path is None:
        matches = list(mask_dir.glob(f"{uid}*mask*"))
        if matches:
            mask_path = matches[0]

    if mask_path is None or not mask_path.exists():
        zeros = np.zeros(target_size, dtype=bool)
        return zeros, zeros, 0, 0

    try:
        img = np.array(Image.open(mask_path))
    except Exception:
        import tifffile
        img = np.array(tifffile.imread(str(mask_path)))
    if img.ndim == 2:
        m2d = img
    elif np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
        m2d = img[..., 0]
    else:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        m2d = np.zeros(img.shape[:2], dtype=np.uint8)
        # xBD official disaster RGB palette:
        m2d[(r < 80) & (g > 150) & (b < 80)] = 1      # Green: intact
        m2d[(r > 180) & (g >= 200) & (b < 80)] = 2   # Yellow: minor damage
        m2d[(r > 180) & (g >= 80) & (g < 200) & (b < 80)] = 3  # Orange: major damage
        m2d[(r > 150) & (g < 80) & (b < 80)] = 4      # Red: destroyed
        m2d[(r < 80) & (g < 80) & (b > 150)] = 5      # Blue: unclassified

    if m2d.shape[:2] != target_size:
        m2d = cv2.resize(m2d, target_size, interpolation=cv2.INTER_NEAREST)

    b_mask = (m2d >= 1) & (m2d <= 4)
    d_mask = (m2d >= 2) & (m2d <= 4)
    return b_mask, d_mask, int(b_mask.sum()), int(d_mask.sum())


def save_eval_figure(
    save_path: Path,
    uid: str,
    pre_s2: torch.Tensor,
    post_s1: torch.Tensor,
    real_post_s2: torch.Tensor,
    synth_post_s2: torch.Tensor,
    optical_mode: str,
    rgb_psnr: float,
    rgb_ssim: float,
):
    pre_rgb = optical_to_rgb_uint8(pre_s2, optical_mode)
    real_rgb = optical_to_rgb_uint8(real_post_s2, optical_mode)
    synth_rgb = optical_to_rgb_uint8(synth_post_s2, optical_mode)
    sar_vv = ((post_s1[0].detach().cpu().numpy() + 1.0) * 0.5).clip(0, 1)
    delta_mag = np.abs(synth_rgb.astype(np.float32) - pre_rgb.astype(np.float32)).mean(axis=2)

    if optical_mode == "s2":
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        axes[0, 0].imshow(pre_rgb)
        axes[0, 0].set_title(f"Pre-optical (S2 RGB)\n{uid[:22]}", fontsize=9)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(sar_vv, cmap="gray")
        axes[0, 1].set_title("Post-S1 SAR (VV)", fontsize=9)
        axes[0, 1].axis("off")

        heat = axes[0, 2].imshow(delta_mag, cmap="hot")
        axes[0, 2].set_title("Synthetic Optical Change |post-pre|", fontsize=9)
        axes[0, 2].axis("off")
        plt.colorbar(heat, ax=axes[0, 2], fraction=0.046, pad=0.04)

        axes[1, 0].imshow(real_rgb)
        axes[1, 0].set_title("Real Post-disaster Optical (RGB)", fontsize=9)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(synth_rgb)
        axes[1, 1].set_title(f"Synthetic Post-disaster (RGB)\nPSNR {rgb_psnr:.2f} dB, SSIM {rgb_ssim:.4f}", fontsize=9)
        axes[1, 1].axis("off")

        # 12-Band Spectral Signature Mean Profile
        real_ref = normalized_s2_to_reflectance(real_post_s2.unsqueeze(0))[0].detach().cpu().numpy()
        synth_ref = normalized_s2_to_reflectance(synth_post_s2.unsqueeze(0))[0].detach().cpu().numpy()
        pre_ref = normalized_s2_to_reflectance(pre_s2.unsqueeze(0))[0].detach().cpu().numpy()
        real_curve = real_ref.mean(axis=(1, 2))
        synth_curve = synth_ref.mean(axis=(1, 2))
        pre_curve = pre_ref.mean(axis=(1, 2))

        ax_spec = axes[1, 2]
        bands_list = list(S2_BAND_NAMES)
        ax_spec.plot(bands_list, real_curve, marker="o", label="Real Post S2", color="#1f77b4", linewidth=1.8)
        ax_spec.plot(bands_list, synth_curve, marker="s", linestyle="--", label="Synthetic Post", color="#ff7f0e", linewidth=1.8)
        ax_spec.plot(bands_list, pre_curve, marker="^", linestyle=":", label="Pre S2", color="#7f7f7f", alpha=0.7)
        ax_spec.set_title("12-Band Mean Spectral Signature", fontsize=9)
        ax_spec.set_xlabel("Sentinel-2 Band", fontsize=8)
        ax_spec.set_ylabel("Mean Physical Reflectance", fontsize=8)
        ax_spec.grid(True, linestyle=":", alpha=0.6)
        ax_spec.legend(fontsize=7, loc="best")
        ax_spec.tick_params(axis="x", rotation=45, labelsize=7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        panels = (
            (pre_rgb, f"Pre optical ({optical_mode})\n{uid[:22]}", None),
            (sar_vv, "Post-S1 VV", "gray"),
            (real_rgb, "Real post optical", None),
            (synth_rgb, f"Synthetic post\nRGB PSNR {rgb_psnr:.2f}, SSIM {rgb_ssim:.4f}", None),
        )
        for axis, (image, title, cmap) in zip(axes[:4], panels):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        heat = axes[4].imshow(delta_mag, cmap="hot")
        axes[4].set_title("Synthetic RGB |post-pre|")
        axes[4].axis("off")
        plt.colorbar(heat, ax=axes[4], fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute mean, std, median, and interquartile range (IQR)."""
    if not values:
        return {"mean": None, "std": None, "median": None, "iqr": None}
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "std": None, "median": None, "iqr": None}
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
    }


def _mean_std(values):
    st = _stats(values)
    return st["mean"], st["std"]


def load_generator(checkpoint_path: Path, channels: int, residual_scale: float, device: torch.device) -> BitemporalChangeGenerator:
    """Load generator weights with exact architecture match."""
    generator = BitemporalChangeGenerator(
        ngf=64, optical_channels=channels, residual_scale=residual_scale
    ).to(device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "generator" in loaded:
        state_dict = loaded["generator"]
    else:
        state_dict = loaded
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    dataset,
    loader: DataLoader,
    mask_dir: Optional[Path],
    channels: int,
    residual_scale: float,
    device: torch.device,
    optical_mode: str,
    change_threshold: float,
    output_dir: Path,
    save_images: bool = False,
    save_figures: bool = False,
    max_figures: int = 30,
    model_label: str = "Model",
) -> Tuple[Dict, List[Dict]]:
    """Execute evaluation with exact original metrics and extended MAE/L1, RMSE, and building-aware LPIPS."""
    synth_dir = output_dir / f"synthetic_{optical_mode}_{model_label.lower().replace(' ', '_')}"
    figures_dir = output_dir / f"comparison_figures_{model_label.lower().replace(' ', '_')}"
    if save_images:
        synth_dir.mkdir(parents=True, exist_ok=True)
    if save_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    generator = load_generator(checkpoint_path, channels, residual_scale, device)

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        vgg_fallback = None
    except Exception:
        lpips_fn = None
        vgg_fallback = VGGLoss().to(device).eval()

    all_psnr, all_ssim, all_mae, all_rmse = [], [], [], []
    rgb_psnr, rgb_ssim, rgb_mae, rgb_rmse, rgb_lpips = [], [], [], [], []
    sam_degrees, change_error = [], []
    ergas_list = []
    band_names_active = S2_BAND_NAMES if optical_mode == "s2" else ("R", "G", "B")
    per_band_accum = {b_name: {"mae": [], "rmse": [], "psnr": [], "ssim": [], "corr": []} for b_name in band_names_active}

    # Building-aware metric lists
    b_mae_list, b_rmse_list, b_psnr_list, b_lpips_list = [], [], [], []
    d_mae_list, d_rmse_list, d_psnr_list, d_lpips_list = [], [], [], []
    b_ch_mae_list, d_ch_mae_list = [], []
    b_prec_list, b_rec_list, b_f1_list, b_iou_list = [], [], [], []
    d_prec_list, d_rec_list, d_f1_list, d_iou_list = [], [], [], []
    b_sam_list, d_sam_list = [], []

    per_sample_records: List[Dict] = []
    figures_saved = 0
    started = time.time()

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {model_label} ({optical_mode})")
        for batch in pbar:
            pre_s1 = batch["pre_s1"].to(device, non_blocking=True)
            pre_s2 = batch["pre_s2"].to(device, non_blocking=True)
            post_s1 = batch["post_s1"].to(device, non_blocking=True)
            real = batch["post_s2"].to(device, non_blocking=True)
            fake = generator(pre_s1, pre_s2, post_s1)
            fake_rgb = select_rgb(fake, optical_mode)
            real_rgb = select_rgb(real, optical_mode)

            for index, uid in enumerate(batch["uid"]):
                f_i, r_i, p_i = fake[index:index + 1], real[index:index + 1], pre_s2[index:index + 1]
                fr_i, rr_i = fake_rgb[index:index + 1], real_rgb[index:index + 1]

                # ==========================================================
                # Section A: Existing + All-Band / RGB Metrics
                # ==========================================================
                p_all, s_all, m_all, rmse_all = calc_metrics(f_i, r_i)
                p_rgb, s_rgb, m_rgb, rmse_rgb = calc_metrics(fr_i, rr_i)
                all_psnr.append(p_all); all_ssim.append(s_all); all_mae.append(m_all); all_rmse.append(rmse_all)
                rgb_psnr.append(p_rgb); rgb_ssim.append(s_rgb); rgb_mae.append(m_rgb); rgb_rmse.append(rmse_rgb)

                # Multispectral per-band metrics & ERGAS
                band_metrics_dict, ergas_val = calc_per_band_metrics(f_i, r_i, optical_mode)
                ergas_list.append(ergas_val)
                for b_name, b_data in band_metrics_dict.items():
                    per_band_accum[b_name]["mae"].append(b_data["mae"])
                    per_band_accum[b_name]["rmse"].append(b_data["rmse"])
                    per_band_accum[b_name]["psnr"].append(b_data["psnr"])
                    per_band_accum[b_name]["ssim"].append(b_data["ssim"])
                    per_band_accum[b_name]["corr"].append(b_data["correlation"])

                if lpips_fn is not None:
                    perceptual = float(lpips_fn(fr_i, rr_i).mean().item())
                else:
                    perceptual = float(vgg_fallback(fr_i, rr_i).item() / 15.0)
                rgb_lpips.append(perceptual)

                if optical_mode == "s2":
                    sam_degrees.append(spectral_angle_degrees(f_i, r_i))
                    f_ref = normalized_s2_to_reflectance(f_i) * 10000.0
                    r_ref = normalized_s2_to_reflectance(r_i) * 10000.0
                    p_ref = normalized_s2_to_reflectance(p_i) * 10000.0
                    change_error.append(float((torch.abs(torch.abs(f_ref - p_ref) - torch.abs(r_ref - p_ref))).mean().item()))
                else:
                    error = torch.abs(torch.abs(f_i - p_i) - torch.abs(r_i - p_i)) * 127.5
                    change_error.append(float(error.mean().item()))

                if save_images:
                    suffix = "_post_disaster_s2" if optical_mode == "s2" else "_post_disaster_s2_tci"
                    save_optical_prediction(f_i, synth_dir / f"{uid}{suffix}", optical_mode)
                if save_figures and figures_saved < max_figures:
                    save_eval_figure(
                        figures_dir / f"eval_{figures_saved + 1:03d}_{uid}.png", uid,
                        p_i[0], post_s1[index], r_i[0], f_i[0], optical_mode, p_rgb, s_rgb,
                    )
                    figures_saved += 1

                # ==========================================================
                # Section B: Building-Aware Metrics
                # ==========================================================
                bm_np, dm_np, n_b, n_d = load_building_masks(mask_dir, uid, target_size=(128, 128))
                bm = torch.from_numpy(bm_np).to(device)
                dm = torch.from_numpy(dm_np).to(device)

                # Unit-range [0, 1] representations matching calc_metrics
                f_u = ((f_i[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
                r_u = ((r_i[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
                p_u = ((p_i[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

                diff = torch.abs(f_u - r_u)
                sq_diff = (f_u - r_u) ** 2

                # Optical change residuals
                real_ch = torch.abs(r_u - p_u)
                synth_ch = torch.abs(f_u - p_u)
                ch_diff = torch.abs(synth_ch - real_ch)

                # Binary change map thresholding across all bands
                r_bin = real_ch.mean(dim=0) > change_threshold
                s_bin = synth_ch.mean(dim=0) > change_threshold

                # Pixel-wise Spectral Angle Mapper
                if optical_mode == "s2":
                    f_ref_raw = normalized_s2_to_reflectance(f_i)
                    r_ref_raw = normalized_s2_to_reflectance(r_i)
                    num = (f_ref_raw * r_ref_raw).sum(dim=1)
                    den = torch.linalg.vector_norm(f_ref_raw, dim=1) * torch.linalg.vector_norm(r_ref_raw, dim=1)
                    cos = (num / den.clamp_min(1e-8)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                    sam_map = torch.rad2deg(torch.acos(cos))[0]
                else:
                    sam_map = None

                sample_rec = {
                    "sample_id": uid,
                    "number_of_building_pixels": n_b,
                    "number_of_damaged_building_pixels": n_d,
                    "all_band_psnr": p_all,
                    "all_band_ssim": s_all,
                    "all_band_mae_l1": m_all,
                    "all_band_rmse": rmse_all,
                    "rgb_psnr": p_rgb,
                    "rgb_ssim": s_rgb,
                    "rgb_mae_255": m_rgb * 255.0,
                    "rgb_rmse_255": rmse_rgb * 255.0,
                    "rgb_lpips": perceptual,
                    "ergas": ergas_val,
                    "building_mae": None,
                    "building_l1": None,
                    "building_rmse": None,
                    "building_psnr": None,
                    "building_lpips": None,
                    "damaged_building_mae": None,
                    "damaged_building_l1": None,
                    "damaged_building_rmse": None,
                    "damaged_building_psnr": None,
                    "damaged_building_lpips": None,
                    "building_change_mae": None,
                    "damaged_building_change_mae": None,
                    "building_change_precision": None,
                    "building_change_recall": None,
                    "building_change_f1": None,
                    "building_change_iou": None,
                    "damaged_building_change_precision": None,
                    "damaged_building_change_recall": None,
                    "damaged_building_change_f1": None,
                    "damaged_building_change_iou": None,
                    "building_sam": None,
                    "damaged_building_sam": None,
                }

                # Populate per-band metric values in per-sample record
                for b_name in band_names_active:
                    sample_rec[f"{b_name.lower()}_mae"] = band_metrics_dict[b_name]["mae"]
                    sample_rec[f"{b_name.lower()}_rmse"] = band_metrics_dict[b_name]["rmse"]
                    sample_rec[f"{b_name.lower()}_psnr"] = band_metrics_dict[b_name]["psnr"]
                    sample_rec[f"{b_name.lower()}_ssim"] = band_metrics_dict[b_name]["ssim"]
                    sample_rec[f"{b_name.lower()}_corr"] = band_metrics_dict[b_name]["correlation"]

                if n_b > 0:
                    b_mae = float(diff[:, bm].mean().item())
                    b_mse = float(sq_diff[:, bm].mean().item())
                    b_rmse = float(math.sqrt(max(b_mse, 0.0)))
                    b_psnr = float(10.0 * math.log10(1.0 / max(b_mse, 1e-8)))
                    b_ch_mae = float(ch_diff[:, bm].mean().item())

                    tp_b = float((s_bin & r_bin & bm).sum().item())
                    fp_b = float((s_bin & (~r_bin) & bm).sum().item())
                    fn_b = float(((~s_bin) & r_bin & bm).sum().item())
                    b_prec = float(tp_b / (tp_b + fp_b)) if (tp_b + fp_b) > 0 else 0.0
                    b_rec = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
                    b_f1 = float(2 * tp_b / (2 * tp_b + fp_b + fn_b)) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
                    b_iou = float(tp_b / (tp_b + fp_b + fn_b)) if (tp_b + fp_b + fn_b) > 0 else 0.0

                    b_mae_list.append(b_mae)
                    b_rmse_list.append(b_rmse)
                    b_psnr_list.append(b_psnr)
                    b_ch_mae_list.append(b_ch_mae)
                    b_prec_list.append(b_prec)
                    b_rec_list.append(b_rec)
                    b_f1_list.append(b_f1)
                    b_iou_list.append(b_iou)

                    sample_rec["building_mae"] = b_mae
                    sample_rec["building_l1"] = b_mae
                    sample_rec["building_rmse"] = b_rmse
                    sample_rec["building_psnr"] = b_psnr
                    sample_rec["building_change_mae"] = b_ch_mae
                    sample_rec["building_change_precision"] = b_prec
                    sample_rec["building_change_recall"] = b_rec
                    sample_rec["building_change_f1"] = b_f1
                    sample_rec["building_change_iou"] = b_iou

                    if lpips_fn is not None:
                        bm_4d = bm.unsqueeze(0).unsqueeze(0).float()
                        b_lpips = float(lpips_fn(fr_i * bm_4d, rr_i * bm_4d).mean().item())
                        b_lpips_list.append(b_lpips)
                        sample_rec["building_lpips"] = b_lpips
                    elif vgg_fallback is not None:
                        bm_4d = bm.unsqueeze(0).unsqueeze(0).float()
                        b_lpips = float(vgg_fallback(fr_i * bm_4d, rr_i * bm_4d).item() / 15.0)
                        b_lpips_list.append(b_lpips)
                        sample_rec["building_lpips"] = b_lpips

                    if sam_map is not None:
                        b_sam = float(sam_map[bm].mean().item())
                        b_sam_list.append(b_sam)
                        sample_rec["building_sam"] = b_sam

                if n_d > 0:
                    d_mae = float(diff[:, dm].mean().item())
                    d_mse = float(sq_diff[:, dm].mean().item())
                    d_rmse = float(math.sqrt(max(d_mse, 0.0)))
                    d_psnr = float(10.0 * math.log10(1.0 / max(d_mse, 1e-8)))
                    d_ch_mae = float(ch_diff[:, dm].mean().item())

                    tp_d = float((s_bin & r_bin & dm).sum().item())
                    fp_d = float((s_bin & (~r_bin) & dm).sum().item())
                    fn_d = float(((~s_bin) & r_bin & dm).sum().item())
                    d_prec = float(tp_d / (tp_d + fp_d)) if (tp_d + fp_d) > 0 else 0.0
                    d_rec = float(tp_d / (tp_d + fn_d)) if (tp_d + fn_d) > 0 else 0.0
                    d_f1 = float(2 * tp_d / (2 * tp_d + fp_d + fn_d)) if (2 * tp_d + fp_d + fn_d) > 0 else 0.0
                    d_iou = float(tp_d / (tp_d + fp_d + fn_d)) if (tp_d + fp_d + fn_d) > 0 else 0.0

                    d_mae_list.append(d_mae)
                    d_rmse_list.append(d_rmse)
                    d_psnr_list.append(d_psnr)
                    d_ch_mae_list.append(d_ch_mae)
                    d_prec_list.append(d_prec)
                    d_rec_list.append(d_rec)
                    d_f1_list.append(d_f1)
                    d_iou_list.append(d_iou)

                    sample_rec["damaged_building_mae"] = d_mae
                    sample_rec["damaged_building_l1"] = d_mae
                    sample_rec["damaged_building_rmse"] = d_rmse
                    sample_rec["damaged_building_psnr"] = d_psnr
                    sample_rec["damaged_building_change_mae"] = d_ch_mae
                    sample_rec["damaged_building_change_precision"] = d_prec
                    sample_rec["damaged_building_change_recall"] = d_rec
                    sample_rec["damaged_building_change_f1"] = d_f1
                    sample_rec["damaged_building_change_iou"] = d_iou

                    if lpips_fn is not None:
                        dm_4d = dm.unsqueeze(0).unsqueeze(0).float()
                        d_lpips = float(lpips_fn(fr_i * dm_4d, rr_i * dm_4d).mean().item())
                        d_lpips_list.append(d_lpips)
                        sample_rec["damaged_building_lpips"] = d_lpips
                    elif vgg_fallback is not None:
                        dm_4d = dm.unsqueeze(0).unsqueeze(0).float()
                        d_lpips = float(vgg_fallback(fr_i * dm_4d, rr_i * dm_4d).item() / 15.0)
                        d_lpips_list.append(d_lpips)
                        sample_rec["damaged_building_lpips"] = d_lpips

                    if sam_map is not None:
                        d_sam = float(sam_map[dm].mean().item())
                        d_sam_list.append(d_sam)
                        sample_rec["damaged_building_sam"] = d_sam

                per_sample_records.append(sample_rec)

            pbar.set_postfix({"AllPSNR": f"{np.mean(all_psnr):.2f}", "RGB-LPIPS": f"{np.mean(rgb_lpips):.3f}"})

    elapsed = time.time() - started

    results = {
        # Exact existing keys
        "checkpoint": str(checkpoint_path),
        "optical_mode": optical_mode,
        "optical_channels": channels,
        "residual_scale": residual_scale,
        "num_samples": len(all_psnr),
        "all_band_psnr_db_mean": _mean_std(all_psnr)[0],
        "all_band_psnr_db_std": _mean_std(all_psnr)[1],
        "all_band_ssim_mean": _mean_std(all_ssim)[0],
        "all_band_ssim_std": _mean_std(all_ssim)[1],
        "all_band_unit_mae_mean": _mean_std(all_mae)[0],
        "rgb_psnr_db_mean": _mean_std(rgb_psnr)[0],
        "rgb_psnr_db_std": _mean_std(rgb_psnr)[1],
        "rgb_ssim_mean": _mean_std(rgb_ssim)[0],
        "rgb_ssim_std": _mean_std(rgb_ssim)[1],
        "rgb_mae_255_mean": _mean_std(rgb_mae)[0] * 255.0,
        "rgb_lpips_mean": _mean_std(rgb_lpips)[0],
        "rgb_lpips_std": _mean_std(rgb_lpips)[1],
        "sam_degrees_mean": _mean_std(sam_degrees)[0] if sam_degrees else None,
        "sam_degrees_std": _mean_std(sam_degrees)[1] if sam_degrees else None,
        "change_error_mean": _mean_std(change_error)[0],
        "change_error_units": "raw_S2_reflectance_value" if optical_mode == "s2" else "uint8_intensity",
        "evaluation_time_sec": elapsed,

        # Explicit MAE / L1 & RMSE metrics
        "all_band_mae_mean": _mean_std(all_mae)[0],
        "all_band_mae_std": _mean_std(all_mae)[1],
        "all_band_l1_mean": _mean_std(all_mae)[0],
        "all_band_l1_std": _mean_std(all_mae)[1],
        "all_band_rmse_mean": _mean_std(all_rmse)[0],
        "all_band_rmse_std": _mean_std(all_rmse)[1],

        "rgb_mae_mean": _mean_std(rgb_mae)[0],
        "rgb_mae_std": _mean_std(rgb_mae)[1],
        "rgb_l1_mean": _mean_std(rgb_mae)[0],
        "rgb_l1_std": _mean_std(rgb_mae)[1],
        "rgb_rmse_mean": _mean_std(rgb_rmse)[0],
        "rgb_rmse_std": _mean_std(rgb_rmse)[1],
        "rgb_rmse_255_mean": _mean_std(rgb_rmse)[0] * 255.0 if rgb_rmse else None,
        "rgb_rmse_255_std": _mean_std(rgb_rmse)[1] * 255.0 if rgb_rmse else None,

        # ERGAS (Relative Global Dimensionless Synthesis Error)
        "ergas_mean": _stats(ergas_list)["mean"],
        "ergas_std": _stats(ergas_list)["std"],
        "ergas_median": _stats(ergas_list)["median"],
        "ergas_iqr": _stats(ergas_list)["iqr"],

        # Statistical distributions (Median and IQR for skewed distributions)
        "all_band_psnr_db_median": _stats(all_psnr)["median"],
        "all_band_psnr_db_iqr": _stats(all_psnr)["iqr"],
        "all_band_ssim_median": _stats(all_ssim)["median"],
        "all_band_ssim_iqr": _stats(all_ssim)["iqr"],
        "all_band_mae_median": _stats(all_mae)["median"],
        "all_band_mae_iqr": _stats(all_mae)["iqr"],
        "all_band_rmse_median": _stats(all_rmse)["median"],
        "all_band_rmse_iqr": _stats(all_rmse)["iqr"],

        "rgb_psnr_db_median": _stats(rgb_psnr)["median"],
        "rgb_psnr_db_iqr": _stats(rgb_psnr)["iqr"],
        "rgb_ssim_median": _stats(rgb_ssim)["median"],
        "rgb_ssim_iqr": _stats(rgb_ssim)["iqr"],
        "rgb_mae_255_median": _stats(rgb_mae)["median"] * 255.0 if _stats(rgb_mae)["median"] is not None else None,
        "rgb_mae_255_iqr": _stats(rgb_mae)["iqr"] * 255.0 if _stats(rgb_mae)["iqr"] is not None else None,
        "rgb_rmse_255_median": _stats(rgb_rmse)["median"] * 255.0 if _stats(rgb_rmse)["median"] is not None else None,
        "rgb_rmse_255_iqr": _stats(rgb_rmse)["iqr"] * 255.0 if _stats(rgb_rmse)["iqr"] is not None else None,
        "rgb_lpips_median": _stats(rgb_lpips)["median"],
        "rgb_lpips_iqr": _stats(rgb_lpips)["iqr"],

        "sam_degrees_median": _stats(sam_degrees)["median"] if sam_degrees else None,
        "sam_degrees_iqr": _stats(sam_degrees)["iqr"] if sam_degrees else None,
        "change_error_median": _stats(change_error)["median"],
        "change_error_iqr": _stats(change_error)["iqr"],

        # Per-band multispectral metrics dictionary
        "per_band_metrics": {
            b: {
                "mae": _stats(per_band_accum[b]["mae"]),
                "rmse": _stats(per_band_accum[b]["rmse"]),
                "psnr": _stats(per_band_accum[b]["psnr"]),
                "ssim": _stats(per_band_accum[b]["ssim"]),
                "correlation": _stats(per_band_accum[b]["corr"]),
            } for b in band_names_active
        },
        "multispectral_documentation": SENTINEL2_DOCS.strip(),

        # Building-aware metadata & metrics
        "change_threshold_used": change_threshold,
        "num_samples_evaluated": len(all_psnr),
        "num_samples_valid_building": len(b_mae_list),
        "num_samples_valid_damaged_building": len(d_mae_list),

        "building_mae_mean": _mean_std(b_mae_list)[0],
        "building_mae_std": _mean_std(b_mae_list)[1],
        "building_mae_median": _stats(b_mae_list)["median"],
        "building_mae_iqr": _stats(b_mae_list)["iqr"],
        "building_l1_mean": _mean_std(b_mae_list)[0],
        "building_l1_std": _mean_std(b_mae_list)[1],
        "building_rmse_mean": _mean_std(b_rmse_list)[0],
        "building_rmse_std": _mean_std(b_rmse_list)[1],
        "building_rmse_median": _stats(b_rmse_list)["median"],
        "building_rmse_iqr": _stats(b_rmse_list)["iqr"],
        "building_psnr_db_mean": _mean_std(b_psnr_list)[0],
        "building_psnr_db_std": _mean_std(b_psnr_list)[1],
        "building_psnr_db_median": _stats(b_psnr_list)["median"],
        "building_psnr_db_iqr": _stats(b_psnr_list)["iqr"],
        "building_lpips_mean": _mean_std(b_lpips_list)[0],
        "building_lpips_std": _mean_std(b_lpips_list)[1],
        "building_lpips_median": _stats(b_lpips_list)["median"],
        "building_lpips_iqr": _stats(b_lpips_list)["iqr"],

        "damaged_building_mae_mean": _mean_std(d_mae_list)[0],
        "damaged_building_mae_std": _mean_std(d_mae_list)[1],
        "damaged_building_mae_median": _stats(d_mae_list)["median"],
        "damaged_building_mae_iqr": _stats(d_mae_list)["iqr"],
        "damaged_building_l1_mean": _mean_std(d_mae_list)[0],
        "damaged_building_l1_std": _mean_std(d_mae_list)[1],
        "damaged_building_rmse_mean": _mean_std(d_rmse_list)[0],
        "damaged_building_rmse_std": _mean_std(d_rmse_list)[1],
        "damaged_building_rmse_median": _stats(d_rmse_list)["median"],
        "damaged_building_rmse_iqr": _stats(d_rmse_list)["iqr"],
        "damaged_building_psnr_db_mean": _mean_std(d_psnr_list)[0],
        "damaged_building_psnr_db_std": _mean_std(d_psnr_list)[1],
        "damaged_building_psnr_db_median": _stats(d_psnr_list)["median"],
        "damaged_building_psnr_db_iqr": _stats(d_psnr_list)["iqr"],
        "damaged_building_lpips_mean": _mean_std(d_lpips_list)[0],
        "damaged_building_lpips_std": _mean_std(d_lpips_list)[1],
        "damaged_building_lpips_median": _stats(d_lpips_list)["median"],
        "damaged_building_lpips_iqr": _stats(d_lpips_list)["iqr"],

        "building_change_mae_mean": _mean_std(b_ch_mae_list)[0],
        "building_change_mae_std": _mean_std(b_ch_mae_list)[1],
        "damaged_building_change_mae_mean": _mean_std(d_ch_mae_list)[0],
        "damaged_building_change_mae_std": _mean_std(d_ch_mae_list)[1],

        "building_change_precision_mean": _mean_std(b_prec_list)[0],
        "building_change_precision_std": _mean_std(b_prec_list)[1],
        "building_change_recall_mean": _mean_std(b_rec_list)[0],
        "building_change_recall_std": _mean_std(b_rec_list)[1],
        "building_change_f1_mean": _mean_std(b_f1_list)[0],
        "building_change_f1_std": _mean_std(b_f1_list)[1],
        "building_change_iou_mean": _mean_std(b_iou_list)[0],
        "building_change_iou_std": _mean_std(b_iou_list)[1],

        "damaged_building_change_precision_mean": _mean_std(d_prec_list)[0],
        "damaged_building_change_precision_std": _mean_std(d_prec_list)[1],
        "damaged_building_change_recall_mean": _mean_std(d_rec_list)[0],
        "damaged_building_change_recall_std": _mean_std(d_rec_list)[1],
        "damaged_building_change_f1_mean": _mean_std(d_f1_list)[0],
        "damaged_building_change_f1_std": _mean_std(d_f1_list)[1],
        "damaged_building_change_iou_mean": _mean_std(d_iou_list)[0],
        "damaged_building_change_iou_std": _mean_std(d_iou_list)[1],

        "building_sam_degrees_mean": _mean_std(b_sam_list)[0],
        "building_sam_degrees_std": _mean_std(b_sam_list)[1],
        "damaged_building_sam_degrees_mean": _mean_std(d_sam_list)[0],
        "damaged_building_sam_degrees_std": _mean_std(d_sam_list)[1],
    }

    return results, per_sample_records


def print_evaluation_summary(results: Dict):
    """Print comprehensive summary with statistical distributions and multispectral breakdown."""
    if results.get("optical_mode") == "s2":
        print(SENTINEL2_DOCS)

    print("\n" + "=" * 84)
    print(" SECTION A: ESSENTIAL & GLOBAL MULTISPECTRAL METRICS")
    print("=" * 84)
    print(f"Total Samples Evaluated        : {results['num_samples']}")
    print(f"All-band PSNR (dB) [Higher is better] : {results['all_band_psnr_db_mean']:.2f} +/- {results['all_band_psnr_db_std']:.2f} dB  |  Median: {results.get('all_band_psnr_db_median', 0.0):.2f} (IQR: {results.get('all_band_psnr_db_iqr', 0.0):.2f})")
    print(f"All-band SSIM [Higher is better]      : {results['all_band_ssim_mean']:.4f} +/- {results['all_band_ssim_std']:.4f}      |  Median: {results.get('all_band_ssim_median', 0.0):.4f} (IQR: {results.get('all_band_ssim_iqr', 0.0):.4f})")
    if results.get("all_band_mae_mean") is not None:
        print(f"All-band MAE / L1 [Lower is better]  : {results['all_band_mae_mean']:.4f} +/- {results['all_band_mae_std']:.4f}      |  Median: {results.get('all_band_mae_median', 0.0):.4f} (IQR: {results.get('all_band_mae_iqr', 0.0):.4f})")
    if results.get("all_band_rmse_mean") is not None:
        print(f"All-band RMSE [Lower is better]      : {results['all_band_rmse_mean']:.4f} +/- {results['all_band_rmse_std']:.4f}      |  Median: {results.get('all_band_rmse_median', 0.0):.4f} (IQR: {results.get('all_band_rmse_iqr', 0.0):.4f})")
    if results.get("ergas_mean") is not None:
        print(f"ERGAS (Multispectral Error) [Lower]  : {results['ergas_mean']:.4f} +/- {results['ergas_std']:.4f}      |  Median: {results['ergas_median']:.4f} (IQR: {results['ergas_iqr']:.4f})")
    print("-" * 84)
    print(f"RGB PSNR (dB) [Higher is better]      : {results['rgb_psnr_db_mean']:.2f} +/- {results['rgb_psnr_db_std']:.2f} dB  |  Median: {results.get('rgb_psnr_db_median', 0.0):.2f} (IQR: {results.get('rgb_psnr_db_iqr', 0.0):.2f})")
    print(f"RGB SSIM [Higher is better]           : {results['rgb_ssim_mean']:.4f} +/- {results['rgb_ssim_std']:.4f}      |  Median: {results.get('rgb_ssim_median', 0.0):.4f} (IQR: {results.get('rgb_ssim_iqr', 0.0):.4f})")
    print(f"RGB LPIPS [Lower is better]           : {results['rgb_lpips_mean']:.4f} +/- {results['rgb_lpips_std']:.4f}      |  Median: {results.get('rgb_lpips_median', 0.0):.4f} (IQR: {results.get('rgb_lpips_iqr', 0.0):.4f})")
    print(f"RGB MAE (255) / RMSE (255) [Lower]    : {results['rgb_mae_255_mean']:.2f} / {results.get('rgb_rmse_255_mean', 0.0):.2f} (out of 255)")
    if results.get("sam_degrees_mean") is not None:
        print(f"Global SAM (degrees) [Lower is better]: {results['sam_degrees_mean']:.2f} deg +/- {results['sam_degrees_std']:.2f} deg  |  Median: {results.get('sam_degrees_median', 0.0):.2f} deg (IQR: {results.get('sam_degrees_iqr', 0.0):.2f} deg)")
    print(f"Global Optical Change Error [Lower]   : {results['change_error_mean']:.2f} {results['change_error_units']}")

    # Section A.2: Multispectral Per-Band Performance
    if results.get("optical_mode") == "s2" and "per_band_metrics" in results:
        print("\n" + "=" * 84)
        print(" SECTION A.2: SENTINEL-2 12-BAND MULTISPECTRAL BREAKDOWN")
        print("=" * 84)
        spec_map = {s["band"]: s for s in SENTINEL2_BAND_SPECS}
        b_hdr = f"{'Band':<5} | {'Description':<22} | {'Wl':<7} | {'MAE (Refl)':<11} | {'RMSE (Refl)':<11} | {'PSNR':<9} | {'SSIM':<7} | {'Corr (r)':<8}"
        print(b_hdr)
        print("-" * len(b_hdr))

        best_band_mae = (None, 999.0)
        worst_band_mae = (None, -1.0)

        for band_id, b_stats in results["per_band_metrics"].items():
            spec = spec_map.get(band_id, {})
            desc = spec.get("desc", band_id)[:22]
            wl = spec.get("central_wl", "-")
            mae_m = b_stats["mae"]["mean"]
            rmse_m = b_stats["rmse"]["mean"]
            psnr_m = b_stats["psnr"]["mean"]
            ssim_m = b_stats["ssim"]["mean"]
            corr_m = b_stats["correlation"]["mean"]

            if mae_m < best_band_mae[1]:
                best_band_mae = (band_id, mae_m)
            if mae_m > worst_band_mae[1]:
                worst_band_mae = (band_id, mae_m)

            print(f"{band_id:<5} | {desc:<22} | {wl:<7} | {mae_m:<11.4f} | {rmse_m:<11.4f} | {psnr_m:<6.2f} dB | {ssim_m:<7.4f} | {corr_m:<+8.4f}")

        print("-" * len(b_hdr))
        print(f"Lowest Synthesis Error (Best Band)  : {best_band_mae[0]} (MAE: {best_band_mae[1]:.4f} reflectance)")
        print(f"Highest Synthesis Error (Worst Band) : {worst_band_mae[0]} (MAE: {worst_band_mae[1]:.4f} reflectance)")

    print("\n" + "=" * 84)
    print(" SECTION B: BUILDING-AWARE METRICS (INCLUDING MAE/L1, RMSE & LPIPS)")
    print("=" * 84)
    print(f"Samples with Valid Building Masks         : {results['num_samples_valid_building']} / {results['num_samples_evaluated']}")
    print(f"Samples with Valid Damaged Building Masks : {results['num_samples_valid_damaged_building']} / {results['num_samples_evaluated']}")
    print(f"Optical Change Threshold Used             : {results['change_threshold_used']} (on [0,1] unit range)")
    print("-" * 84)
    if results["building_mae_mean"] is not None:
        print(f"Building MAE / L1 [Lower is better]       : {results['building_mae_mean']:.4f} +/- {results['building_mae_std']:.4f}  |  Median: {results.get('building_mae_median', 0.0):.4f} (IQR: {results.get('building_mae_iqr', 0.0):.4f})")
        print(f"Building RMSE [Lower is better]           : {results['building_rmse_mean']:.4f} +/- {results['building_rmse_std']:.4f}  |  Median: {results.get('building_rmse_median', 0.0):.4f} (IQR: {results.get('building_rmse_iqr', 0.0):.4f})")
        print(f"Building PSNR [Higher is better]          : {results['building_psnr_db_mean']:.2f} +/- {results['building_psnr_db_std']:.2f} dB")
    if results.get("building_lpips_mean") is not None:
        print(f"Building LPIPS (Masked) [Lower]           : {results['building_lpips_mean']:.4f} +/- {results['building_lpips_std']:.4f}  |  Median: {results.get('building_lpips_median', 0.0):.4f} (IQR: {results.get('building_lpips_iqr', 0.0):.4f})")
    if results["damaged_building_mae_mean"] is not None:
        print(f"Damaged Building MAE / L1 [Lower]         : {results['damaged_building_mae_mean']:.4f} +/- {results['damaged_building_mae_std']:.4f}  |  Median: {results.get('damaged_building_mae_median', 0.0):.4f} (IQR: {results.get('damaged_building_mae_iqr', 0.0):.4f})")
        print(f"Damaged Building RMSE [Lower]             : {results['damaged_building_rmse_mean']:.4f} +/- {results['damaged_building_rmse_std']:.4f}  |  Median: {results.get('damaged_building_rmse_median', 0.0):.4f} (IQR: {results.get('damaged_building_rmse_iqr', 0.0):.4f})")
        print(f"Damaged Building PSNR [Higher]            : {results['damaged_building_psnr_db_mean']:.2f} +/- {results['damaged_building_psnr_db_std']:.2f} dB")
    if results.get("damaged_building_lpips_mean") is not None:
        print(f"Damaged Building LPIPS (Masked) [Lower]   : {results['damaged_building_lpips_mean']:.4f} +/- {results['damaged_building_lpips_std']:.4f}  |  Median: {results.get('damaged_building_lpips_median', 0.0):.4f} (IQR: {results.get('damaged_building_lpips_iqr', 0.0):.4f})")
    if results["building_change_mae_mean"] is not None:
        print(f"Building Optical Change MAE [Lower]       : {results['building_change_mae_mean']:.4f} +/- {results['building_change_mae_std']:.4f}")
    if results["damaged_building_change_mae_mean"] is not None:
        print(f"Damaged Building Optical Change MAE [Lower]: {results['damaged_building_change_mae_mean']:.4f} +/- {results['damaged_building_change_mae_std']:.4f}")
    if results["building_change_f1_mean"] is not None:
        print(f"Building Change Precision / Recall        : {results['building_change_precision_mean']:.4f} / {results['building_change_recall_mean']:.4f}")
        print(f"Building Change F1 / IoU [Higher]         : {results['building_change_f1_mean']:.4f} / {results['building_change_iou_mean']:.4f}")
    if results["damaged_building_change_f1_mean"] is not None:
        print(f"Damaged Building Change Precision / Recall: {results['damaged_building_change_precision_mean']:.4f} / {results['damaged_building_change_recall_mean']:.4f}")
        print(f"Damaged Building Change F1 / IoU [Higher] : {results['damaged_building_change_f1_mean']:.4f} / {results['damaged_building_change_iou_mean']:.4f}")
    if results["building_sam_degrees_mean"] is not None:
        print(f"Building SAM [Lower is better]            : {results['building_sam_degrees_mean']:.2f} deg +/- {results['building_sam_degrees_std']:.2f} deg")
    if results["damaged_building_sam_degrees_mean"] is not None:
        print(f"Damaged Building SAM [Lower is better]    : {results['damaged_building_sam_degrees_mean']:.2f} deg +/- {results['damaged_building_sam_degrees_std']:.2f} deg")
    print("=" * 84)


def generate_aggregate_metrics_table(results: Dict) -> List[Dict]:
    """Compile comprehensive aggregate summary table with higher/lower is better indicators."""
    items = [
        ("All-Band PSNR", "Global Multispectral", "all_band_psnr_db", True, "dB"),
        ("All-Band SSIM", "Global Multispectral", "all_band_ssim", True, "unitless"),
        ("All-Band MAE / L1", "Global Multispectral", "all_band_mae", False, "unit [0,1]"),
        ("All-Band RMSE", "Global Multispectral", "all_band_rmse", False, "unit [0,1]"),
        ("ERGAS", "Global Multispectral", "ergas", False, "unitless"),
        ("Global SAM", "Global Multispectral", "sam_degrees", False, "degrees"),
        ("Global Optical Change Error", "Global Multispectral", "change_error", False, results.get("change_error_units", "")),

        ("RGB PSNR", "RGB Perceptual", "rgb_psnr_db", True, "dB"),
        ("RGB SSIM", "RGB Perceptual", "rgb_ssim", True, "unitless"),
        ("RGB MAE (255)", "RGB Perceptual", "rgb_mae_255", False, "intensity [0-255]"),
        ("RGB RMSE (255)", "RGB Perceptual", "rgb_rmse_255", False, "intensity [0-255]"),
        ("RGB LPIPS", "RGB Perceptual", "rgb_lpips", False, "perceptual dist"),

        ("Building MAE / L1", "Building Regional", "building_mae", False, "unit [0,1]"),
        ("Building RMSE", "Building Regional", "building_rmse", False, "unit [0,1]"),
        ("Building PSNR", "Building Regional", "building_psnr_db", True, "dB"),
        ("Building LPIPS", "Building Regional", "building_lpips", False, "perceptual dist"),
        ("Building SAM", "Building Regional", "building_sam_degrees", False, "degrees"),
        ("Building Change MAE", "Building Regional", "building_change_mae", False, "unit [0,1]"),
        ("Building Change Precision", "Building Regional", "building_change_precision", True, "score"),
        ("Building Change Recall", "Building Regional", "building_change_recall", True, "score"),
        ("Building Change F1", "Building Regional", "building_change_f1", True, "score"),
        ("Building Change IoU", "Building Regional", "building_change_iou", True, "score"),

        ("Damaged Building MAE / L1", "Damaged Building", "damaged_building_mae", False, "unit [0,1]"),
        ("Damaged Building RMSE", "Damaged Building", "damaged_building_rmse", False, "unit [0,1]"),
        ("Damaged Building PSNR", "Damaged Building", "damaged_building_psnr_db", True, "dB"),
        ("Damaged Building LPIPS", "Damaged Building", "damaged_building_lpips", False, "perceptual dist"),
        ("Damaged Building SAM", "Damaged Building", "damaged_building_sam_degrees", False, "degrees"),
        ("Damaged Building Change MAE", "Damaged Building", "damaged_building_change_mae", False, "unit [0,1]"),
        ("Damaged Building Change Precision", "Damaged Building", "damaged_building_change_precision", True, "score"),
        ("Damaged Building Change Recall", "Damaged Building", "damaged_building_change_recall", True, "score"),
        ("Damaged Building Change F1", "Damaged Building", "damaged_building_change_f1", True, "score"),
        ("Damaged Building Change IoU", "Damaged Building", "damaged_building_change_iou", True, "score"),
    ]

    records = []
    for name, domain, prefix, higher_is_better, units in items:
        mean_val = results.get(f"{prefix}_mean")
        std_val = results.get(f"{prefix}_std")
        med_val = results.get(f"{prefix}_median")
        iqr_val = results.get(f"{prefix}_iqr")

        if mean_val is None:
            continue

        records.append({
            "metric_name": name,
            "domain": domain,
            "direction": "Higher is better (↑)" if higher_is_better else "Lower is better (↓)",
            "units": units,
            "mean": mean_val,
            "std": std_val,
            "median": med_val,
            "iqr": iqr_val,
        })
    return records


def generate_per_band_table(results: Dict, optical_mode: str) -> List[Dict]:
    """Compile per-band metrics summary table."""
    records = []
    per_band = results.get("per_band_metrics", {})
    if not per_band:
        return records

    spec_map = {s["band"]: s for s in SENTINEL2_BAND_SPECS} if optical_mode == "s2" else {}

    for band, data in per_band.items():
        spec = spec_map.get(band, {})
        records.append({
            "band": band,
            "description": spec.get("desc", band),
            "central_wavelength": spec.get("central_wl", "-"),
            "spatial_resolution": spec.get("res", "-"),
            "mae_reflectance_mean": data["mae"]["mean"],
            "mae_reflectance_std": data["mae"]["std"],
            "mae_reflectance_median": data["mae"]["median"],
            "mae_reflectance_iqr": data["mae"]["iqr"],
            "rmse_reflectance_mean": data["rmse"]["mean"],
            "rmse_reflectance_std": data["rmse"]["std"],
            "rmse_reflectance_median": data["rmse"]["median"],
            "rmse_reflectance_iqr": data["rmse"]["iqr"],
            "psnr_db_mean": data["psnr"]["mean"],
            "psnr_db_std": data["psnr"]["std"],
            "ssim_mean": data["ssim"]["mean"],
            "ssim_std": data["ssim"]["std"],
            "pearson_correlation_mean": data["correlation"]["mean"],
            "pearson_correlation_std": data["correlation"]["std"],
        })
    return records


def generate_comparison_table(naive_res: Dict, topk_res: Dict) -> Tuple[str, List[Dict]]:
    """Format SECTION C: Naive vs Top-K comparison table including MAE/L1, RMSE, and LPIPS."""
    metric_specs = [
        # Building Level
        ("Building MAE / L1", "building_mae_mean", True, ".4f"),
        ("Building RMSE", "building_rmse_mean", True, ".4f"),
        ("Building PSNR", "building_psnr_db_mean", False, ".2f dB"),
        ("Building LPIPS", "building_lpips_mean", True, ".4f"),
        ("Damaged Building MAE / L1", "damaged_building_mae_mean", True, ".4f"),
        ("Damaged Building RMSE", "damaged_building_rmse_mean", True, ".4f"),
        ("Damaged Building PSNR", "damaged_building_psnr_db_mean", False, ".2f dB"),
        ("Damaged Building LPIPS", "damaged_building_lpips_mean", True, ".4f"),
        ("Building Change MAE", "building_change_mae_mean", True, ".4f"),
        ("Damaged Building Change MAE", "damaged_building_change_mae_mean", True, ".4f"),
        ("Building Change Precision", "building_change_precision_mean", False, ".4f"),
        ("Building Change Recall", "building_change_recall_mean", False, ".4f"),
        ("Building Change F1", "building_change_f1_mean", False, ".4f"),
        ("Building Change IoU", "building_change_iou_mean", False, ".4f"),
        ("Damaged Building Change Precision", "damaged_building_change_precision_mean", False, ".4f"),
        ("Damaged Building Change Recall", "damaged_building_change_recall_mean", False, ".4f"),
        ("Damaged Building Change F1", "damaged_building_change_f1_mean", False, ".4f"),
        ("Damaged Building Change IoU", "damaged_building_change_iou_mean", False, ".4f"),
        ("Building SAM", "building_sam_degrees_mean", True, ".2f°"),
        ("Damaged Building SAM", "damaged_building_sam_degrees_mean", True, ".2f°"),

        # Global Level
        ("All-Band MAE / L1 (Global)", "all_band_mae_mean", True, ".4f"),
        ("All-Band RMSE (Global)", "all_band_rmse_mean", True, ".4f"),
        ("All-Band PSNR (Global)", "all_band_psnr_db_mean", False, ".2f dB"),
        ("All-Band SSIM (Global)", "all_band_ssim_mean", False, ".4f"),
        ("ERGAS (Multispectral Error)", "ergas_mean", True, ".4f"),
        ("RGB MAE (255)", "rgb_mae_255_mean", True, ".2f"),
        ("RGB RMSE (255)", "rgb_rmse_255_mean", True, ".2f"),
        ("RGB LPIPS (Global)", "rgb_lpips_mean", True, ".4f"),
        ("Global SAM", "sam_degrees_mean", True, ".2f°"),
    ]

    header = f"{'Metric':<35} | {'Naive':<12} | {'Top-K':<12} | {'Difference':<12} | {'Better':<10}"
    separator = "-" * len(header)
    lines = ["\n" + "=" * len(header), " SECTION C: NAIVE VS TOP-K COMPARISON", "=" * len(header), header, separator]

    records = []
    for label, key, lower_is_better, fmt in metric_specs:
        v_naive = naive_res.get(key)
        v_topk = topk_res.get(key)
        if v_naive is None or v_topk is None:
            continue

        diff = v_topk - v_naive
        if lower_is_better:
            better = "Top-K" if diff < -1e-6 else ("Naive" if diff > 1e-6 else "Tie")
        else:
            better = "Top-K" if diff > 1e-6 else ("Naive" if diff < -1e-6 else "Tie")

        # Format values
        suffix = ""
        if "dB" in fmt:
            suffix = " dB"
            fmt_clean = fmt.replace(" dB", "")
        elif "°" in fmt:
            suffix = "°"
            fmt_clean = fmt.replace("°", "")
        else:
            fmt_clean = fmt

        str_n = f"{v_naive:{fmt_clean}}{suffix}"
        str_t = f"{v_topk:{fmt_clean}}{suffix}"
        str_d = f"{diff:+{fmt_clean}}{suffix}"

        line = f"{label:<35} | {str_n:<12} | {str_t:<12} | {str_d:<12} | {better:<10}"
        lines.append(line)
        records.append({
            "metric": label,
            "naive": v_naive,
            "top_k": v_topk,
            "difference": diff,
            "better": better,
            "lower_is_better": lower_is_better,
        })

    lines.append("=" * len(header))
    return "\n".join(lines), records


def save_per_sample_csv(records: List[Dict], output_file: Path):
    """Save per-sample metrics to CSV file."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bitemporal SPADE GAN with Building-Aware Metrics")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--compare_checkpoint", "--compare-checkpoint", dest="compare_checkpoint", type=str, default=None,
        help="Optional second checkpoint (e.g., Top-K) to run side-by-side comparison with the primary checkpoint."
    )
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument(
        "--masks_dir", "--masks-dir", dest="masks_dir", type=str, default=None,
        help="Optional path to disaster masks (defaults to {dataset_path}/{split}_masks or {dataset_path}/masks)"
    )
    parser.add_argument(
        "--change_threshold", "--change-threshold", dest="change_threshold", type=float, default=0.1,
        help="Threshold on normalized optical difference [0, 1] to construct binary optical change masks (default: 0.1)."
    )
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="eval_bitemporal_results")
    parser.add_argument(
        "--optical-mode", "--optical_mode", dest="optical_mode",
        choices=["s2_tci", "s2"], default="s2_tci",
    )
    parser.add_argument(
        "--residual-scale", "--residual_scale", dest="residual_scale", type=float, default=None,
        help="Must match training (default: 1 for TCI, 2 for full S2)",
    )
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--save_images", "--save-images", dest="save_images", action="store_true")
    parser.add_argument("--save_figures", "--save-figures", dest="save_figures", action="store_true")
    parser.add_argument("--max_figures", "--max-figures", dest="max_figures", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = {}
    if args.config and Path(args.config).is_file():
        with Path(args.config).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    dataset_path = args.dataset_path or cfg.get("data", {}).get("dataset_path")
    checkpoint = args.checkpoint or cfg.get("eval", {}).get("checkpoint")
    if not dataset_path or not checkpoint:
        raise ValueError("Both --dataset_path and --checkpoint are required.")

    channels = optical_channels(args.optical_mode)
    residual_scale = args.residual_scale
    if residual_scale is None:
        residual_scale = 2.0 if args.optical_mode == "s2" else 1.0
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve masks directory
    resolved_mask_dir = None
    if args.masks_dir and Path(args.masks_dir).is_dir():
        resolved_mask_dir = Path(args.masks_dir)
    else:
        for candidate_name in [f"{args.split}_masks", f"{args.split}_mask", "masks", "mask"]:
            cand = Path(dataset_path) / candidate_name
            if cand.is_dir():
                resolved_mask_dir = cand
                break

    print("=" * 78)
    print(" Bitemporal Change-Guided SPADE GAN Evaluation (Building-Aware Extension)")
    print("=" * 78)
    print(f"Device/Mode       : {device} / {args.optical_mode} ({channels} optical channels)")
    print(f"Normalization     : {'xBD-S12 per-band P1/P99' if args.optical_mode == 's2' else 'uint8 RGB'}")
    print(f"S1 Norm           : {'xBD-S12 per-channel P1/P99' if args.optical_mode == 's2' else 'legacy shared [-25,0]'}")
    print(f"Residual Scale    : {residual_scale}")
    print(f"Checkpoint        : {checkpoint}")
    if args.compare_checkpoint:
        print(f"Compare Checkpoint: {args.compare_checkpoint}")
    print(f"Dataset/Split     : {dataset_path} / {args.split}")
    print(f"Masks Directory   : {resolved_mask_dir or '[NONE DETECTED]'}")
    print(f"Change Threshold  : {args.change_threshold} (in [0, 1] unit range)")
    print(f"Output Dir        : {output_dir}")
    print("=" * 78)

    dataset = BitemporalS1S2Dataset(
        dataset_path=dataset_path, split=args.split, optical_mode=args.optical_mode, augment=False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    print(f"[DATA] Total complete samples: {len(dataset)}")

    # 1. Evaluate Primary Checkpoint
    primary_ckpt = Path(checkpoint)
    primary_label = "Naive" if "naive" in primary_ckpt.name.lower() or "naive" in str(primary_ckpt.parent).lower() else "Primary"
    results_primary, per_sample_primary = evaluate_single_checkpoint(
        checkpoint_path=primary_ckpt,
        dataset=dataset,
        loader=loader,
        mask_dir=resolved_mask_dir,
        channels=channels,
        residual_scale=residual_scale,
        device=device,
        optical_mode=args.optical_mode,
        change_threshold=args.change_threshold,
        output_dir=output_dir,
        save_images=args.save_images,
        save_figures=args.save_figures,
        max_figures=args.max_figures,
        model_label=primary_label,
    )

    results_primary["dataset_path"] = str(dataset_path)
    results_primary["split"] = args.split

    # Save primary results
    json_path = output_dir / "bitemporal_eval_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results_primary, handle, indent=2)

    csv_path = output_dir / "per_sample_building_metrics.csv"
    save_per_sample_csv(per_sample_primary, csv_path)

    # Save aggregate summary metrics table (CSV & JSON)
    agg_table = generate_aggregate_metrics_table(results_primary)
    agg_csv = output_dir / "aggregate_summary_metrics.csv"
    if agg_table:
        with agg_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(agg_table[0].keys()))
            writer.writeheader()
            writer.writerows(agg_table)
        with (output_dir / "aggregate_summary_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(agg_table, handle, indent=2)

    # Save per-band metrics summary table (CSV & JSON)
    if args.optical_mode == "s2" and "per_band_metrics" in results_primary:
        band_table = generate_per_band_table(results_primary, args.optical_mode)
        band_csv = output_dir / "per_band_summary.csv"
        if band_table:
            with band_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(band_table[0].keys()))
                writer.writeheader()
                writer.writerows(band_table)
            with (output_dir / "per_band_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(band_table, handle, indent=2)

    # Print summary
    print_evaluation_summary(results_primary)
    print(f"Summary JSON saved        : {json_path}")
    print(f"Per-sample CSV saved      : {csv_path}")
    if agg_table:
        print(f"Aggregate Summary saved   : {agg_csv}")
    if args.optical_mode == "s2":
        print(f"Per-band Summary saved    : {output_dir / 'per_band_summary.csv'}")

    # 2. Evaluate Optional Comparison Checkpoint
    if args.compare_checkpoint:
        comp_ckpt = Path(args.compare_checkpoint)
        comp_label = "Top-K" if ("k50" in comp_ckpt.name.lower() or "k50" in str(comp_ckpt.parent).lower() or "finetuned" in comp_ckpt.name.lower()) else "Comparison"
        results_comp, per_sample_comp = evaluate_single_checkpoint(
            checkpoint_path=comp_ckpt,
            dataset=dataset,
            loader=loader,
            mask_dir=resolved_mask_dir,
            channels=channels,
            residual_scale=residual_scale,
            device=device,
            optical_mode=args.optical_mode,
            change_threshold=args.change_threshold,
            output_dir=output_dir,
            save_images=args.save_images,
            save_figures=args.save_figures,
            max_figures=args.max_figures,
            model_label=comp_label,
        )

        results_comp["dataset_path"] = str(dataset_path)
        results_comp["split"] = args.split

        comp_json_path = output_dir / f"bitemporal_eval_summary_{comp_label.lower()}.json"
        with comp_json_path.open("w", encoding="utf-8") as handle:
            json.dump(results_comp, handle, indent=2)

        comp_csv_path = output_dir / f"per_sample_building_metrics_{comp_label.lower()}.csv"
        save_per_sample_csv(per_sample_comp, comp_csv_path)

        # Save comparison aggregate and per-band summaries
        agg_table_comp = generate_aggregate_metrics_table(results_comp)
        if agg_table_comp:
            with (output_dir / f"aggregate_summary_metrics_{comp_label.lower()}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(agg_table_comp[0].keys()))
                writer.writeheader()
                writer.writerows(agg_table_comp)
            with (output_dir / f"aggregate_summary_metrics_{comp_label.lower()}.json").open("w", encoding="utf-8") as handle:
                json.dump(agg_table_comp, handle, indent=2)

        if args.optical_mode == "s2" and "per_band_metrics" in results_comp:
            band_table_comp = generate_per_band_table(results_comp, args.optical_mode)
            if band_table_comp:
                with (output_dir / f"per_band_summary_{comp_label.lower()}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(band_table_comp[0].keys()))
                    writer.writeheader()
                    writer.writerows(band_table_comp)
                with (output_dir / f"per_band_summary_{comp_label.lower()}.json").open("w", encoding="utf-8") as handle:
                    json.dump(band_table_comp, handle, indent=2)

        table_str, table_records = generate_comparison_table(results_primary, results_comp)
        print(table_str)

        comparison_payload = {
            "primary_model": primary_label,
            "primary_checkpoint": str(primary_ckpt),
            "comparison_model": comp_label,
            "comparison_checkpoint": str(comp_ckpt),
            "change_threshold": args.change_threshold,
            "num_samples_evaluated": results_primary["num_samples_evaluated"],
            "num_samples_valid_building": results_primary["num_samples_valid_building"],
            "num_samples_valid_damaged_building": results_primary["num_samples_valid_damaged_building"],
            "comparison_records": table_records,
        }
        with (output_dir / "naive_vs_topk_comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison_payload, handle, indent=2)

        with (output_dir / "naive_vs_topk_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_records[0].keys()))
            writer.writeheader()
            writer.writerows(table_records)

        print(f"Comparison JSON saved     : {output_dir / 'naive_vs_topk_comparison.json'}")
        print(f"Comparison CSV saved      : {output_dir / 'naive_vs_topk_comparison.csv'}")

    print("=" * 78)


if __name__ == "__main__":
    main()