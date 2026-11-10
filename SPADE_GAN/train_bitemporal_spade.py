"""Training script for Bitemporal Change-Guided SPADE GAN.

Inputs: [Pre_S1 (2ch), Pre_S2 (3-channel TCI or 12-band S2), Post_S1 (2ch)]
Output: Post_S2 with the same optical modality as Pre_S2

Features:
  - Cross-modal grounding between Pre_S1 and Pre_S2
  - Explicit differential SAR change computation (ΔS1 = Post_S1 - Pre_S1)
  - Change-guided SPADE modulation
  - Anti-Identity Change Consistency Loss (prevents copying Pre_S2)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

from models.bitemporal_change_spade import BitemporalChangeGenerator, MultiScaleChangeDiscriminator
from datasets.bitemporal_s1_s2_dataset import BitemporalS1S2Dataset
from utils.optical import (
    normalized_s2_to_reflectance,
    optical_channels,
    optical_to_rgb01,
    select_rgb,
)


class VGGLoss(nn.Module):
    """VGG-19 perceptual feature matching loss."""

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).eval()
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).eval()
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 18)]).eval()
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(18, 27)]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Scale from [-1, 1] to ImageNet normalized
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x * 0.5 + 0.5 - mean) / std
        y = (y * 0.5 + 0.5 - mean) / std

        loss = 0.0
        for block in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


class ColorLoss(nn.Module):
    """Enforces regional color matching while preventing high-frequency neon blooming."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        fake_blurred = self.pool(fake_img)
        real_blurred = self.pool(real_img)
        # L1 strictly penalizes extreme pixel intensities, eliminating glowing halos
        return F.l1_loss(fake_blurred, real_blurred)


class TVLoss(nn.Module):
    """Total Variation Regularization Loss on residual delta (ΔY). Enforces compact boundaries."""

    def __init__(self):
        super().__init__()

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])
        diff_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()


class SpectralAngleLoss(nn.Module):
    """Spectral-angle mapper loss on denormalized 12-band reflectance vectors."""

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_ref = normalized_s2_to_reflectance(fake)
        real_ref = normalized_s2_to_reflectance(real)
        numerator = (fake_ref * real_ref).sum(dim=1)
        denominator = torch.linalg.vector_norm(fake_ref, dim=1) * torch.linalg.vector_norm(real_ref, dim=1)
        cosine = (numerator / denominator.clamp_min(1e-8)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cosine).mean()


def spectral_angle_degrees(fake: torch.Tensor, real: torch.Tensor) -> float:
    return float(torch.rad2deg(SpectralAngleLoss()(fake, real)).item())


def calc_psnr_ssim_pt(fake: torch.Tensor, real: torch.Tensor) -> Tuple[float, float]:
    """Calculates PSNR (dB) and SSIM natively on GPU tensors in [-1, 1]."""
    f = torch.clamp((fake + 1.0) / 2.0, 0.0, 1.0)
    r = torch.clamp((real + 1.0) / 2.0, 0.0, 1.0)

    # 1. PSNR
    mse = F.mse_loss(f, r).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))

    # 2. Fast GPU SSIM
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.avg_pool2d(f, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(r, kernel_size=11, stride=1, padding=5)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(f * f, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(r * r, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(f * r, kernel_size=11, stride=1, padding=5) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    ssim = ssim_map.mean().item()
    return float(psnr), float(ssim)


def save_comparison_grid(
    save_path: Path,
    pre_s2: torch.Tensor,
    post_s1: torch.Tensor,
    real_post_s2: torch.Tensor,
    fake_post_s2: torch.Tensor,
    epoch: int,
    optical_mode: str,
):
    """Saves a 4-panel visual comparison: Pre Optical, Post SAR, GT Post, Fake Post."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def denorm_rgb(t):
        return optical_to_rgb01(t.detach().cpu(), optical_mode).permute(1, 2, 0).numpy()

    def denorm_sar(t):
        arr = (t.detach().cpu().squeeze().permute(1, 2, 0).numpy()[..., 0] + 1.0) / 2.0
        return np.clip(arr, 0, 1)

    axes[0].imshow(denorm_rgb(pre_s2[0]))
    axes[0].set_title("Input: Pre-S2 Optical")
    axes[0].axis("off")

    axes[1].imshow(denorm_sar(post_s1[0]), cmap="gray")
    axes[1].set_title("Input: Post-S1 SAR (VV)")
    axes[1].axis("off")

    axes[2].imshow(denorm_rgb(real_post_s2[0]))
    axes[2].set_title("Target: Real Post-S2 Optical")
    axes[2].axis("off")

    axes[3].imshow(denorm_rgb(fake_post_s2[0]))
    axes[3].set_title(f"Synthesized Post-S2 (Epoch {epoch})")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: list, output_dir: Path, optical_mode: str):
    """Generates and saves publication-quality loss curves, learning curves, and paper summary plots."""
    if not history:
        return

    df = pd.DataFrame(history)
    epochs = df["epoch"].values

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. LOSS CURVES (4-panel publication plot)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # Panel (a): Overall Objectives
    ax = axes[0, 0]
    ax.plot(epochs, df["loss_g"], label="Total Generator ($L_G$)", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, df["loss_d"], label="Discriminator ($L_D$)", color="#d62728", linewidth=2.0, linestyle="--")
    ax.set_title("(a) Overall Adversarial Objectives", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (b): Reconstruction Losses
    ax = axes[0, 1]
    ax.plot(epochs, df["loss_l1"], label="L1 Optical ($L_1$)", color="#2ca02c", linewidth=2.0)
    ax.plot(epochs, df["loss_vgg"], label="Perceptual RGB ($L_{VGG}$)", color="#ff7f0e", linewidth=2.0)
    if optical_mode == "s2" and "loss_sam" in df and df["loss_sam"].max() > 0:
        ax.plot(epochs, df["loss_sam"], label="Spectral Angle ($L_{SAM}$)", color="#9467bd", linewidth=2.0)
    ax.set_title("(b) Fidelity & Reconstruction Losses", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Weighted Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (c): Regularization & Consistency
    ax = axes[1, 0]
    ax.plot(epochs, df["loss_change"], label="Change Consistency ($L_{change}$)", color="#8c564b", linewidth=2.0)
    ax.plot(epochs, df["loss_color"], label="Regional Color ($L_{color}$)", color="#e377c2", linewidth=2.0)
    ax.plot(epochs, df["loss_tv"], label="Total Variation ($L_{TV}$)", color="#7f7f7f", linewidth=2.0)
    ax.set_title("(c) Regularization & Change Consistency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Weighted Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (d): Adversarial Convergence & Learning Rate
    ax = axes[1, 1]
    color_adv = "#bcbd22"
    ax.plot(epochs, df["loss_adv"], label="Generator Adv ($L_{adv}$)", color=color_adv, linewidth=2.0)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Adversarial Loss", color=color_adv, fontsize=11)
    ax.tick_params(axis="y", labelcolor=color_adv)
    ax.grid(True, linestyle="--", alpha=0.5)

    if "lr_g" in df:
        ax2 = ax.twinx()
        color_lr = "#17becf"
        ax2.plot(epochs, df["lr_g"], label="Learning Rate ($lr_G$)", color=color_lr, linestyle=":", linewidth=2.0)
        ax2.set_ylabel("Learning Rate", color=color_lr, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_lr)

    ax.set_title("(d) Adversarial Convergence & Learning Rate", fontsize=12, fontweight="bold")

    plt.suptitle("Bitemporal SPADE GAN: Training Loss Dynamics", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_dir / "loss_curves.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "loss_curves.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "loss_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. LEARNING CURVES (Validation Metrics: PSNR, SSIM, LPIPS, SAM)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # Panel (a): PSNR
    ax = axes[0, 0]
    ax.plot(epochs, df["all_band_psnr_db"], label="All-Band PSNR", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, df["rgb_psnr_db"], label="RGB PSNR", color="#aec7e8", linewidth=1.8, linestyle="--")
    best_p_idx = df["all_band_psnr_db"].idxmax()
    best_p_epoch = df.loc[best_p_idx, "epoch"]
    best_p_val = df.loc[best_p_idx, "all_band_psnr_db"]
    ax.scatter([best_p_epoch], [best_p_val], color="#d62728", s=80, zorder=5, label=f"Peak PSNR: {best_p_val:.2f} dB (Ep {int(best_p_epoch)})")
    ax.set_title("(a) Peak Signal-to-Noise Ratio (PSNR)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (b): SSIM
    ax = axes[0, 1]
    ax.plot(epochs, df["all_band_ssim"], label="All-Band SSIM", color="#2ca02c", linewidth=2.0)
    ax.plot(epochs, df["rgb_ssim"], label="RGB SSIM", color="#98df8a", linewidth=1.8, linestyle="--")
    best_s_idx = df["all_band_ssim"].idxmax()
    best_s_epoch = df.loc[best_s_idx, "epoch"]
    best_s_val = df.loc[best_s_idx, "all_band_ssim"]
    ax.scatter([best_s_epoch], [best_s_val], color="#d62728", s=80, zorder=5, label=f"Peak SSIM: {best_s_val:.4f} (Ep {int(best_s_epoch)})")
    ax.set_title("(b) Structural Similarity Index (SSIM)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("SSIM", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (c): Perceptual Distance (LPIPS - Lower is better)
    ax = axes[1, 0]
    ax.plot(epochs, df["rgb_lpips"], label="RGB LPIPS Distance", color="#ff7f0e", linewidth=2.0)
    best_l_idx = df["rgb_lpips"].idxmin()
    best_l_epoch = df.loc[best_l_idx, "epoch"]
    best_l_val = df.loc[best_l_idx, "rgb_lpips"]
    ax.scatter([best_l_epoch], [best_l_val], color="#2ca02c", s=80, zorder=5, label=f"Best LPIPS: {best_l_val:.4f} (Ep {int(best_l_epoch)})")
    ax.set_title("(c) Perceptual Metric (LPIPS, Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("LPIPS Score", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Panel (d): Spectral Angle (SAM) & Composite Score
    ax = axes[1, 1]
    has_sam = optical_mode == "s2" and "sam_degrees" in df and df["sam_degrees"].notna().any()
    if has_sam:
        color_sam = "#9467bd"
        ax.plot(epochs, df["sam_degrees"], label="SAM Angle (°)", color=color_sam, linewidth=2.0)
        best_sam_idx = df["sam_degrees"].idxmin()
        best_sam_epoch = df.loc[best_sam_idx, "epoch"]
        best_sam_val = df.loc[best_sam_idx, "sam_degrees"]
        ax.scatter([best_sam_epoch], [best_sam_val], color="#d62728", s=80, zorder=5, label=f"Min SAM: {best_sam_val:.2f}° (Ep {int(best_sam_epoch)})")
        ax.set_ylabel("Spectral Angle (degrees, lower is better)", color=color_sam, fontsize=11)
        ax.tick_params(axis="y", labelcolor=color_sam)
        ax.grid(True, linestyle="--", alpha=0.5)

        if "overall_score" in df:
            ax2 = ax.twinx()
            color_sc = "#8c564b"
            ax2.plot(epochs, df["overall_score"], label="Composite Score", color=color_sc, linestyle="--", linewidth=1.8)
            ax2.set_ylabel("Composite Quality Score", color=color_sc, fontsize=11)
            ax2.tick_params(axis="y", labelcolor=color_sc)
        ax.set_title("(d) Spectral Fidelity (SAM) & Composite Score", fontsize=12, fontweight="bold")
    else:
        ax.plot(epochs, df["overall_score"], label="Composite Score", color="#8c564b", linewidth=2.0)
        best_sc_idx = df["overall_score"].idxmax()
        best_sc_epoch = df.loc[best_sc_idx, "epoch"]
        best_sc_val = df.loc[best_sc_idx, "overall_score"]
        ax.scatter([best_sc_epoch], [best_sc_val], color="#2ca02c", s=80, zorder=5, label=f"Best Composite: {best_sc_val:.2f} (Ep {int(best_sc_epoch)})")
        ax.set_title("(d) Validation Composite Quality Score", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Overall Quality Score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10)

    plt.suptitle("Bitemporal SPADE GAN: Validation Learning Curves", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_dir / "learning_curves.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "learning_curves.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "learning_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. TRAINING SUMMARY CURVES (6-panel composite overview for paper)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # (1) Total G & D
    axes[0, 0].plot(epochs, df["loss_g"], label="Loss G", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, df["loss_d"], label="Loss D", color="#d62728", lw=2, ls="--")
    axes[0, 0].set_title("Generator & Discriminator Losses", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, ls="--", alpha=0.5)
    axes[0, 0].legend()

    # (2) L1 & Perceptual
    axes[0, 1].plot(epochs, df["loss_l1"], label="L1 Loss", color="#2ca02c", lw=2)
    axes[0, 1].plot(epochs, df["loss_vgg"], label="VGG Loss", color="#ff7f0e", lw=2)
    axes[0, 1].set_title("Reconstruction Losses", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, ls="--", alpha=0.5)
    axes[0, 1].legend()

    # (3) Change Consistency & Regularization
    axes[0, 2].plot(epochs, df["loss_change"], label="Change Loss", color="#8c564b", lw=2)
    axes[0, 2].plot(epochs, df["loss_color"], label="Color Loss", color="#e377c2", lw=2)
    axes[0, 2].plot(epochs, df["loss_tv"], label="TV Loss", color="#7f7f7f", lw=2)
    axes[0, 2].set_title("Change & Regularization Losses", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].grid(True, ls="--", alpha=0.5)
    axes[0, 2].legend()

    # (4) PSNR
    axes[1, 0].plot(epochs, df["all_band_psnr_db"], label="All-Band", color="#1f77b4", lw=2)
    axes[1, 0].plot(epochs, df["rgb_psnr_db"], label="RGB", color="#aec7e8", lw=1.8, ls="--")
    axes[1, 0].set_title("Validation PSNR (dB)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("dB")
    axes[1, 0].grid(True, ls="--", alpha=0.5)
    axes[1, 0].legend()

    # (5) SSIM
    axes[1, 1].plot(epochs, df["all_band_ssim"], label="All-Band", color="#2ca02c", lw=2)
    axes[1, 1].plot(epochs, df["rgb_ssim"], label="RGB", color="#98df8a", lw=1.8, ls="--")
    axes[1, 1].set_title("Validation SSIM", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].grid(True, ls="--", alpha=0.5)
    axes[1, 1].legend()

    # (6) LPIPS & SAM
    axes[1, 2].plot(epochs, df["rgb_lpips"], label="LPIPS (↓)", color="#ff7f0e", lw=2)
    if has_sam:
        axes[1, 2].plot(epochs, df["sam_degrees"] / 10.0, label="SAM/10 (↓)", color="#9467bd", lw=2, ls="--")
    axes[1, 2].set_title("Perceptual / Spectral Error", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Error Metric")
    axes[1, 2].grid(True, ls="--", alpha=0.5)
    axes[1, 2].legend()

    plt.suptitle("Bitemporal Change-Guided SPADE GAN: Paper Summary Curves", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_dir / "training_summary_curves.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "training_summary_curves.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "training_summary_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def safe_torch_save(obj, path, retries: int = 5, delay: float = 0.5):
    """Save PyTorch tensor/model/state robustly against Windows file locking (Error 1224)."""
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            if path.exists():
                try:
                    os.replace(tmp_path, path)
                except OSError:
                    time.sleep(delay)
                    os.replace(tmp_path, path)
            else:
                os.replace(tmp_path, path)
            return
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                try:
                    torch.save(obj, path)
                except Exception as final_e:
                    print(f"[WARN] Non-fatal save failure for {path.name}: {final_e}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Train Bitemporal Change-Guided SPADE GAN")
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="checkpoints/bitemporal_spade", help="Output directory")
    parser.add_argument(
        "--optical-mode", "--optical_mode", dest="optical_mode",
        choices=["s2_tci", "s2"], default="s2_tci",
        help="Optical modality: 3-channel TCI or complete 12-band Sentinel-2",
    )
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--decay_epoch", "--decay-epoch", dest="decay_epoch", type=int, default=25, help="Epoch to start LR linear decay")
    parser.add_argument("--lr_g", "--lr-g", dest="lr_g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", "--lr-d", dest="lr_d", type=float, default=4e-4, help="Discriminator learning rate")
    parser.add_argument("--lambda_l1", "--lambda-l1", dest="lambda_l1", type=float, default=10.0, help="L1 optical loss weight")
    parser.add_argument("--lambda_vgg", "--lambda-vgg", dest="lambda_vgg", type=float, default=10.0, help="VGG perceptual loss weight")
    parser.add_argument("--lambda_change", "--lambda-change", dest="lambda_change", type=float, default=5.0, help="Anti-identity change consistency weight")
    parser.add_argument("--lambda_color", "--lambda-color", dest="lambda_color", type=float, default=1.0, help="Regional low-frequency color loss weight")
    parser.add_argument("--lambda_tv", "--lambda-tv", dest="lambda_tv", type=float, default=0.1, help="Total variation regularization weight on delta optical")
    parser.add_argument(
        "--lambda-sam", "--lambda_sam", dest="lambda_sam", type=float, default=1.0,
        help="12-band spectral-angle loss weight; ignored for s2_tci",
    )
    parser.add_argument(
        "--residual-scale", "--residual_scale", dest="residual_scale", type=float, default=None,
        help="Maximum normalized residual magnitude (default: 1 for TCI, 2 for full S2)",
    )
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--checkpoint-interval", "--checkpoint_interval", dest="checkpoint_interval",
        type=int, default=5, help="Save a numbered training-state checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from latest_training_state.pth or a numbered training-state checkpoint",
    )
    parser.add_argument(
        "--init-generator", "--pretrained_generator", dest="init_generator", type=str, default=None,
        help="Initialize only the generator from a compatible checkpoint, then start a new run",
    )

    args = parser.parse_args()

    if args.resume and args.init_generator:
        raise ValueError("Use either --resume or --init-generator, not both.")

    channels = optical_channels(args.optical_mode)
    if args.residual_scale is None:
        # A normalized full-S2 pre/post difference spans [-2, 2].
        args.residual_scale = 2.0 if args.optical_mode == "s2" else 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Bitemporal Change-Guided SPADE GAN Training ")
    print(
        f" Inputs: Pre_S1 (2ch) + Pre_S2 ({channels}ch) + Post_S1 (2ch) "
        f"-> Output: Post_S2 ({channels}ch)"
    )
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Dataset Path  : {args.dataset_path}")
    print(f"Optical Mode  : {args.optical_mode} ({channels} channels)")
    print(f"Normalization : {'xBD-S12 per-band P1/P99' if args.optical_mode == 's2' else 'uint8 RGB to [-1,1]'}")
    print(f"S1 Norm       : {'xBD-S12 per-channel P1/P99' if args.optical_mode == 's2' else 'legacy shared [-25,0]'}")
    print(f"Residual Scale: {args.residual_scale}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs} (Linear decay starting at epoch {args.decay_epoch})")
    print(f"Output Dir    : {output_dir}")
    print(f"Weights       : L1={args.lambda_l1}, RGB-VGG={args.lambda_vgg}, Change={args.lambda_change}, LowFreq={args.lambda_color}, TV={args.lambda_tv}, SAM={args.lambda_sam if args.optical_mode == 's2' else 0.0}")
    print("=" * 70)

    # 1. Datasets
    train_dataset = BitemporalS1S2Dataset(
        dataset_path=args.dataset_path, split="train", optical_mode=args.optical_mode, augment=True
    )
    val_dataset = BitemporalS1S2Dataset(
        dataset_path=args.dataset_path, split="val", optical_mode=args.optical_mode, augment=False
    )

    print(f"[DATA] Matched Training samples  : {len(train_dataset)}")
    print(f"[DATA] Matched Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("[ERROR] No matched quadruplet samples found! Check dataset directory.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # 2. Models & Loss Functions
    generator = BitemporalChangeGenerator(
        ngf=64, optical_channels=channels, residual_scale=args.residual_scale
    ).to(device)
    discriminator = MultiScaleChangeDiscriminator(
        ndf=64, num_scales=2, optical_channels=channels
    ).to(device)
    if args.init_generator:
        try:
            initial = torch.load(args.init_generator, map_location=device, weights_only=False)
        except TypeError:
            initial = torch.load(args.init_generator, map_location=device)
        if isinstance(initial, dict) and "generator" in initial:
            saved_mode = initial.get("optical_mode")
            if saved_mode and saved_mode != args.optical_mode:
                raise ValueError(
                    f"Initialization checkpoint is {saved_mode!r}; command requested {args.optical_mode!r}."
                )
            initial = initial["generator"]
        generator.load_state_dict(initial, strict=True)
        print(f"[INIT] Loaded generator with exact architecture match: {args.init_generator}")
    vgg_loss_fn = VGGLoss().to(device)
    color_loss_fn = ColorLoss(kernel_size=7).to(device)
    tv_loss_fn = TVLoss().to(device)
    sam_loss_fn = SpectralAngleLoss().to(device) if args.optical_mode == "s2" else None

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    # 3. Optimizers (TTUR: Two Time-Scale Update Rule)
    optimizer_g = Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.999))

    # 4. Schedulers
    def lr_lambda(epoch):
        if epoch < args.decay_epoch:
            return 1.0
        return 1.0 - (epoch - args.decay_epoch) / max(args.epochs - args.decay_epoch, 1)

    scheduler_g = LambdaLR(optimizer_g, lr_lambda=lr_lambda)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda=lr_lambda)

    start_epoch = 1
    best_psnr = -float("inf")
    best_ssim = -float("inf")
    best_lpips = float("inf")
    best_sam = float("inf")
    best_overall = -float("inf")
    if args.resume:
        try:
            state = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(args.resume, map_location=device)
        if not isinstance(state, dict) or "generator" not in state:
            raise ValueError("--resume requires a training-state checkpoint, not a generator-only .pth file.")
        saved_mode = state.get("optical_mode", args.optical_mode)
        if saved_mode != args.optical_mode:
            raise ValueError(f"Resume checkpoint is {saved_mode!r}, command requested {args.optical_mode!r}.")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        scheduler_g.load_state_dict(state["scheduler_g"])
        scheduler_d.load_state_dict(state["scheduler_d"])
        start_epoch = int(state["epoch"]) + 1
        best_psnr = state.get("best_psnr", best_psnr)
        best_ssim = state.get("best_ssim", best_ssim)
        best_lpips = state.get("best_lpips", best_lpips)
        best_sam = state.get("best_sam", best_sam)
        best_overall = state.get("best_overall", best_overall)
        print(f"[RESUME] Continuing after epoch {state['epoch']} from {args.resume}")

    # Training history tracking for publication curves
    history_csv = output_dir / "training_history.csv"
    history_records = []
    if args.resume and history_csv.exists():
        try:
            history_records = pd.read_csv(history_csv).to_dict(orient="records")
            print(f"[RESUME] Loaded {len(history_records)} existing epochs from {history_csv}")
        except Exception as e:
            print(f"[WARN] Could not load prior training_history.csv: {e}")

    # Initialize LPIPS perceptual metric
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        print("[METRIC] Official LPIPS (AlexNet) loaded successfully.")
    except Exception:
        lpips_fn = None
        print("[METRIC] Note: 'lpips' package not found, using normalized VGG perceptual distance as LPIPS proxy. (pip install lpips to enable official AlexNet)")

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        generator.train()
        discriminator.train()

        total_loss_g, total_loss_d = 0.0, 0.0
        total_loss_adv, total_loss_l1 = 0.0, 0.0
        total_loss_vgg, total_loss_change = 0.0, 0.0
        total_loss_color, total_loss_tv = 0.0, 0.0
        total_loss_sam = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=True)
        for batch in pbar:
            pre_s1 = batch["pre_s1"].to(device, non_blocking=True)
            pre_s2 = batch["pre_s2"].to(device, non_blocking=True)
            post_s1 = batch["post_s1"].to(device, non_blocking=True)
            real_post_s2 = batch["post_s2"].to(device, non_blocking=True)

            # ------------------------------------
            # Train Discriminator
            # ------------------------------------
            optimizer_d.zero_grad()
            with torch.no_grad():
                fake_post_s2 = generator(pre_s1, pre_s2, post_s1)

            # Real predictions
            d_real_outs = discriminator(pre_s1, pre_s2, post_s1, real_post_s2)
            # Fake predictions
            d_fake_outs = discriminator(pre_s1, pre_s2, post_s1, fake_post_s2.detach())

            loss_d = 0.0
            for pred_real, pred_fake in zip(d_real_outs, d_fake_outs):
                loss_d += F.relu(1.0 - pred_real).mean() + F.relu(1.0 + pred_fake).mean()

            loss_d.backward()
            optimizer_d.step()
            total_loss_d += loss_d.item()

            # ------------------------------------
            # Train Generator
            # ------------------------------------
            optimizer_g.zero_grad()
            fake_post_s2, delta_optical = generator(pre_s1, pre_s2, post_s1, return_delta=True)

            # Adversarial Loss
            d_fake_for_g = discriminator(pre_s1, pre_s2, post_s1, fake_post_s2)
            loss_g_adv = 0.0
            for pred_fake in d_fake_for_g:
                loss_g_adv += -pred_fake.mean()

            # L1 Reconstruction Loss
            loss_l1 = F.l1_loss(fake_post_s2, real_post_s2) * args.lambda_l1

            # VGG is an RGB network.  For full S2, use B4/B3/B2 only; never feed
            # all 12 spectral bands into an ImageNet feature extractor.
            fake_rgb = select_rgb(fake_post_s2, args.optical_mode)
            real_rgb = select_rgb(real_post_s2, args.optical_mode)
            loss_vgg = vgg_loss_fn(fake_rgb, real_rgb) * args.lambda_vgg

            # Anti-Identity Change Consistency Loss:
            # Ensures optical change matches ground-truth change
            diff_fake_opt = torch.abs(fake_post_s2 - pre_s2)
            diff_real_opt = torch.abs(real_post_s2 - pre_s2)
            loss_change = F.l1_loss(diff_fake_opt, diff_real_opt) * args.lambda_change

            # Regional Color Matching Loss (prevents high-frequency neon blooming)
            loss_color = color_loss_fn(fake_post_s2, real_post_s2) * args.lambda_color

            # Total Variation Regularization on Delta Y (prevents diffuse circular splats)
            loss_tv = tv_loss_fn(delta_optical) * args.lambda_tv

            # Spectral geometry is meaningful only for the 12-band product.
            loss_sam = (
                sam_loss_fn(fake_post_s2, real_post_s2) * args.lambda_sam
                if sam_loss_fn is not None else fake_post_s2.new_zeros(())
            )

            loss_g = loss_g_adv + loss_l1 + loss_vgg + loss_change + loss_color + loss_tv + loss_sam
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
            total_loss_adv += loss_g_adv.item()
            total_loss_l1 += loss_l1.item()
            total_loss_vgg += loss_vgg.item()
            total_loss_change += loss_change.item()
            total_loss_color += loss_color.item()
            total_loss_tv += loss_tv.item()
            total_loss_sam += (loss_sam.item() if isinstance(loss_sam, torch.Tensor) and loss_sam.numel() == 1 else 0.0)

            pbar.set_postfix({
                "G": f"{loss_g.item():.2f}",
                "D": f"{loss_d.item():.2f}",
                "L1": f"{loss_l1.item():.2f}",
                "Chg": f"{loss_change.item():.2f}",
                "Clr": f"{loss_color.item():.2f}",
                "TV": f"{loss_tv.item():.2f}",
                "SAM": f"{loss_sam.item():.2f}",
            })

        scheduler_g.step()
        scheduler_d.step()
        elapsed = time.time() - t0

        avg_loss_g = total_loss_g / max(num_batches, 1)
        avg_loss_d = total_loss_d / max(num_batches, 1)
        avg_loss_adv = total_loss_adv / max(num_batches, 1)
        avg_loss_l1 = total_loss_l1 / max(num_batches, 1)
        avg_loss_vgg = total_loss_vgg / max(num_batches, 1)
        avg_loss_change = total_loss_change / max(num_batches, 1)
        avg_loss_color = total_loss_color / max(num_batches, 1)
        avg_loss_tv = total_loss_tv / max(num_batches, 1)
        avg_loss_sam = total_loss_sam / max(num_batches, 1)

        # ------------------------------------
        # Full validation set: all-band fidelity plus RGB perceptual fidelity.
        # ------------------------------------
        generator.eval()
        val_psnr_list, val_ssim_list = [], []
        val_rgb_psnr_list, val_rgb_ssim_list, val_lpips_list = [], [], []
        val_sam_list = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val   [{epoch}/{args.epochs}]", leave=True)
            for v_batch in val_pbar:
                v_pre_s1 = v_batch["pre_s1"].to(device, non_blocking=True)
                v_pre_s2 = v_batch["pre_s2"].to(device, non_blocking=True)
                v_post_s1 = v_batch["post_s1"].to(device, non_blocking=True)
                v_real_post_s2 = v_batch["post_s2"].to(device, non_blocking=True)
                v_fake = generator(v_pre_s1, v_pre_s2, v_post_s1)
                v_fake_rgb = select_rgb(v_fake, args.optical_mode)
                v_real_rgb = select_rgb(v_real_post_s2, args.optical_mode)

                for sample_idx in range(v_fake.shape[0]):
                    p, s = calc_psnr_ssim_pt(
                        v_fake[sample_idx:sample_idx + 1], v_real_post_s2[sample_idx:sample_idx + 1]
                    )
                    rgb_p, rgb_s = calc_psnr_ssim_pt(
                        v_fake_rgb[sample_idx:sample_idx + 1], v_real_rgb[sample_idx:sample_idx + 1]
                    )
                    val_psnr_list.append(p)
                    val_ssim_list.append(s)
                    val_rgb_psnr_list.append(rgb_p)
                    val_rgb_ssim_list.append(rgb_s)
                    if sam_loss_fn is not None:
                        val_sam_list.append(spectral_angle_degrees(
                            v_fake[sample_idx:sample_idx + 1],
                            v_real_post_s2[sample_idx:sample_idx + 1],
                        ))

                if lpips_fn is not None:
                    lp_batch = lpips_fn(v_fake_rgb, v_real_rgb).flatten().detach().cpu().tolist()
                else:
                    # Proxy is batch-level and is used only when LPIPS is unavailable.
                    proxy = vgg_loss_fn(v_fake_rgb, v_real_rgb).item() / 15.0
                    lp_batch = [proxy] * v_fake.shape[0]
                val_lpips_list.extend(lp_batch)
                val_pbar.set_postfix({
                    "PSNR": f"{np.mean(val_psnr_list):.2f}",
                    "RGB": f"{np.mean(val_rgb_psnr_list):.2f}",
                    "LPIPS": f"{np.mean(val_lpips_list):.3f}",
                })

        mean_val_psnr = float(np.mean(val_psnr_list))
        mean_val_ssim = float(np.mean(val_ssim_list))
        mean_val_rgb_psnr = float(np.mean(val_rgb_psnr_list))
        mean_val_rgb_ssim = float(np.mean(val_rgb_ssim_list))
        mean_val_lpips = float(np.mean(val_lpips_list))
        mean_val_sam = float(np.mean(val_sam_list)) if val_sam_list else None

        # The overall score uses full-band PSNR/SSIM, RGB LPIPS, and (for S2)
        # a mild spectral-angle penalty.  Individual best checkpoints are also
        # saved so downstream damage mapping can select empirically.
        overall_score = (mean_val_psnr * mean_val_ssim) / (mean_val_lpips + 0.05)
        if mean_val_sam is not None:
            overall_score /= 1.0 + mean_val_sam / 10.0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss_G: {avg_loss_g:.4f} | Loss_D: {avg_loss_d:.4f} | "
            f"All PSNR/SSIM: {mean_val_psnr:.2f}/{mean_val_ssim:.4f} | "
            f"RGB PSNR/SSIM: {mean_val_rgb_psnr:.2f}/{mean_val_rgb_ssim:.4f} | "
            f"LPIPS: {mean_val_lpips:.4f}"
            + (f" | SAM: {mean_val_sam:.2f} deg" if mean_val_sam is not None else "")
        )

        # Save visual comparison sample every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                val_sample = val_dataset[0]
                v_pre_s1 = val_sample["pre_s1"].unsqueeze(0).to(device)
                v_pre_s2 = val_sample["pre_s2"].unsqueeze(0).to(device)
                v_post_s1 = val_sample["post_s1"].unsqueeze(0).to(device)
                v_real_post_s2 = val_sample["post_s2"].unsqueeze(0).to(device)
                v_fake_post_s2 = generator(v_pre_s1, v_pre_s2, v_post_s1)

                sample_path = sample_dir / f"epoch_{epoch:03d}_comparison.png"
                save_comparison_grid(
                    sample_path, v_pre_s2, v_post_s1, v_real_post_s2,
                    v_fake_post_s2, epoch, args.optical_mode
                )
                print(f"[*] Visual sample saved to: {sample_path}")

        # 1. Latest checkpoint
        safe_torch_save(generator.state_dict(), output_dir / "latest_bitemporal_spade_generator.pth")

        # 2. Best PSNR checkpoint
        if mean_val_psnr > best_psnr:
            best_psnr = mean_val_psnr
            safe_torch_save(generator.state_dict(), output_dir / "best_psnr_generator.pth")
            print(f"[*] New Best PSNR Checkpoint ({best_psnr:.2f} dB) -> {output_dir / 'best_psnr_generator.pth'}")

        # 3. Best SSIM checkpoint
        if mean_val_ssim > best_ssim:
            best_ssim = mean_val_ssim
            safe_torch_save(generator.state_dict(), output_dir / "best_ssim_generator.pth")
            print(f"[*] New Best SSIM Checkpoint ({best_ssim:.4f}) -> {output_dir / 'best_ssim_generator.pth'}")

        # 4. Best LPIPS checkpoint (Lower is better)
        if mean_val_lpips < best_lpips:
            best_lpips = mean_val_lpips
            safe_torch_save(generator.state_dict(), output_dir / "best_lpips_generator.pth")
            print(f"[*] New Best LPIPS Checkpoint ({best_lpips:.4f}) -> {output_dir / 'best_lpips_generator.pth'}")

        # 5. Best spectral-angle checkpoint for complete S2.
        if mean_val_sam is not None and mean_val_sam < best_sam:
            best_sam = mean_val_sam
            safe_torch_save(generator.state_dict(), output_dir / "best_sam_generator.pth")
            print(f"[*] New Best SAM Checkpoint ({best_sam:.2f} deg) -> {output_dir / 'best_sam_generator.pth'}")

        # 6. Best Overall (Composite) checkpoint
        if overall_score > best_overall:
            best_overall = overall_score
            safe_torch_save(generator.state_dict(), output_dir / "best_overall_generator.pth")
            safe_torch_save(generator.state_dict(), output_dir / "best_bitemporal_spade_generator.pth")
            print(f"[*] New Best Overall Checkpoint (Score: {best_overall:.2f}) -> {output_dir / 'best_overall_generator.pth'}")

        current_lr_g = float(optimizer_g.param_groups[0]["lr"])
        current_lr_d = float(optimizer_d.param_groups[0]["lr"])

        epoch_record = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 1),
            "lr_g": current_lr_g,
            "lr_d": current_lr_d,
            "loss_g": avg_loss_g,
            "loss_d": avg_loss_d,
            "loss_adv": avg_loss_adv,
            "loss_l1": avg_loss_l1,
            "loss_vgg": avg_loss_vgg,
            "loss_change": avg_loss_change,
            "loss_color": avg_loss_color,
            "loss_tv": avg_loss_tv,
            "loss_sam": avg_loss_sam,
            "all_band_psnr_db": mean_val_psnr,
            "all_band_ssim": mean_val_ssim,
            "rgb_psnr_db": mean_val_rgb_psnr,
            "rgb_ssim": mean_val_rgb_ssim,
            "rgb_lpips": mean_val_lpips,
            "sam_degrees": mean_val_sam,
            "overall_score": overall_score,
        }
        with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        # Update training history CSV and generate publication loss & learning curves
        history_records.append(epoch_record)
        pd.DataFrame(history_records).to_csv(history_csv, index=False)
        try:
            save_training_curves(history_records, output_dir, args.optical_mode)
        except Exception as e:
            print(f"[WARN] Failed to save curve plots: {e}")

        training_state = {
            "epoch": epoch,
            "optical_mode": args.optical_mode,
            "optical_channels": channels,
            "residual_scale": args.residual_scale,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_lpips": best_lpips,
            "best_sam": best_sam,
            "best_overall": best_overall,
        }
        safe_torch_save(training_state, output_dir / "latest_training_state.pth")
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            safe_torch_save(training_state, output_dir / f"training_state_epoch_{epoch:03d}.pth")

    print("\nTraining Complete!")
    print(f"Best PSNR Checkpoint    : {output_dir / 'best_psnr_generator.pth'}")
    print(f"Best SSIM Checkpoint    : {output_dir / 'best_ssim_generator.pth'}")
    print(f"Best LPIPS Checkpoint   : {output_dir / 'best_lpips_generator.pth'}")
    if args.optical_mode == "s2":
        print(f"Best SAM Checkpoint     : {output_dir / 'best_sam_generator.pth'}")
    print(f"Best Overall Checkpoint : {output_dir / 'best_overall_generator.pth'}")


if __name__ == "__main__":
    main()