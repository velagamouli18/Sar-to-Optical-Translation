"""Bitemporal Change-Guided SPADE GAN for Disaster Satellite Synthesis.

Physical Multi-Stage Modeling:
  Stage 1: Pre-Disaster Cross-Modal Grounding (Pre_S1 <-> Pre_S2)
  Stage 2: Differential SAR Change Feature Extraction (ΔS1 = Post_S1 - Pre_S1)
  Stage 3: Change-Guided SPADE Modulation & Residual Optical Synthesis

The optical branch can be either 3-channel S2-TCI or the complete 12-band S2
stack.  ``optical_channels`` controls every optical-facing layer.
"""

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SPADEBlock(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE) layer."""

    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int = 128):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        # Interpolate condition to match feature map spatial size
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=True)
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # Clamp gamma and beta to [-2.0, 2.0] to prevent explosive feature scaling
        gamma = torch.clamp(gamma, min=-2.0, max=2.0)
        beta = torch.clamp(beta, min=-2.0, max=2.0)
        return normalized * (1.0 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    """Residual block with SPADE normalization conditioned on cross-modal change."""

    def __init__(self, in_ch: int, out_ch: int, cond_nc: int):
        super().__init__()
        self.learned_shortcut = (in_ch != out_ch)

        self.conv_0 = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.spade_0 = SPADEBlock(in_ch, cond_nc)
        self.spade_1 = SPADEBlock(out_ch, cond_nc)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            self.spade_s = SPADEBlock(in_ch, cond_nc)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_s = x
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.spade_s(x, cond)))

        dx = self.conv_0(self.actvn(self.spade_0(x, cond)))
        dx = self.conv_1(self.actvn(self.spade_1(dx, cond)))
        return x_s + dx


class DifferentialSAREncoder(nn.Module):
    """Explicitly computes multi-scale differential radar change features."""

    def __init__(self, in_ch: int = 8, base_ch: int = 64):
        """in_ch = 8: [pre_s1(2), post_s1(2), |post - pre|(2), (post - pre)(2)]"""
        super().__init__()
        self.conv_in = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, pre_s1: torch.Tensor, post_s1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diff_abs = torch.abs(post_s1 - pre_s1)
        diff_signed = post_s1 - pre_s1
        sar_stack = torch.cat([pre_s1, post_s1, diff_abs, diff_signed], dim=1)  # 8 ch

        f0 = self.conv_in(sar_stack)  # (B, 64, H, W)
        f1 = self.down1(f0)           # (B, 128, H/2, W/2)
        f2 = self.down2(f1)           # (B, 256, H/4, W/4)
        return f0, f1, f2


class CrossModalPreEncoder(nn.Module):
    """Learns normal baseline grounding between pre_s1 and pre_s2."""

    def __init__(self, in_ch: int = 5, base_ch: int = 64):
        """in_ch is 2 S1 channels plus 3 TCI or 12 complete-S2 channels."""
        super().__init__()
        self.conv_in = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, pre_s1: torch.Tensor, pre_s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([pre_s1, pre_s2], dim=1)
        f0 = self.conv_in(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        return f0, f1, f2


class BitemporalChangeGenerator(nn.Module):
    """Change-Guided SPADE Generator synthesizing Post-Disaster Optical from Pre-S2 + SAR Change."""

    def __init__(self, ngf: int = 64, optical_channels: int = 3, residual_scale: float = 1.0):
        super().__init__()
        if optical_channels not in (3, 12):
            raise ValueError(f"optical_channels must be 3 or 12, got {optical_channels}")
        self.optical_channels = optical_channels
        self.residual_scale = float(residual_scale)
        self.sar_encoder = DifferentialSAREncoder(in_ch=8, base_ch=ngf)
        self.pre_encoder = CrossModalPreEncoder(in_ch=2 + optical_channels, base_ch=ngf)

        # Content trunk: initial projection of pre_s2
        self.content_conv = spectral_norm(nn.Conv2d(optical_channels, ngf * 4, kernel_size=3, padding=1))

        # Condition channels at scale 2 (H/4): 256 (SAR) + 256 (Pre) = 512 ch
        # Condition channels at scale 1 (H/2): 128 (SAR) + 128 (Pre) = 256 ch
        # Condition channels at scale 0 (H):   64 (SAR) + 64 (Pre)   = 128 ch

        # Bottle neck resnet blocks at H/4: ngf * 4 (SAR) + ngf * 4 (Pre) = ngf * 8 ch
        self.res_b1 = SPADEResnetBlock(ngf * 4, ngf * 4, cond_nc=ngf * 8)
        self.res_b2 = SPADEResnetBlock(ngf * 4, ngf * 4, cond_nc=ngf * 8)

        # Upsampling Stage 1: H/4 -> H/2: ngf * 2 (SAR) + ngf * 2 (Pre) = ngf * 4 ch
        self.res_up1 = SPADEResnetBlock(ngf * 4, ngf * 2, cond_nc=ngf * 4)

        # Upsampling Stage 2: H/2 -> H: ngf (SAR) + ngf (Pre) = ngf * 2 ch
        self.res_up0 = SPADEResnetBlock(ngf * 2, ngf, cond_nc=ngf * 2)

        # Final residual prediction head with High-Resolution Skip Connection:
        # Fuses decoder trunk (ngf) with original 128x128 pre-optical features pre_f0 (ngf) -> ngf * 2
        # This constrains the synthesized residual to respect crisp building edges.
        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, optical_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        pre_s1: torch.Tensor,
        pre_s2: torch.Tensor,
        post_s1: torch.Tensor,
        return_delta: bool = False,
    ):
        """
        Synthesizes post-disaster optical image.

        Args:
            pre_s1: (B, 2, H, W)
            pre_s2: (B, optical_channels, H, W)
            post_s1: (B, 2, H, W)
            return_delta: If True, returns (post_s2_pred, delta_optical)

        Returns:
            post_s2_pred: (B, optical_channels, H, W) in range [-1, 1]
        """
        # 1. Extract SAR Differential Change Features (Scale 0, 1, 2)
        sar_f0, sar_f1, sar_f2 = self.sar_encoder(pre_s1, post_s1)

        # 2. Extract Cross-Modal Pre-Disaster Grounding Features
        pre_f0, pre_f1, pre_f2 = self.pre_encoder(pre_s1, pre_s2)

        # Combined conditions at each scale:
        cond_2 = torch.cat([sar_f2, pre_f2], dim=1)  # 512 ch at H/4
        cond_1 = torch.cat([sar_f1, pre_f1], dim=1)  # 256 ch at H/2
        cond_0 = torch.cat([sar_f0, pre_f0], dim=1)  # 128 ch at H

        # 3. Content Seed from Pre-S2
        x = F.interpolate(pre_s2, scale_factor=0.25, mode="bilinear", align_corners=True)
        h = self.content_conv(x)  # (B, 256, H/4, W/4)

        # 4. Modulated Processing
        h = self.res_b1(h, cond_2)
        h = self.res_b2(h, cond_2)

        # Upsample to H/2
        h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=True)
        h = self.res_up1(h, cond_1)

        # Upsample to H
        h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=True)
        h = self.res_up0(h, cond_0)

        # 5. Predict Optical Residual Delta with High-Resolution Skip Connection
        # Concat decoder features (h) with pre_f0 (128x128 structural features)
        h_fused = torch.cat([h, pre_f0], dim=1)
        delta_optical = self.final_conv(h_fused) * self.residual_scale

        # Residual Addition: post_s2 = clamp(pre_s2 + delta, -1, 1)
        post_s2_pred = torch.clamp(pre_s2 + delta_optical, -1.0, 1.0)
        if return_delta:
            return post_s2_pred, delta_optical
        return post_s2_pred


class MultiScaleChangeDiscriminator(nn.Module):
    """Multi-Scale PatchGAN Discriminator conditioned on [pre_s1, pre_s2, post_s1, post_s2]."""

    def __init__(
        self,
        in_ch: int = None,
        ndf: int = 64,
        num_scales: int = 2,
        optical_channels: int = 3,
    ):
        """Condition on S1 pre/post and optical pre/post (10 TCI or 28 full-S2 channels)."""
        super().__init__()
        expected_in_ch = 4 + 2 * optical_channels
        if in_ch is None:
            in_ch = expected_in_ch
        if in_ch != expected_in_ch:
            raise ValueError(
                f"Discriminator in_ch={in_ch}, but optical_channels={optical_channels} "
                f"requires {expected_in_ch}."
            )
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            self._build_sub_discriminator(in_ch, ndf) for _ in range(num_scales)
        ])

    def _build_sub_discriminator(self, in_ch: int, ndf: int) -> nn.Sequential:
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(
        self,
        pre_s1: torch.Tensor,
        pre_s2: torch.Tensor,
        post_s1: torch.Tensor,
        post_s2: torch.Tensor,
    ) -> List[torch.Tensor]:
        x = torch.cat([pre_s1, pre_s2, post_s1, post_s2], dim=1)
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:
                x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        return outputs