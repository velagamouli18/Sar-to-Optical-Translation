import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):
    """
    Spatially-Adaptive Denormalization.

    Study 2:
    - Uses BatchNorm2d, matching the baseline.
    - cond_nc is dynamic because each decoder level receives
      a different multi-scale SAR feature map.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        nhidden: int = 128,
        norm_type: str = "instance",
    ):
        super().__init__()

        if norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        else:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    cond_nc,
                    nhidden,
                    kernel_size=3,
                    padding=1
                )
            ),
            nn.ReLU(inplace=True),
        )

        self.mlp_gamma = spectral_norm(
            nn.Conv2d(
                nhidden,
                norm_nc,
                kernel_size=3,
                padding=1
            )
        )

        self.mlp_beta = spectral_norm(
            nn.Conv2d(
                nhidden,
                norm_nc,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:

        normalized = self.param_free_norm(x)

        # Safety fallback.
        # Normally Study 2 feature maps already have
        # the correct spatial resolution.
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(
                cond,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        actv = self.mlp_shared(cond)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    """
    Residual block with SPADE normalization.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        cond_nc: int,
        norm_type: str = "instance",
    ):
        super().__init__()

        self.learned_shortcut = fin != fout

        fmiddle = min(fin, fout)

        # Main branch
        self.spade_0 = SPADE(
            fin,
            cond_nc,
            norm_type=norm_type,
        )

        self.actvn_0 = nn.LeakyReLU(
            0.2,
            inplace=False
        )

        self.conv_0 = spectral_norm(
            nn.Conv2d(
                fin,
                fmiddle,
                kernel_size=3,
                padding=1
            )
        )

        self.spade_1 = SPADE(
            fmiddle,
            cond_nc,
            norm_type=norm_type,
        )

        self.actvn_1 = nn.LeakyReLU(
            0.2,
            inplace=False
        )

        self.conv_1 = spectral_norm(
            nn.Conv2d(
                fmiddle,
                fout,
                kernel_size=3,
                padding=1
            )
        )

        # Shortcut
        if self.learned_shortcut:

            self.spade_s = SPADE(
                fin,
                cond_nc,
                norm_type=norm_type,
            )

            self.conv_s = spectral_norm(
                nn.Conv2d(
                    fin,
                    fout,
                    kernel_size=1,
                    bias=False
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:

        x_s = self.shortcut(x, cond)

        dx = self.spade_0(x, cond)
        dx = self.actvn_0(dx)
        dx = self.conv_0(dx)

        dx = self.spade_1(dx, cond)
        dx = self.actvn_1(dx)
        dx = self.conv_1(dx)

        out = x_s + dx

        return out

    def shortcut(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:

        if self.learned_shortcut:

            x_s = self.spade_s(x, cond)
            x_s = self.conv_s(x_s)

        else:

            x_s = x

        return x_s