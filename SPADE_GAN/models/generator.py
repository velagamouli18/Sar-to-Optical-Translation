import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .spade_modules import SPADEResnetBlock


class SPADEGenerator(nn.Module):
    """
    SPADE Generator for Image-to-Image translation.
    Takes a 2-channel SAR condition (Sentinel-1) and generates
    an optical image (3-channel TCI or 12-channel Sentinel-2) at 128x128.
    """

    def __init__(
        self,
        cond_nc: int = 2,
        out_nc: int = 12,
        ngf: int = 64,
        init_size: int = 16,
        norm_type: str = "instance",
    ):
        super().__init__()
        self.cond_nc = cond_nc
        self.out_nc = out_nc
        self.init_size = init_size

        # Learned latent spatial grid (as specified in SPADE literature / Document 2 Table Stage 0)
        self.base_latent = nn.Parameter(torch.randn(1, 8 * ngf, init_size, init_size) * 0.02)

        # Initial linear/convolutional projection from condition to initial feature grid
        # 16x16 feature grid with 8 * ngf = 512 channels
        self.fc = spectral_norm(nn.Conv2d(cond_nc, 8 * ngf, kernel_size=3, padding=1))

        # Resnet blocks with SPADE modulation at progressive resolutions:
        # Scale 1: 16x16
        self.head_0 = SPADEResnetBlock(8 * ngf, 8 * ngf, cond_nc, norm_type=norm_type)

        # Scale 2: 16 -> 32
        self.up_0 = SPADEResnetBlock(8 * ngf, 4 * ngf, cond_nc, norm_type=norm_type)

        # Scale 3: 32 -> 64
        self.up_1 = SPADEResnetBlock(4 * ngf, 2 * ngf, cond_nc, norm_type=norm_type)

        # Scale 4: 64 -> 128
        self.up_2 = SPADEResnetBlock(2 * ngf, 1 * ngf, cond_nc, norm_type=norm_type)

        # Final refinement block at 128x128
        self.head_1 = SPADEResnetBlock(1 * ngf, 1 * ngf, cond_nc, norm_type=norm_type)

        # Output convolution predicts the configured optical channels.
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(1 * ngf, out_nc, kernel_size=3, padding=1),
            nn.Tanh(),  # Normalized optical target is [-1.0, 1.0]
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # 1. Create initial coarse feature map at 16x16 with learned latent grid prior
        cond_init = F.interpolate(cond, size=(self.init_size, self.init_size), mode="bilinear", align_corners=False)
        x = self.fc(cond_init) + self.base_latent.expand(cond.size(0), -1, -1, -1)

        # 2. ResNet processing at 16x16
        x = self.head_0(x, cond)

        # 3. Upsample to 32x32
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_0(x, cond)

        # 4. Upsample to 64x64
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_1(x, cond)

        # 5. Upsample to 128x128
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_2(x, cond)

        # 6. Refine at 128x128 and project to the configured output channels.
        x = self.head_1(x, cond)
        out = self.conv_out(x)
        return out
