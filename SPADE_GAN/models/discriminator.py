from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    Standard PatchGAN discriminator with Spectral Normalization.
    Returns intermediate features for Feature Matching Loss.
    """

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        self.n_layers = n_layers

        def conv_layer(in_c, out_c, kernel_size=4, stride=2, padding=1):
            conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
            return spectral_norm(conv) if use_spectral_norm else conv

        # First layer
        layers = [
            nn.Sequential(
                conv_layer(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        # Intermediate downsampling layers (SPADE uses pure Spectral Norm without InstanceNorm)
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.append(
                nn.Sequential(
                    conv_layer(nf_prev, nf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output block
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.append(
            nn.Sequential(
                conv_layer(nf_prev, nf, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        layers.append(
            nn.Sequential(
                conv_layer(nf, 1, kernel_size=4, stride=1, padding=1)
            )
        )

        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        results = []
        for layer in self.model:
            x = layer(x)
            results.append(x)
        return results


class MultiscaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator (2 or 3 scales).
    Evaluates both high-frequency edge details and macro regional textures.
    """

    def __init__(self, input_nc: int = 14, ndf: int = 64, n_layers: int = 3, num_D: int = 2):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers)
            for _ in range(num_D)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        results = []
        input_downsampled = x
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                input_downsampled = self.downsample(input_downsampled)
            results.append(disc(input_downsampled))
        return results
