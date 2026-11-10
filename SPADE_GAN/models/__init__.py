from .spade_modules import SPADE, SPADEResnetBlock
from .generator import SPADEGenerator
from .discriminator import MultiscaleDiscriminator, PatchGANDiscriminator
from .losses import HingeGANLoss, SAMLoss, GANFeatureMatchingLoss

__all__ = [
    "SPADE",
    "SPADEResnetBlock",
    "SPADEGenerator",
    "MultiscaleDiscriminator",
    "PatchGANDiscriminator",
    "HingeGANLoss",
    "SAMLoss",
    "GANFeatureMatchingLoss",
]
