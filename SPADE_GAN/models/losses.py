from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HingeGANLoss(nn.Module):
    """
    Hinge Loss for GANs (State of the art for SPADE and BigGAN).
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(
        self, pred_real_list: List[List[torch.Tensor]], pred_fake_list: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        loss = 0.0
        for pred_real, pred_fake in zip(pred_real_list, pred_fake_list):
            # The final layer output is the patch prediction
            d_real = pred_real[-1]
            d_fake = pred_fake[-1]
            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
            loss += (loss_real + loss_fake) * 0.5
        return loss / len(pred_real_list)

    def generator_loss(self, pred_fake_list: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = 0.0
        for pred_fake in pred_fake_list:
            d_fake = pred_fake[-1]
            loss += -d_fake.mean()
        return loss / len(pred_fake_list)


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss for 12-channel multispectral remote sensing.
    Computes the angular distance in radians between the predicted 12-band spectral vector
    and ground truth 12-band spectral vector at every pixel.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 12, H, W)
        # Shift back from [-1, 1] to reflectance [0, 1] for physical spectral angles
        pred_ref = (pred + 1.0) * 0.5
        target_ref = (target + 1.0) * 0.5

        # Inner product along channel dimension
        dot_product = (pred_ref * target_ref).sum(dim=1)
        norm_pred = torch.norm(pred_ref, p=2, dim=1)
        norm_target = torch.norm(target_ref, p=2, dim=1)

        denominator = norm_pred * norm_target + self.eps
        cos_theta = dot_product / denominator
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

        sam_rad = torch.acos(cos_theta)
        return sam_rad.mean()


class GANFeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss across intermediate discriminator layers.
    Stabilizes GAN training and eliminates high-frequency noise.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(
        self, pred_real_list: List[List[torch.Tensor]], pred_fake_list: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        loss = 0.0
        num_d = len(pred_real_list)
        for pred_real, pred_fake in zip(pred_real_list, pred_fake_list):
            num_intermediate_layers = len(pred_real) - 1
            for i in range(num_intermediate_layers):
                loss += self.criterion(pred_fake[i], pred_real[i].detach())
        return loss / (num_d * max(1, num_intermediate_layers))


class VGG19Feat(nn.Module):
    """
    Extracts multi-scale intermediate feature representations from VGG19.
    Corresponds to relu1_2, relu2_2, relu3_2, relu4_2, relu5_2.
    """

    def __init__(self, requires_grad: bool = False):
        super().__init__()
        vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module):
    """
    Perceptual Loss based on VGG19 feature representations.
    Standard in SPADE (Park et al. 2019) and SAR-to-Optical translation literature.
    Input images are expected to be in [-1.0, 1.0] with 3 channels (RGB/TCI).
    """

    def __init__(self):
        super().__init__()
        self.vgg = VGG19Feat(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        # Register ImageNet normalization constants as buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 3, H, W) in [-1, 1]
        pred_01 = (pred + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5

        # Normalize with ImageNet constants
        pred_norm = (pred_01 - self.mean) / self.std
        target_norm = (target_01 - self.mean) / self.std

        pred_feats = self.vgg(pred_norm)
        target_feats = self.vgg(target_norm)

        loss = 0.0
        for p_f, t_f, w in zip(pred_feats, target_feats, self.weights):
            loss += w * self.criterion(p_f, t_f.detach())
        return loss


class ColorConsistencyLoss(nn.Module):
    """
    Low-frequency Color Consistency Loss for SAR-to-Optical translation.
    Applies an average pooling blur to strip out high-frequency spatial structures
    (edges, buildings, roads) and forces the generator to match the broad spatial
    color/illumination distribution of ground truth Sentinel-2.
    Prevents the generator from oscillating wildly between green (vegetation)
    and brown (soil/mud).
    """

    def __init__(self, kernel_size: int = 21):
        super().__init__()
        self.kernel_size = kernel_size
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        # Apply spatial average pooling as a fast, differentiable low-pass filter
        pred_blur = F.avg_pool2d(pred, kernel_size=self.kernel_size, stride=4, padding=self.kernel_size // 2)
        target_blur = F.avg_pool2d(target, kernel_size=self.kernel_size, stride=4, padding=self.kernel_size // 2)
        return self.criterion(pred_blur, target_blur)


class ColorLoss(nn.Module):
    """
    Regional Color Matching Loss for mixed-disaster optical synthesis.
    Applies spatial average pooling (kernel=7) with L1 loss on blurred representations.
    Using L1 instead of Cosine Similarity penalizes extreme pixel intensities,
    strictly preventing 'pooling bloom' (neon glowing halos around building footprints).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        fake_blurred = self.pool(fake_img)
        real_blurred = self.pool(real_img)
        return F.l1_loss(fake_blurred, real_blurred)


class TVLoss(nn.Module):
    """
    Total Variation Regularization Loss on residual optical delta (ΔY).
    Penalizes diffuse circular 'splats' by enforcing localized, compact gradients.
    """

    def __init__(self):
        super().__init__()

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        # delta: (B, C, H, W)
        diff_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])
        diff_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()



