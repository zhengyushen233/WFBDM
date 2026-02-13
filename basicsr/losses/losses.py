import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from basicsr.metrics.psnr_ssim import _ssim

# from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')



@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)
    

@LOSS_REGISTRY.register()
class AFFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        #self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_fft = torch.abs(pred_fft)
        target_fft = torch.abs(target_fft)
        #return self.loss_weight * self.criterion(pred_fft, target_fft)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, crop_border=0, test_y_channel=True):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel

    def to_y_channel(self, x):
        y = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        return y.unsqueeze(1)

    def _ssim_pytorch(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.float()
        img2 = img2.float()

        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=img1.device)
        kernel = kernel.view(1, 1, 3, 3) / 16.0
        padding = 1

        mu1 = F.conv2d(img1, kernel, padding=padding, groups=1)
        mu2 = F.conv2d(img2, kernel, padding=padding, groups=1)

        sigma1 = F.conv2d(img1 ** 2, kernel, padding=padding, groups=1) - mu1 **2
        sigma2 = F.conv2d(img2** 2, kernel, padding=padding, groups=1) - mu2 **2
        sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=1) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        if self.crop_border > 0:
            pred_crop = pred[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            target_crop = target[:, :, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
        else:
            pred_crop = pred
            target_crop = target

        if self.test_y_channel:
            pred_y = self.to_y_channel(pred_crop)
            target_y = self.to_y_channel(target_crop)
        else:
            pred_y = pred_crop
            target_y = target_crop

        ssim_val = self._ssim_pytorch(pred_y, target_y)
        return self.loss_weight * (1 - ssim_val)

