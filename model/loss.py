from piqa import MS_SSIM
import numpy as np
import torch.nn as nn
import torch

# LOSS FUNCTION(S)

# slope for scaling each band frequency
interval = 2

class RateLoss(nn.Module):

    band_freq_scales = [i for i in range(1, 16*interval, interval)]
    lbf_scale = torch.tensor(np.array([band_freq_scales[0]]+
                                      [band_freq_scales[1]]*2+
                                      [band_freq_scales[2]]*3, 
                                      dtype=np.float64)
                                      )
    mbf_scale = torch.tensor(np.array([band_freq_scales[3]]*4+
                                      [band_freq_scales[4]]*5+
                                      [band_freq_scales[5]]*6+
                                      [band_freq_scales[6]]*7, 
                                      dtype=np.float64)
                                      )
    hbf_scale = torch.tensor(np.array([band_freq_scales[7]]*8+
                                      [band_freq_scales[8]]*7+
                                      [band_freq_scales[9]]*6+
                                      [band_freq_scales[10]]*5+
                                      [band_freq_scales[11]]*4+
                                      [band_freq_scales[12]]*3+
                                      [band_freq_scales[13]]*2+
                                      [band_freq_scales[14]], 
                                      dtype=np.float64)
                                      )
    
    def __init__(self, band_scales, device):
        super(RateLoss, self).__init__()
        self.band_scales = band_scales
        self.lbf_scale = self.lbf_scale.to(device)
        self.mbf_scale = self.mbf_scale.to(device)
        self.hbf_scale = self.hbf_scale.to(device)

    def forward(self, z):
        z[...,:6] *= self.band_scales[0] * self.lbf_scale
        z[...,6:28] *= self.band_scales[1] * self.mbf_scale
        z[...,28:] *= self.band_scales[2] * self.hbf_scale
        # enforce_sparsity = torch.mean(torch.linalg.norm(z, ord=2, dim=1))
        enforce_sparsity = torch.mean(torch.linalg.norm(z, ord=2, dim=-1))
        return enforce_sparsity


class DistortionLoss(nn.Module):

    def __init__(self, win_size=3, blur_kernel=13, ssim_scale=1e6, alpha=0.84, n_channels=1, blur_fn=None):
        super(DistortionLoss, self).__init__()
        self.alpha = alpha
        self.ssim_scale = ssim_scale
        self.blur_fn = blur_fn
        self.l1loss = nn.L1Loss()
        self.criterion = MS_SSIM(window_size=win_size, 
                                 sigma=1e5, 
                                 n_channels=n_channels, 
                                 padding=True, 
                                 value_range=255.0,
                                 reduction='mean').double()

    def forward(self, x, y):
        ssim = self.criterion(x, y)
        ssim_loss = -1 * self.ssim_scale * torch.log(ssim)
        if self.blur_fn is not None:
            l1_loss = self.l1loss(self.blur_fn(x), self.blur_fn(y))
        else:
            l1_loss = self.l1loss(x, y)
        img_quality = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
        return ssim, img_quality


class QuantizationLoss(nn.Module):
    
    def __init__(self, rate_weight, distortion_weight):
        super(QuantizationLoss, self).__init__()
        self.rate_weight = rate_weight
        self.distortion_weight = distortion_weight

    def forward(self, rate, distortion):
        print('rate:', self.rate_weight * rate.item())
        print('distortion:', self.distortion_weight * distortion.item())
        return self.rate_weight * rate + self.distortion_weight * distortion
