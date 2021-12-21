import torch

# Annealing Optimizer

temp_update = lambda t, beta: t * beta

class AnnealingOptimizer:

    def __init__(self, rate_weight, distortion_weight, t=1e6, beta=0.98):
        self.rate_weight = rate_weight
        self.distortion_weight = distortion_weight
        self.beta = beta
        self.t = t
        self.inflection = True
    
    def set_original_entropy(self, rate):
        self.original_entropy = rate.item()

    def forward(self, ssim, entropy_estimate, epoch):

        prev_rate_weight = self.rate_weight

        if epoch == 0:
            self.prev_ssim = ssim
            self.prev_entropy_estimate = entropy_estimate

        if ssim < 0.97:
            self.distortion_weight = 1e5

        delta_c = ssim - self.prev_ssim
        # if model performs worse
        if delta_c < 0:
            self.distortion_weight += (1 - ssim.item()) * self.t
                
        delta_c = entropy_estimate / self.rate_weight - self.prev_entropy_estimate
        # if model performs worse
        if delta_c > 0:
            # increase hyperparameter
            entropy_estimate_normalized = (entropy_estimate / self.rate_weight).item()
            self.rate_weight += entropy_estimate_normalized / self.original_entropy * self.t
        else:
            self.rate_weight *= 1e-1
            if self.rate_weight < 1:
                self.rate_weight = 1

        # sufficient condition to decrease temperature
        if ssim > 0.98:
            self.distortion_weight = 1
            if self.inflection:
                self.t = temp_update(self.t, self.beta)
                self.inflection = False
        else:
            self.inflection = True

        self.prev_ssim = ssim
        self.prev_entropy_estimate = entropy_estimate / prev_rate_weight

        return self.rate_weight, self.distortion_weight

