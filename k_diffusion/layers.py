import math
import torch
from torch import nn
from torch.nn import functional as F
from . import utils

class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, target, noise, sigma, model_cond=None, history_latents=None, **kwargs):
        """
        --- 改造后的 Loss ---
        'input': 要加噪的输入 (z_t-1)
        'target': 预测的目标 (z_t)
        'history_latents': LSDC时序条件 (z_t-T...z_t-1)
        """
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        t = sigma.log().mul(120).add(500).clamp(0, 999).long()
        predicted_noise = self.inner_model(noised_input * c_in, t, model_cond=model_cond, history_latents=history_latents, **kwargs)
        predicted_x0 = noised_input - utils.append_dims(sigma, predicted_noise.ndim) * predicted_noise
        diffusion_loss = F.mse_loss(predicted_x0, target, reduction='none').mean(dim=list(range(1, target.ndim)))
        return diffusion_loss, predicted_x0

    def forward(self, input, sigma, model_cond=None, history_latents=None, **kwargs):
        """
        'input': 要去噪的输入 (z_t-1 或其噪声版本)
        'history_latents': LSDC时序条件 (z_t-T...z_t-1)
        """
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        t = sigma.log().mul(120).add(500).clamp(0, 999).long()
        predicted_noise = self.inner_model(input * c_in, t, model_cond=model_cond, history_latents=history_latents, **kwargs)
        denoised = input - utils.append_dims(sigma, predicted_noise.ndim) * predicted_noise
        return denoised


class DenoiserWithVariance(Denoiser):
    def loss(self, input, target, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        
        predicted_x0 = noised_input - utils.append_dims(sigma, model_output.ndim) * model_output

        losses = ((predicted_x0 - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)


# Embeddings

class FourierFeatures_aot(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return f.cos(), f.sin()