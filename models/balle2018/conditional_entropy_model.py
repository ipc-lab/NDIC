from torch import nn
import torch
import torch.optim
from models.balle2017.entropy_model import lower_bound_fn

lower_bound = lower_bound_fn.apply


class ConditionalEntropyBottleneck(nn.Module):

    def __init__(self, likelihood_bound=1e-9, cor_input=False):
        super(ConditionalEntropyBottleneck, self).__init__()
        self.likelihood_bound = float(likelihood_bound)
        self.cor_input = cor_input

    def standardized_cumulative(self, inputs):
        half = torch.tensor(.5, dtype=inputs.dtype)
        const = torch.tensor(-(2 ** -0.5), dtype=inputs.dtype)
        return half * torch.erfc(const * inputs)

    def forward(self, x, sigma):
        x = x.permute(1, 0, 2, 3)
        shape = x.shape
        x = torch.reshape(x, (shape[0], 1, -1)).float()

        sigma = sigma.permute(1, 0, 2, 3)
        sigma_shape = sigma.shape
        sigma = torch.reshape(sigma, (sigma_shape[0], 1, -1)).float()

        if not self.cor_input:
            if self.training:
                x = x + torch.rand_like(x) - 0.5
            else:
                x = torch.round(x)

        values = abs(x)

        upper = self.standardized_cumulative((.5 - values) / sigma)
        lower = self.standardized_cumulative((-.5 - values) / sigma)

        likelihood = upper - lower

        if self.likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self.likelihood_bound)

        x = torch.reshape(x, shape)
        x = x.permute(1, 0, 2, 3)
        likelihood = torch.reshape(likelihood, shape)
        likelihood = likelihood.permute(1, 0, 2, 3)

        return x, likelihood
