import torch
from torch import nn
import torch.optim
from models.balle2017.entropy_model import lower_bound_fn

torch.set_printoptions(precision=10)

lower_bound = lower_bound_fn.apply


class GDN(nn.Module):
    def __init__(self, channels_in, inverse = False):
        super(GDN, self).__init__()
        self.channels_in = channels_in
        self.inverse = inverse
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(channels_in) * .1 + 2 ** -36), requires_grad=True)
        self.beta = nn.Parameter(torch.sqrt(torch.ones(channels_in) * 1. + 2 ** -36), requires_grad=True)

    def parametrize(self, reparam_offset=2 ** -18):
        reparam_offset = float(reparam_offset)
        pedestal = reparam_offset ** 2
        beta_parametrized = lower_bound(self.beta, (1e-6 + reparam_offset ** 2) ** .5)
        beta_parametrized = torch.pow(beta_parametrized, 2) - pedestal
        gamma_parametrized = lower_bound(self.gamma, (0 + reparam_offset ** 2) ** .5)
        gamma_parametrized = torch.pow(gamma_parametrized, 2) - pedestal
        return beta_parametrized, gamma_parametrized

    def forward(self, input):

        beta_parametrized, gamma_parametrized = self.parametrize()
        x = torch.tensordot(torch.pow(input, 2), gamma_parametrized, dims=[[1], [0]])
        x = torch.add(x, beta_parametrized)
        x = x.permute(0, 3, 1, 2)
        if self.inverse:
            x = torch.mul(input, torch.sqrt(x))
        else:
            x = torch.mul(input, torch.rsqrt(x))
        return x


# TESTING
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)