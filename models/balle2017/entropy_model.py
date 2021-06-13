from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import torch.optim

# The `lower_bound_fn' is 
# from the repo: https://github.com/jorge-pessoa/pytorch-gdn/blob/master/pytorch_gdn/__init__.py,
# which is available under the MIT license:

'''MIT License

Copyright (c) 2019 Jorge Pessoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

class lower_bound_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


lower_bound = lower_bound_fn.apply

# This "entropy bottleneck" implementation clearly corresponds to the one mentioned in the paper below. 
# `Appendix' describes each step of the entropy estimation part.

"""
> "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://arxiv.org/abs/1802.01436
"""

# Their TensorFlow-based implementation can be found at:
# https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/entropy_models.py

class EntropyBottleneck(nn.Module):
    def __init__(self, channels, init_scale=10, filters=(3, 3, 3), tail_mass=1e-9,
                 optimize_integer_offset=True, likelihood_bound=1e-9, quantize=True,
                 range_coder_precision=16):

        super(EntropyBottleneck, self).__init__()
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        self.tail_mass = float(tail_mass)
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.likelihood_bound = float(likelihood_bound)
        self.range_coder_precision = int(range_coder_precision)
        self.matrices = []
        self.biases = []
        self.factors = []
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        # Initializing layer parameters: matrices, biases and factors
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.ones((channels, filters[i + 1], filters[i])) * init
            matrix = nn.Parameter(data=matrix, requires_grad=True)
            self.matrices.append(matrix)

            bias = nn.Parameter(data=torch.ones((channels, filters[i + 1], 1)).uniform_(-.5, .5), requires_grad=True)
            self.biases.append(bias)

            if i < len(self.filters):
                factor = torch.zeros((channels, filters[i + 1], 1))
                factor = nn.Parameter(data=factor, requires_grad=True)
                self.factors.append(factor)
        self.factors = nn.Parameter(data=torch.stack(self.factors), requires_grad=True)
        for i, matrix in enumerate(self.matrices):
            self.register_parameter('matrix{}'.format(i), matrix)
        for i, bias in enumerate(self.biases):
            self.register_parameter('bias{}'.format(i), bias)

            # To figure out what range of the densities to sample, we need to compute
            # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. 
            # Since we can't take inverses of the cumulative directly, we make it an optimization
            # problem:
            # `quantiles = argmin(|logit(cumulative) - target|)`
            # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
            # Taking the logit (inverse of sigmoid) of the cumulative makes the
            # representation of the right target more numerically stable.

            # Numerically "stable" way of computing logits of `tail_mass / 2`
            # and `1 - tail_mass / 2`.
        target = np.log(2 / self.tail_mass - 1)
        self.target = torch.tensor([-target, 0, target])
        self.quantize = quantize

    def logits_cumulative(self, inputs):
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.__getattr__('matrix{}'.format(i))
            matrix = F.softplus(matrix)
            logits = torch.matmul(matrix, logits)
            bias = self.__getattr__('bias{}'.format(i))
            logits += bias

            if i < len(self.factors):
                factor = self.factors[i]
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)
        return logits

    def forward(self, x):
        # x : Shape of the input tensor is used to get the number of channel.
        # Convert to (channels, 1, batch) format by commuting channels to front
        x = x.permute(1, 0, 2, 3)
        shape = x.shape
        x = torch.reshape(x, (shape[0], 1, -1)).float()

        if self.quantize:
            if self.training:
                x = x + torch.rand_like(x) - 0.5  # during training, adding noise for differentiability.
            else:
                x = torch.round(x)  
                # during evaluation, the data is quantized and the entropies are discrete (Shannon entropies)

        # Evaluate densities.
        # We can use the special rule below to only compute differences in the left
        # tail of the sigmoid. 
        lower = self.logits_cumulative(x - 0.5)
        upper = self.logits_cumulative(x + 0.5)

        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = torch.sign(lower.add(upper))
        # Calculating the pmf
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        if self.likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self.likelihood_bound)

        x = torch.reshape(x, shape)
        # Convert back to input tensor shape.
        x = x.permute(1, 0, 2, 3)
        likelihood = torch.reshape(likelihood, shape)
        likelihood = likelihood.permute(1, 0, 2, 3)

        return x, likelihood

