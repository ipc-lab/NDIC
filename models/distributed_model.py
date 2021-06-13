import torch
import math
from models.balle2017 import entropy_model, gdn
from torch import nn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck

lower_bound = entropy_model.lower_bound_fn.apply


'''
The following model is based on the balle2018 model, which uses scale hyperpriors (z). 
'''


class HyperPriorDistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(HyperPriorDistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.ha_primary_image = HyperAnalysisTransform(num_filters)
        self.hs_primary_image = HyperSynthesisTransform(num_filters)

        self.ha_cor_image = HyperAnalysisTransform(num_filters)
        self.hs_cor_image = HyperSynthesisTransform(num_filters)

        self.entropy_bottleneck_sigma_x = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_sigma_y = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_common_info = entropy_model.EntropyBottleneck(num_filters, quantize=False)

        self.conditional_entropy_bottleneck_hx = ConditionalEntropyBottleneck()
        self.conditional_entropy_bottleneck_hy = ConditionalEntropyBottleneck()

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv2(x)
        x = self.igdn3(x)
        x = self.deconv4(x)

        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)

        return x

    def forward(self, x, y):
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        z = self.ha_primary_image(abs(hx))
        z_tilde, z_likelihoods = self.entropy_bottleneck_sigma_x(z)
        sigma = self.hs_primary_image(z_tilde)
        sigma_lower_bounded = lower_bound(sigma, self.bound)
        
        z_cor = self.ha_cor_image(abs(hy))
        z_tilde_cor, z_likelihoods_cor = self.entropy_bottleneck_sigma_y(
            z_cor)
        sigma_cor = self.hs_cor_image(z_tilde_cor)
        sigma_cor_lower_bounded = lower_bound(sigma_cor, self.bound)

        hx_tilde, x_likelihoods = self.conditional_entropy_bottleneck_hx(hx, sigma_lower_bounded)
        hy_tilde, y_likelihoods = self.conditional_entropy_bottleneck_hy(hy, sigma_cor_lower_bounded)

        w = self.encode_w(y)  # p(w|y), i.e. the "common variable"
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding a small Gaussian noise improves stability of the training
        _, w_likelihoods = self.entropy_bottleneck_common_info(w)

        x_tilde = self.decode(hx_tilde, w)
        y_tilde = self.decode_cor(hy_tilde, w)

        return x_tilde, y_tilde, x_likelihoods, y_likelihoods, z_likelihoods, \
               z_likelihoods_cor, w_likelihoods


'''
This model is based on balle2017 model.
'''


class DistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(DistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv2(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x

    def forward(self, x, y):
        w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding small Gaussian noise improves the stability of training
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        hy_tilde, y_likelihoods = self.entropy_bottleneck_hy(hy)
        _, w_likelihoods = self.entropy_bottleneck(w)

        x_tilde = self.decode(hx_tilde, w)
        y_tilde = self.decode_cor(hy_tilde, w)
        return x_tilde, y_tilde, x_likelihoods, y_likelihoods, w_likelihoods


if __name__ == '__main__':
    net = HyperPriorDistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
    net = DistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
