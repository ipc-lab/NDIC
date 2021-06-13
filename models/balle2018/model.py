import torch
from torch import nn
from models.balle2017 import entropy_model, gdn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck, lower_bound


class  BMSHJ2018Model(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(BMSHJ2018Model, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.ha = HyperAnalysisTransform(num_filters)
        self.hs = HyperSynthesisTransform(num_filters)
        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters)
        self.conditional_entropy_bottleneck = ConditionalEntropyBottleneck()

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

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

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x

    def forward(self, x):
        y = self.encode(x)
        z = self.ha(abs(y))
        z_tilde, z_likelihoods = self.entropy_bottleneck(z)
        sigma = self.hs(z_tilde)
        sigma = lower_bound(sigma, self.bound)
        y_tilde, y_likelihoods = self.conditional_entropy_bottleneck(y, sigma)
        x_tilde = self.decode(y_tilde)
        return x_tilde, y_likelihoods, z_likelihoods


if __name__ == '__main__':
    net = BMSHJ2018Model()
    print(net(torch.randn(1, 3, 256, 256))[0].shape)
