import torch
from torch import nn
from models.balle2017 import entropy_model, gdn


class BLS2017Model(nn.Module):
    def __init__(self, num_filters=192):
        super(BLS2017Model, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 9, stride=4, padding=4)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters)

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, 3, 9, stride=4, padding=4, output_padding=3)

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn2(x)
        x = self.deconv2(x)
        x = self.igdn3(x)
        x = self.deconv3(x)
        return x

    def forward(self, x):
        y = self.encode(x)
        y_tilde, likelihoods = self.entropy_bottleneck(y)
        x_tilde = self.decode(y_tilde)
        return x_tilde, likelihoods


if __name__ == '__main__':
    net = BLS2017Model().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda())[0].shape)
