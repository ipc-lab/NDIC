from torch import nn


class HyperAnalysisTransform(nn.Module):
    def __init__(self, num_filters=192):
        super(HyperAnalysisTransform, self).__init__()
        self.conv_h1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.relu_h1 = nn.ReLU()
        self.conv_h2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.relu_h2 = nn.ReLU()
        self.conv_h3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.relu_h1(x)
        x = self.conv_h1(x)
        x = self.conv_h2(x)
        x = self.relu_h2(x)
        x = self.conv_h3(x)

        return x


class HyperSynthesisTransform(nn.Module):
    def __init__(self, num_filters=192, num_filters_out=192):
        super(HyperSynthesisTransform, self).__init__()
        self.conv_h4 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h4 = nn.ReLU()
        self.conv_h5 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h5 = nn.ReLU()
        self.conv_h6 = nn.ConvTranspose2d(num_filters, num_filters_out, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_h4(x)
        x = self.relu_h4(x)
        x = self.conv_h5(x)
        x = self.relu_h5(x)
        x = self.conv_h6(x)

        return x
