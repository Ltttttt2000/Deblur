"""
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)
output should be shape [1,1,26,26]
"""

import torch
import torch.nn as nn


# This block is repeated 3 times in the Discriminator
# with different 'out_channels' values
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # CNNBlock consists of Conv layer, Instance norm, Leaky relu
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    # method overload for 'forward' method in the nn.Module class
    def forward(self, input):
        return self.conv(input)


# Implement the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super().__init__()

        # first conv block does not have Instance norm, hence implementing separately
        if features is None:
            features = [64, 128, 256, 512]
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        # last layer which is signature to PatchGAN
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

    # overload forward in the nn.Module class
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


blur = torch.FloatTensor(1, 3, 256, 256)
clear = torch.FloatTensor(1, 3, 256, 256)
dis = Discriminator()
output = dis(blur, clear)
print(output.shape)
