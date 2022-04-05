"""
generator network
Input: torch (1, 3, 256, 256)
upsampling: 放大图像
downsampling: 缩小图像，降低图像质量
downsampling first, then upsampling
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


# input: torch.rand([1, 3, 256, 256])
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        # 7 downsampling blocks
        # first downsampling block does not have Instance Norm
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)  # 64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)  # 32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)  # 16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)  # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU(),  # 1
        )

        self.up1 = Block(features * 8, features * 8, down=False, act='relu', use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act='relu', use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act='relu', use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act='relu', use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # Implement U-net with skip connections
    def forward(self, x):
        d1 = self.initial_down(x)  # [1, 64, 128, 128]
        d2 = self.down1(d1)  # [1, 128, 64, 64]
        d3 = self.down2(d2)  # [1, 256, 32, 32]
        d4 = self.down3(d3)  # [1, 512, 16, 16]
        d5 = self.down4(d4)  # [1, 512, 8, 8]
        d6 = self.down5(d5)  # [1, 512, 4, 4]
        d7 = self.down6(d6)  # [1, 512, 2, 2]
        bottleneck = self.bottleneck(d7)  # [1, 512, 1, 1]
        up1 = self.up1(bottleneck)  # [1, 512, 2, 2]
        # 按照维数1拼接
        up2 = self.up2(torch.cat([up1, d7], 1))  # [1, 512, 4, 4]
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))  # [1, 64, 128, 128]
        return self.final_up(torch.cat([up7, d1], 1))  # 1, 3, 256, 256]

        # return up7


generator = Generator()
input = torch.FloatTensor(1, 3, 256, 256)
output = generator(input)
print(output.shape)
