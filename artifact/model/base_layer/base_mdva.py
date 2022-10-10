import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResBlock, self).__init__()
        conv_block = [nn.Conv2d(num_feat, num_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.InstanceNorm2d(num_feat),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(num_feat, num_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.InstanceNorm2d(num_feat),
                      nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class gen(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, num_feat):
        super(gen, self).__init__()

        head = [nn.Conv2d(num_in_ch, num_feat, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.InstanceNorm2d(num_feat),
                nn.ReLU(inplace=True)]

        downsample = [nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                      nn.InstanceNorm2d(2 * num_feat),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                      nn.InstanceNorm2d(4 * num_feat),
                      nn.ReLU(inplace=True)]

        body = []
        for _ in range(6):
            body.append(ResBlock(4 * num_feat))

        upsample = [nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.PixelShuffle(2),
                    nn.InstanceNorm2d(2 * num_feat),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.PixelShuffle(2),
                    nn.InstanceNorm2d(num_feat),
                    nn.ReLU(inplace=True)]

        tail = [nn.Conv2d(num_feat, num_out_ch, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.InstanceNorm2d(num_feat),
                nn.Tanh()]

        self.head = nn.Sequential(*head)
        self.downsample = nn.Sequential(*downsample)
        self.body = nn.Sequential(*body)
        self.upsample = nn.Sequential(*upsample)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        x = self.downsample(x)
        x = self.body(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


class disc(nn.Module):

    def __init__(self, in_ch, num_feat=64):
        super(disc, self).__init__()

        head = [nn.Conv2d(in_ch, num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        body = []
        for _ in range(5):
            body.append(nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            num_feat = num_feat * 2
        body.append(nn.AdaptiveAvgPool2d((1, 1)))

        tail = [nn.Linear(num_feat, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = torch.flatten(x, 1)
        x = self.tail(x)
        return x


if __name__ == '__main__':

    from torchsummary import summary

    model = disc(in_ch=1)

    summary(model, (1, 224, 224))