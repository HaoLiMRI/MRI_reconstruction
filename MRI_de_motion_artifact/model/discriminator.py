import torch.nn as nn
import torch
from model.base_layer.DenseNet import DenseNet
from pytorch_wavelets import DWTForward


############################################################
#  VGG Discriminator
############################################################
class DiscriminatorVGG(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(DiscriminatorVGG, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=(3, 3), stride=(1, 1), padding=1),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),  # state size. 64 x 64 x 64
            # nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d * 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            # nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),  # state size. 128 x 32 x 32
            # nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            # nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 4, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),  # state size. 256 x 16 x 16
            # nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            # nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),  # state size. 512 x 8 x 8
            # nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            # nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),  # state size. 512 x 4 x 4
            # nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.d * 8, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


############################################################
#  DSN Discriminator
############################################################
class Discriminator_DSN(nn.Module):
    def __init__(self, in_ch, kernel_size=5, highpass=False, norm_layer='Instance', cs='cat'):
        super(Discriminator_DSN, self).__init__()

        n_input_channel = in_ch
        if highpass:
            self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
            self.filter = self.filter_wavelet
            self.cs = cs
            n_input_channel = n_input_channel * 3 if self.cs == 'cat' else 3
            print('# FS type: {}, kernel size={}'.format('wavelet', kernel_size))
        else:
            self.filter = None

        self.net = DiscriminatorBasic(in_ch=n_input_channel, norm_layer=norm_layer)
        print('# Initializing FSSR-DiscriminatorBasic with {} norm layer'.format(norm_layer))

    def forward(self, x, y=None):
        if self.filter is not None:
            x = self.filter(x)
        x = self.net(x)
        if y is not None:
            x -= self.net(self.filter(y)).mean(0, keepdim=True)
        x = torch.sigmoid(x)
        return x

    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return (LH + HL + HH) / 3.
        elif self.cs.lower() == 'cat':
            return torch.cat((LH, HL, HH), 1)
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))


############################################################
#  Basic Discriminator
############################################################
class DiscriminatorBasic(nn.Module):
    def __init__(self, in_ch=3, norm_layer='Batch'):
        super(DiscriminatorBasic, self).__init__()
        if norm_layer == 'Batch':
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=(5, 5), padding=2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, kernel_size=(5, 5), padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 1, kernel_size=(1, 1))
            )
        elif norm_layer == 'Instance':
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=(5, 5), padding=2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, kernel_size=(5, 5), padding=2),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 1, kernel_size=(1, 1))
            )
        else:
            raise NotImplementedError('{} norm layer is not recognized'.format(norm_layer))

    def forward(self, x):
        return self.net(x)


############################################################
#  Densenet Discriminator
############################################################
class Discriminator_Densenet(nn.Module):

    def __init__(self, in_ch, num_features=64, compress_factor=2, expand_factor=4, growth_rate=32):
        super(Discriminator_Densenet, self).__init__()

        self.net = DenseNet(
            num_denseblock = [6, 12, 24, 16],
            channels = in_ch,
            class_count = 1,
            num_features = num_features,
            compress_factor = compress_factor,
            expand_factor = expand_factor,
            growth_rate=growth_rate
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':

    from torchsummary import summary

    model = Discriminator_Densenet(in_ch=1)

    summary(model, (1, 320, 320))