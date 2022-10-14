import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from pytorch_wavelets import DWTForward

class DiscriminatorVGG(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(DiscriminatorVGG, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, stride=1, padding=1),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64 x 64 x 64
            nn.BatchNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*2, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 32 x 32
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*2, d*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*4, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 16 x 16
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*4, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 8 x 8
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d*8, d*8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 4 x 4
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.d*8) * self.feature_map_size * self.feature_map_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

"""
class UNetDiscriminator(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat*2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat*2, num_feat*4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat*4, num_feat*8, kernel_size=4, stride=2, padding=1, bias=False))

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat*8, num_feat*4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat*4, num_feat*2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat*2, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        self.conv9 = nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_coners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
"""

class Discriminator_DSN(nn.Module):
    def __init__(self, input_channel, kernel_size=5, highpass=False, norm_layer='Instance', cs='cat'):
        super(Discriminator_DSN, self).__init__()

        n_input_channel = input_channel
        if highpass:
            self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
            self.filter = self.filter_wavelet
            self.cs = cs
            n_input_channel = n_input_channel*3 if self.cs == 'cat' else 3
            print('# FS type: {}, kernel size={}'.format('wavelet', kernel_size))
        else:
            self.filter = None

        self.net = DiscriminatorBasic(n_input_channels=n_input_channel, norm_layer=norm_layer)
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


class DiscriminatorBasic(nn.Module):
    def __init__(self, n_input_channels=3, norm_layer='Batch'):
        super(DiscriminatorBasic, self).__init__()
        if norm_layer == 'Batch':
            self.net = nn.Sequential(
                nn.Conv2d(n_input_channels, 64, kernel_size=5, padding=2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 1, kernel_size=1)
            )
        elif norm_layer == 'Instance':
            self.net = nn.Sequential(
                nn.Conv2d(n_input_channels, 64, kernel_size=5, padding=2),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, kernel_size=5, padding=2),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 1, kernel_size=1)
            )
        else:
            raise NotImplementedError('{} norm layer is not recognized'.format(norm_layer))

    def forward(self, x):
        return self.net(x)