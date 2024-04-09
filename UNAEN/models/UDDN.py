import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import functools
import torch.optim as optim
import utils
import numpy as np
import os



class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.InstanceNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ChannelPool(nn.Module):

    @staticmethod
    def forward(x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2*2, ch_2*2 // r_2 , kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2*2 // r_2, ch_2*2, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 *2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        #g_ori = g
        # spatial attention for cnn branch
        g = torch.cat([g, x], 1)
        g_in = g

        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x = g
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([ x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
     zero = nn.ZeroPad2d,
  reflect = nn.ReflectionPad2d,
replicate = nn.ReplicationPad2d)

conv_dict = dict(
   conv2d = nn.Conv2d,
 deconv2d = nn.ConvTranspose2d)

norm_dict = dict(
     none = lambda x: lambda x: x,
 spectral = lambda x: lambda x: x,
    batch = nn.BatchNorm2d,
 instance = nn.InstanceNorm2d,
    layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
      relu = lambda: nn.ReLU(inplace=True),
      gelu = lambda: nn.GELU(),
     lrelu = lambda: nn.LeakyReLU(0.2,inplace=True),
     prelu = lambda: nn.PReLU(),
      selu = lambda: nn.SELU(),
      tanh = lambda: nn.Tanh())


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)   #接收padding这个参数
        self.conv = conv_dict[conv](**conv_opts)
        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)  #对outchannel做norm
        if norm == "spectral": self.conv = spectral_norm(self.conv)
        self.activ = activ_dict[activ]()

    def forward(self,x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block = block + [ConvolutionBlock(
                  in_channels = channels , out_channels=channels , kernel_size =3,
                  stride = 1, padding=1,norm=norm, activ=activ,pad=pad
        )]

        block = block + [ConvolutionBlock(
                  in_channels = channels , out_channels=channels , kernel_size =3,
                  stride = 1, padding=1,norm=norm, activ='none',pad=pad
        )]

        self.model = nn.Sequential(*block)

    def forward(self,x):
        return x + self.model(x)


class Encoder1(nn.Module):
    def __init__(self,  base_ch, num_down, num_residual, res_norm='instance', down_norm='instance',input_ch= 1):
        super(Encoder1, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x):
        sides = []
        for layers_id,layer in enumerate(self.layers):
            x = layer(x)
        return x,sides[::-1]


class Encoder2(nn.Module):
    def __init__(self,  base_ch, num_down, num_residual, res_norm='instance', down_norm='instance',input_ch= 1):
        super(Encoder2, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x, nce, layers=(0, 1, 2, 6)):
        sides = []
        if nce:
            for layer in self.layers:
                x = layer(x)
                sides.append(x)
        else:
            for layers_id,layer in enumerate(self.layers):
                x = layer(x)
                if layers_id in layers:
                    sides.append(x)
        return  sides[::-1]


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer', fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_ch = input_ch*2
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)  #怎么append

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch = input_ch//2
        #input_chs.append(128)
        m = ConvolutionBlock(
            in_channels=base_ch*2, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch*2)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                    nn.Conv2d(input_chs[i] * 2, input_chs[i], (1, 1)))   #通道数减半
            self.fuse = [getattr(self, "fuse{}".format(i)) for i in range(num_sides)]
                    #BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=512)
            self.fuse = lambda x, y, z, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y, z), 1))
        else:
            self.fuse = lambda x, y,z, i: x + y + z

    def forward(self, x, sides=(), sides_fre=()):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            x = self.layers[i](x)

        for i, j in enumerate(range(m - n, m)):
            if i == 0:
                x = self.layers[j](x)
            elif i==5:
                x = x.float()
                x = self.fuse(x, sides[i],sides_fre[i], i)
                x = self.layers[j](x)
            else:

                x = self.fuse(x, sides[i],sides_fre[i], i)
                x = self.layers[j](x)
        return x

    def forward1(self,sides, sides_fre):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"
        for i in range(m - n,m):
            if i == 0:
                x_lower = torch.cat((sides[i],sides_fre[i]),1)
                x = self.layers[i](x_lower)
            else:
                x = self.fuse(x, sides[i],sides_fre[i], i)
                x = self.layers[i](x)
        return x


class ENC_fea(nn.Module):
    def __init__(self, base_ch=64, num_down=2, num_residual=3,
        res_norm='instance', down_norm='instance'):
        super(ENC_fea, self).__init__()
        self.enc_spa = Encoder1(base_ch, num_down, num_residual, res_norm, down_norm,input_ch=1)
        self.enc_fre = Encoder1( base_ch, num_down, num_residual, res_norm, down_norm,input_ch=2)
        self.fusion = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=512)
    def forward(self,x):  #,x_fre
        #
        x = x.float()
        x_fre = torch.fft.fft2(x, norm='backward')
        x_fre = torch.cat((x_fre.real, x_fre.imag), 1)

        x,side = self.enc_spa(x)
        x_fre,side_fre = self.enc_fre(x_fre)
        #x = torch.cat([x,x_fre],1)
        x = self.fusion(x,x_fre)

        return x,side,side_fre


class ENC_side(nn.Module):
    def __init__(self, base_ch=64, num_down=2, num_residual=3,
        res_norm='instance', down_norm='instance'):
        super(ENC_side, self).__init__()
        self.enc_spa = Encoder2(base_ch, num_down, num_residual, res_norm, down_norm,input_ch=1)
        self.enc_fre = Encoder2( base_ch, num_down, num_residual, res_norm, down_norm,input_ch=2)
    def forward(self,x,nce=True):
        x = x.float()
        x_fre = torch.fft.fft(x, norm='backward')
        x_fre = torch.cat((x_fre.real, x_fre.imag), 1)

        sides = self.enc_spa(x,nce)
        sides_fre = self.enc_fre(x_fre,nce)

        return sides,sides_fre


class DEC(nn.Module):
    def __init__(self, input_ch=1,base_ch=64, num_down=2, num_residual=3, num_sides="all",
        res_norm='instance', up_norm='layer', fuse=True):
        super(DEC, self).__init__()
        self.n = num_down + num_residual+1  if num_sides == "all" else num_sides
        self.decoder = Decoder(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse)#,disen

    def forward(self, x, side=(), side1=(), disen=False):
        if disen:
            x = self.decoder.forward1(x,side)
        else:
            if len(side) != 0:
                x = self.decoder(x,side,side1)  #
            else:
                x = self.decoder(x)

        return x


class Dual_Domain_GEN(nn.Module):
    def __init__(self):
        super(Dual_Domain_GEN, self).__init__()

        self.enc_low = ENC_fea()
        self.enc_high = ENC_fea()
        self.enc_art = ENC_side()

        self.dec1 = DEC()   #for high
        self.dec2 = DEC()   #for low

    def forward1(self,x,y):   #input x: low quality   y: high quality
        side, side_fre = self.enc_art(x)  #  ,_   ,_
        x,_,_ = self.enc_low(x)
        y,_,_ = self.enc_high(y)

        y = self.dec2(y, side, side_fre)  # low images
        x = self.dec1(x)  #high images

        return x,y

    def forward2(self, x, y):
        side, side_fre  = self.enc_art(x)  #, side_fre   ,_
        x,_,_ = self.enc_low(x)
        y,_,_ = self.enc_high(y)

        x = self.dec1(x,side, side_fre)  # low images  , side_fre
        y = self.dec2(y)  # high images

        return x, y

    def disen(self,x):
        side, side_fre = self.enc_art(x)
        x = self.dec2(side,side_fre,disen=True)
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, norm_layer, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw), bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw), bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs):
        """Standard forward."""
        return self.model(inputs)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            return loss
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
            return loss


class UDDN(object):

    def __init__(self, config):

        self.lambda_idt = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.netG = Dual_Domain_GEN().to(config.device)
        self.netD_A = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer).to(config.device)
        self.netD_B = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer).to(config.device)

        # define loss functions
        self.criterionGAN = GANLoss('lsgan').to(config.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionCycle1 = torch.nn.L1Loss()
        self.criterionCycle2 = torch.nn.MSELoss()

        self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.netD_A.parameters()) + list(self.netD_B.parameters()), lr=0.0001, betas=(0.5, 0.999))

        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=10, gamma=0.5)

        self.config = config
        self.start_epoch = 0
        # load weight
        if config.load_weight:
            self.load_weight()


    def load_weight(self):

        path = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.config.device))
        self.netG.load_state_dict(checkpoint['netG'])
        self.netD_A.load_state_dict(checkpoint['netD_A'])
        self.netD_B.load_state_dict(checkpoint['netD_B'])
        self.start_epoch = checkpoint['epoch']


    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def train(self, data_loader):

        MA_loader = data_loader[0]  # MA: motion artifact
        MF_loader = data_loader[1]  # MF: motion free
        val_loader = data_loader[2]

        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        max_ssim = 0.0

        for epoch in range(self.start_epoch, self.config.epochs):
            loss_D, loss_G = 0., 0.

            self.netG.train()
            self.netD_A.train()
            self.netD_B.train()

            for step, ((A), (B)) in enumerate(zip(MA_loader, MF_loader)):

                ########################
                #       data load      #
                ########################
                real_A = A['ma_img'].to(self.config.device)
                real_B = B['gt_img'].to(self.config.device)

                # forward
                fake_B, fake_A = self.netG.forward1(real_A, real_B)
                rec_B, rec_A = self.netG.forward1(fake_A, fake_B)
                ###########################
                # (1) Update Gen network #
                ###########################
                self.set_requires_grad([self.netD_A, self.netD_B], False)
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                idt_B, idt_A = self.netG.forward2(real_A, real_B)
                loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_A * self.lambda_idt

                # GAN loss D_A(G_A(A))
                loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
                # GAN loss D_B(G_B(B))
                loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
                # Forward cycle loss || G_B(G_A(A)) - A||
                loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A
                # Backward cycle loss || G_A(G_B(B)) - B||
                loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B

                # ssim loss
                loss_ssim = (1 - utils.ssim(real_A, rec_A)) + (1 - utils.ssim(real_B, rec_B)) * 10

                # combined loss and calculate gradients
                loss_G_total = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_ssim + loss_idt_B
                loss_G_total.backward()
                self.optimizer_G.step()

                loss_G += loss_G_total.item()
                ###########################
                # (2) Update Disc network #
                ###########################
                self.set_requires_grad([self.netD_A, self.netD_B], True)
                self.optimizer_D.zero_grad()

                # Real
                pred_real_A = self.netD_A(real_B)
                pred_real_B = self.netD_B(real_A)
                loss_D_real_A = self.criterionGAN(pred_real_A, True)
                loss_D_real_B = self.criterionGAN(pred_real_B, True)
                # Fake
                pred_fake_A = self.netD_A(fake_B.detach())
                pred_fake_B = self.netD_B(fake_A.detach())
                loss_D_fake_A = self.criterionGAN(pred_fake_A, False)
                loss_D_fake_B = self.criterionGAN(pred_fake_B, False)
                # Combined loss and calculate gradients
                loss_D_total = (loss_D_real_A + loss_D_fake_A) * 0.5 + (loss_D_real_B + loss_D_fake_B) * 0.5
                loss_D_total.backward()
                self.optimizer_D.step()

                loss_D += loss_D_total.item()
                ########################
                #     record loss      #
                ########################
                if (step + 1) % self.config.log_step == 0:
                    log_file.write("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]: loss_D_total={:.5f}  loss_G_total={:.5f}\n"
                                   .format(epoch + 1,
                                           self.config.epochs,
                                           step + 1,
                                           len(MA_loader),
                                           self.optimizer_G.param_groups[0]['lr'],
                                           loss_D / self.config.log_step,
                                           loss_G / self.config.log_step))
                    print("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]: loss_D_total={:.5f}  loss_G_total={:.5f}"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step))
                    loss_D, loss_G = 0., 0.

            self.scheduler_D.step()
            self.scheduler_G.step()
            ########################
            #     validation       #
            ########################
            self.netG.eval()
            with torch.no_grad():
                print('Validation:')
                ssim_eval, psnr_eval = 0, 0
                for i, batch in enumerate(val_loader):
                    real_A = batch['ma_img'].to(self.config.device)
                    real_B = batch['gt_img'].to(self.config.device)
                    fake_B, _ = self.netG.forward1(real_A, real_B)
                    ssim_eval += utils.ssim(fake_B, real_B).item()
                    psnr_eval += utils.psnr(fake_B, real_B).item()
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                log_file.write("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))
                print("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))

                if epoch == 0 or max_ssim < ssim_eval:
                    max_ssim = ssim_eval
                    weights_file = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
                    torch.save({
                        'epoch': epoch,
                        'netG': self.netG.state_dict(),
                        'netD_A': self.netD_A.state_dict(),
                        'netD_B': self.netD_B.state_dict(),
                    }, weights_file)
                    log_file.write('save weights of epoch %d' % (epoch + 1) + '\n')
                    print('save weights of epoch %d' % (epoch + 1) + '\n')

            print('\n')
            log_file.write('\n')
        log_file.close()


    def eval(self, test_loader):

        self.load_weight()
        self.netG.eval()
        if not os.path.exists(self.config.inference_path):
            os.makedirs(self.config.inference_path)
        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        with torch.no_grad():
            ma_ssim, ma_psnr = 0., 0.
            eval_ssim, eval_psnr = 0, 0
            for i, batch in enumerate(test_loader):

                real_A = batch['ma_img'].to(self.config.device)
                real_B = batch['gt_img'].to(self.config.device)
                fake_B, _ = self.netG.forward1(real_A, real_B)

                ma_ssim += utils.ssim(real_A, real_B).item()
                ma_psnr += utils.psnr(real_A, real_B).item()
                eval_ssim += utils.ssim(fake_B, real_B).item()
                eval_psnr += utils.psnr(fake_B, real_B).item()

                fake_B = fake_B[0, 0].cpu().numpy()
                np.savez(os.path.join(self.config.inference_path, batch['filename'][0]), pred=fake_B)

        num = len(test_loader)

        print('\nBefore Reduction: ssim=%.4f and psnr=%.4f' % (ma_ssim / num, ma_psnr / num))
        log_file.write('\nBefore Reduction: ssim=%.4f and psnr=%.4f\n' % (ma_ssim / num, ma_psnr / num))

        print('\nEvaluation: ssim=%.4f and psnr=%.4f' % (eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: ssim=%.4f and psnr=%.4f\n' % (eval_ssim / num, eval_psnr / num))

        log_file.close()