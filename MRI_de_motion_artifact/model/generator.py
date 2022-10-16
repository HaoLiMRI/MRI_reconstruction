import torch.nn as nn
import torch
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from model.base_layer.Unet import U_Net


############################################################
#  base function
############################################################
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


############################################################
#  Resnet Generator
############################################################
class ResBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(num_feat, num_feat, (3, 3)),
                        # nn.InstanceNorm2d(num_feat),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(num_feat, num_feat, (3, 3)),
                        # nn.InstanceNorm2d(num_feat)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network".
    Such sub-pixel conv actually constructs F*S^2 feature maps of dimensions H*W are reshaped into F feature maps of dimensions H*S*W*S,
    where S is the upsampling factor.
    """
    def __init__(self, scale, n_feats):
        super(Upsampler, self).__init__()
        if scale == 1:
            self.upsampler = nn.Sequential(*[nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))])
        elif scale == 2:
            self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.PixelShuffle(scale)
                ])
        elif scale == 4:
            self.upsampler = nn.Sequential(*[
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.PixelShuffle(2),
                ])
        else:
            raise ValueError("scale must be 2 or 4.")

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x

class Generator_Resnet(nn.Module):
    def __init__(self, num_in_ch=6, num_feat=64, num_out_ch=3):
        super(Generator_Resnet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(num_in_ch, num_feat, (7, 7)),
                    # nn.InstanceNorm2d(num_feat),
                    nn.ReLU(inplace=True) ]
        # Downsampling
        model += [nn.Conv2d(num_feat, num_feat, (3, 3), stride=(2, 2), padding=(1, 1)),
                  # nn.InstanceNorm2d(num_feat*2),
                  nn.ReLU(inplace=True)]
        # Residual blocks
        n_residual_blocks = 12
        for _ in range(n_residual_blocks):
            model += [ResBlock(num_feat)]
        # Upsampling
        model += [Upsampler(2, num_feat)]
        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(num_feat, num_out_ch, (7, 7)),
                    # nn.ReLU(True)
                    nn.Tanh()
                    ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


############################################################
#  Residual in Residual Dense Block (RRDB) Generator
############################################################
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class Decoder_Id_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=3, scale=4, num_feat=64, num_block=5, num_grow_ch=16):
        super(Decoder_Id_RRDB, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class Decoder_SR_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=3, scale=2, num_feat=64, num_block=5, num_grow_ch=16):
        super(Decoder_SR_RRDB, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        # self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsampler = Upsampler(scale, num_feat)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        # feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.upsampler(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class Encoder_RRDB(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=16):
        super(Encoder_RRDB, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            #            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            #            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, img):
        featmap = self.conv_featmap(img)

        return featmap

class Encoder_LR_RRDB(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=16):
        super(Encoder_LR_RRDB, self).__init__()
        self.conv_featmap = nn.Sequential(
            nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            #            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
            #            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, img):
        featmap = self.conv_featmap(img)

        return featmap

class Generator_Id_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=3, scale=2, num_feat=64, num_block=5, num_grow_ch=16):
        super(Generator_Id_RRDB, self).__init__()

        self.encoder = Encoder_RRDB(num_in_ch, num_feat)
        self.decoder = Decoder_Id_RRDB(num_feat, num_out_ch, scale, num_feat, num_block, num_grow_ch)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Generator_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=3, scale=2, num_feat=64, num_block=5, num_grow_ch=16):
        super(Generator_RRDB, self).__init__()

        self.encoder = Encoder_LR_RRDB(num_in_ch, num_feat)
        self.decoder = Decoder_SR_RRDB(num_feat, num_out_ch, scale, num_feat, num_block, num_grow_ch)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


############################################################
#  Residual Channel Attention Network (RCAN) Generator
############################################################
"Channel Attention (CA) Layer"
class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer, is basical block in RCAN. One CA forms one RCAB(Residual Channel Attention Block).
    See figure 3 of original RCAN paper.
    Beware the CA Layer used in RCAN is actually same as the channel attention mechanism propsed in SENet(Squeeze-and-Excitation Networks).
    """
    def __init__(self, channel, reduction=16):
        """
        reduction is the r mentioned in 3.3 Channel Attention in RCAN paper
        """
        super(CALayer, self).__init__()
        # global average pooling(GAP): feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global average pooling(GAP), output size is 1 for each channel
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), # W_d in CA
                nn.ReLU(inplace=False), # ReLU in CA
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), # W_u in CA
                nn.Sigmoid() # sigmoid in CA
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # the x is the "feature maps over channels" in size C x H x W. The y now is actual the weights in size C x 1 x 1 which represents "channel statistics",
        # it stands for how much "attention" expected to pay for each channel's feature map
        return x * y

"Residual Channel Attention Block (RCAB)"
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2): # conv --> ReLU --> conv
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feat, reduction)) # use CA Layer

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) # conv --> ReLU --> conv --> CA/channel and spatial attention block
        #res = self.body(x).mul(self.res_scale)
        x = x + res # local skip link of RCAB
        return x

"Residual Group (RG)"
class ResidualGroup(nn.Module):
    """
    RG(Residual Group): There are several RGs belong to one RIR(Residual in Residual module).
    See upper figure in figure 2 of original RCAN paper
    """
    def __init__(self, conv, n_feat, kernel_size, reduction, res_scale, n_rcablocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, res_scale=1) \
            for _ in range(n_rcablocks)]

        modules_body.append(conv(n_feat, n_feat, kernel_size)) # last conv after several RCABs as show in figure 2
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # RCAB_1 --> RCAB_2 --> ... --> RCAB_(n_rcablocks) --> conv
        x = x + res # short skip connection in RG
        return x

"Residual Channel Attention Network (RCAN)"
class Decoder_Id_RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """

    def __init__(self, num_in_ch, n_resgroups=5, n_rcablocks=5, num_out_ch=3, num_feat=64, scale=1):
        super(Decoder_Id_RCAN, self).__init__()
        conv = default_conv

        kernel_size = 3  # conv filter size used for all conv in RCAN
        reduction = 16  # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [
            conv(num_in_ch, num_feat, kernel_size)]  # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(num_feat, num_feat, kernel_size)]

        # define body module for different type of network.
        # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
            ResidualGroup(
                conv, num_feat, kernel_size, reduction, res_scale=1, n_rcablocks=n_rcablocks) \
            for _ in range(n_resgroups)]

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = []
        """
        modules_tail.append(
                    Upsampler(scale, num_feat))
        """
        modules_tail.append(conv(num_feat, num_out_ch, kernel_size))
        # modules_tail.append(nn.ReLU(inplace=True))
        # modules_tail.append(nn.Tanh())
        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.sub_mean(x) """
        x = self.head(x)  # input image goes through first conv layer
        res = self.body(x)  # data goes through sveral ResidualGroups
        res = self.pretail(res)
        x = x + res  # long skip connection of RIR
        #        if (Maintain_Same_Size == True):
        #            res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
        x = self.tail(x)  # data goes through upsampling module and one more conv layer
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x

class Decoder_SR_RCAN(nn.Module):
    """
    RCAN(Deep Residual Channel Attention Network) = RIR(Residual in Residual module) + Upsampler Module.
    See bottom figure in figure 2 of original RCAN paper
    """

    def __init__(self, num_in_ch, n_resgroups=5, n_rcablocks=5, num_out_ch=6, scale=2, num_feat=64):
        super(Decoder_SR_RCAN, self).__init__()
        conv = default_conv

        kernel_size = 3  # conv filter size used for all conv in RCAN
        reduction = 16  # reduction is the r mentioned in 3.3 Channel Attention in RCAN paper

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        # define commonly use head module
        modules_head = [
            conv(num_in_ch, num_feat, kernel_size)]  # the first conv layer in RCAN, show in figure 2 of RCAN paper

        # define commonly use pretail module
        modules_pretail = [conv(num_feat, num_feat, kernel_size)]

        # define body module for different type of network.
        # The RIR(Residual in Residual) which consists of n_resgroups RGs, show in figure 2 of RCAN paper
        modules_body = [
            ResidualGroup(
                conv, num_feat, kernel_size, reduction, res_scale=1, n_rcablocks=n_rcablocks) \
            for _ in range(n_resgroups)]

        # define tail module. The last stage is upsampling module and one more conv layer, show in figure 2 of RCAN paper
        modules_tail = [Upsampler(scale, num_feat),
                        conv(num_feat, num_out_ch, kernel_size),
                        nn.ReLU(inplace=True)]

        # --------------------------------------we may NOT need this section------------------------------------------------------- #
        """ # don't know exactly what is doing here. However, it seems shifting the "rgb_range" to be somewhere in the mean
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1) """
        # ----------------------------------------------------------------------------------------------------------------------- #

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.pretail = nn.Sequential(*modules_pretail)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.sub_mean(x) """
        x = self.head(x)  # input image goes through first conv layer
        res = self.body(x)  # data goes through sveral ResidualGroups
        res = self.pretail(res)
        x = x + res  # long skip connection of RIR
        #        if (Maintain_Same_Size == True):
        #            res = self.down_size_converter(res) # extra down_size_converter is needed to shtik size of image scale times if we expect same size as input LR for SR output
        x = self.tail(x)  # data goes through upsampling module and one more conv layer
        # do NOT understand why need this, may NOT be useful for us
        """ x = self.add_mean(x) """
        return x


############################################################
#  Unet Generator
############################################################
class Generator_Unet(nn.Module):
    def __init__(self, num_in_ch=6, num_feat=64, num_out_ch=3):
        super(Generator_Unet, self).__init__()

        self.net = U_Net(in_ch=num_in_ch, out_ch=num_out_ch, num_feat=num_feat)

    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':

    from torchsummary import summary

    model = Decoder_Id_RCAN(num_in_ch=1, num_out_ch=1)

    summary(model, (1, 320, 320))
