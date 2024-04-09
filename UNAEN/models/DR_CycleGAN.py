from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torchvision
import utils
import numpy as np
import os


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # if x.size(0) == 1:
        #     # These two lines run much faster in pytorch 0.4 than the two lines listed below.
        #     mean = x.view(-1).mean().view(*shape)
        #     std = x.view(-1).std().view(*shape)
        # else:
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim , output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        self.norm_type = norm
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'wn':
            self.conv = weight_norm(self.conv)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # self.acon = MetaAconC(width = output_dim)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm_type != 'wn' and self.norm is not None:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)
            # x = self.acon(x)
        return x

class vgg_19(nn.Module):
    def __init__(self):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:20])
    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for _ in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Encoder and Decoders
##################################################################################
class NoiseEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(NoiseEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for _ in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, (1, 1), (1, 1), (0, 0))]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for _ in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for _ in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='sigmoid', pad_type=pad_type)]  # tanh
        # self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)] # tanh
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Generator
##################################################################################
class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim):
        super(VAEGen, self).__init__()
        dim = 64
        n_downsample = 2
        n_res = 4
        activ = 'relu'
        pad_type = 'reflect'

        # content encoder
        # Replace traditional instance normalization layer (IN) for image translation with batch normalization (BN).
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'bn', activ,
                                  pad_type=pad_type)  # replace 'in' with 'bn'
        self.styc = NoiseEncoder(n_downsample, input_dim, dim, self.enc.output_dim, 'bn', activ,
                                 pad_type=pad_type)  # use similar codes with style encoder
        self.dec_cont = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='bn', activ=activ,
                                pad_type=pad_type)  # 'in'
        self.dec_recs = Decoder(n_downsample, n_res, 2 * self.enc.output_dim, input_dim, res_norm='bn', activ=activ,
                                pad_type=pad_type)  # 'in'

    def encode_cont(self, images):
        hiddens = self.enc(images)
        return hiddens

    def encode_sty(self, images):
        styhiddens = self.styc(images)
        return styhiddens

    def decode_cont(self, hiddens):
        images = self.dec_cont(hiddens)
        return images

    def decode_recs(self, hiddens):
        images = self.dec_recs(hiddens)
        return images

##################################################################################
# Discriminator
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim):
        super(MsImageDis, self).__init__()
        self.n_layer = 4
        self.gan_type = 'nsgan'
        self.dim = 64
        self.norm = 'bn'
        self.activ = 'relu'
        self.num_scales = 3
        self.pad_type = 'reflect'
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for _ in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, (1, 1), (1, 1), (0, 0))]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Model
##################################################################################
class DR_CycleGAN(object):

    def __init__(self, config):

        self.n_disc = 3
        self.n_gen = 1

        # Initiate the networks
        self.gen_a = VAEGen(input_dim=1).to(config.device)  # auto-encoder for domain a
        self.gen_b = VAEGen(input_dim=1).to(config.device)  # auto-encoder for domain b
        self.dis_a = MsImageDis(input_dim=1).to(config.device)  # discriminator for domain a
        self.dis_b = MsImageDis(input_dim=1).to(config.device)  # discriminator for domain b

        # optimizer
        params_G = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        params_D = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        self.optimizer_G = optim.Adam(params_G, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_D = optim.Adam(params_D, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=10, gamma=0.5)

        self.config = config
        self.start_epoch = 0
        # load weight
        if config.load_weight:
            self.load_weight()

        self.vgg = vgg_19().to(config.device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)


    def load_weight(self):

        path = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.config.device))
        self.gen_a.load_state_dict(checkpoint['gen_a'])
        self.gen_b.load_state_dict(checkpoint['gen_b'])
        self.dis_a.load_state_dict(checkpoint['dis_a'])
        self.dis_b.load_state_dict(checkpoint['dis_b'])
        self.start_epoch = checkpoint['epoch']


    def recon_criterion(self, inputs, target):
        loss1 = torch.mean(torch.abs(inputs - target))

        img_vgg = inputs.repeat(1, 3, 1, 1)
        target_vgg = target.repeat(1, 3, 1, 1)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        loss2 = torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
        return loss1 + 0.5 * loss2


    def train(self, data_loader):

        MA_loader = data_loader[0]             # MA: motion artifact
        MF_loader = data_loader[1]             # MF: motion free
        val_loader = data_loader[2]

        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        max_ssim = 0.0

        for epoch in range(self.start_epoch, self.config.epochs):

            loss_D, loss_G = 0., 0.
            # generator
            self.gen_a.train()
            self.gen_b.train()
            # discriminator
            self.dis_a.train()
            self.dis_b.train()

            for step, ((A), (B)) in enumerate(zip(MA_loader, MF_loader)):

                ########################
                #       data load      #
                ########################
                x_a = A['ma_img'].to(self.config.device)
                x_b = B['gt_img'].to(self.config.device)

                ###########################
                # (1) Update Disc network #
                ###########################
                self.optimizer_D.zero_grad()
                for i in range(self.n_disc):
                    # encode
                    h_a = self.gen_a.encode_cont(x_a)       # content
                    h_a_sty = self.gen_a.encode_sty(x_a)    # artifact
                    h_b = self.gen_b.encode_cont(x_b)

                    h_cat = torch.cat((h_b, h_a_sty), 1)
                    x_ba = self.gen_a.decode_recs(h_cat)
                    x_ab = self.gen_b.decode_cont(h_a)

                    # D loss
                    loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
                    loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

                    loss_dis_total = loss_dis_a + loss_dis_b
                    loss_dis_total.backward()
                    self.optimizer_D.step()

                    loss_D += loss_dis_total.item() / self.n_disc
                ##########################
                # (2) Update Gen network #
                ###########################
                self.optimizer_G.zero_grad()
                for i in range(self.n_gen):
                    h_a = self.gen_a.encode_cont(x_a)
                    h_b = self.gen_b.encode_cont(x_b)
                    h_a_sty = self.gen_a.encode_sty(x_a)

                    h_a_cont = torch.cat((h_a, h_a_sty), 1)
                    x_a_recon = self.gen_a.decode_recs(h_a_cont)
                    x_b_recon = self.gen_b.decode_cont(h_b)

                    h_ba_cont = torch.cat((h_b, h_a_sty), 1)

                    x_ba = self.gen_a.decode_recs(h_ba_cont)
                    x_ab = self.gen_b.decode_cont(h_a)

                    # encode again
                    h_b_recon = self.gen_a.encode_cont(x_ba)
                    h_b_sty_recon = self.gen_a.encode_sty(x_ba)

                    h_a_recon = self.gen_b.encode_cont(x_ab)

                    # decode again (if needed)
                    h_a_cat_recs = torch.cat((h_a_recon, h_b_sty_recon), 1)

                    x_aba = self.gen_a.decode_recs(h_a_cat_recs)
                    x_bab = self.gen_b.decode_cont(h_b_recon)

                    # reconstruction loss
                    loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
                    loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

                    loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
                    loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)

                    # GAN loss
                    loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
                    loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

                    # my_sum_loss = 0
                    # for index in range(x_a.shape[0]):
                    #     input_tem = x_a[index, :, :, :].squeeze()
                    #     target_tem = x_ab[index, :, :, :].squeeze()
                    #     sum_a_ori = torch.sum(input_tem, dim=0)
                    #     sum_a_now = torch.sum(target_tem, dim=0)
                    #     max_a_ori = torch.max(sum_a_ori)
                    #     max_a_now = torch.max(sum_a_now)
                    #     sum_a_ori = sum_a_ori / max_a_ori
                    #     sum_a_now = sum_a_now / max_a_now
                    #     my_sum_loss = my_sum_loss + torch.sum(abs(sum_a_ori - sum_a_now))
                    # for index in range(x_b.shape[0]):
                    #     input_tem = x_b[index, :, :, :].squeeze()
                    #     target_tem = x_ba[index, :, :, :].squeeze()
                    #     sum_a_ori = torch.sum(input_tem, dim=0)
                    #     sum_a_now = torch.sum(target_tem, dim=0)
                    #     max_a_ori = torch.max(sum_a_ori)
                    #     max_a_now = torch.max(sum_a_now)
                    #     sum_a_ori = sum_a_ori / max_a_ori
                    #     sum_a_now = sum_a_now / max_a_now
                    #     my_sum_loss = my_sum_loss + torch.sum(abs(sum_a_ori - sum_a_now))
                    # my_sum_loss = my_sum_loss / (x_b.shape[0] * 2.0)

                    loss_gen_total = loss_gen_adv_a + loss_gen_adv_b + \
                                     10* loss_gen_recon_x_a + 10 * loss_gen_recon_x_b + \
                                     10 * loss_gen_cyc_x_a + 10 * loss_gen_cyc_x_b #+ \
                                     # 0.5 * my_sum_loss

                    loss_gen_total.backward()
                    self.optimizer_G.step()

                    loss_G += loss_gen_total.item() / self.n_gen
                ########################
                #     record loss      #
                ########################
                if (step + 1) % self.config.log_step == 0:
                    log_file.write(
                        "Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]: loss_D_total={:.5f}  loss_G_total={:.5f}\n"
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
            self.gen_a.eval()
            self.gen_b.eval()
            with torch.no_grad():
                print('Validation:')
                ssim_eval, psnr_eval = 0, 0
                for i, batch in enumerate(val_loader):
                    x_a = batch['ma_img'].to(self.config.device)
                    x_b = batch['gt_img'].to(self.config.device)
                    h_a = self.gen_a.encode_cont(x_a)
                    x_ab = self.gen_b.decode_cont(h_a)
                    ssim_eval += utils.ssim(x_ab, x_b).item()
                    psnr_eval += utils.psnr(x_ab, x_b).item()
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                log_file.write("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))
                print("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))

                if epoch == 0 or max_ssim < ssim_eval:
                    max_ssim = ssim_eval
                    weights_file = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
                    torch.save({
                        'epoch': epoch,
                        'gen_a': self.gen_a.state_dict(),
                        'gen_b': self.gen_b.state_dict(),
                        'dis_a': self.dis_a.state_dict(),
                        'dis_b': self.dis_b.state_dict(),
                        }, weights_file)
                    log_file.write('save weights of epoch %d' % (epoch + 1) + '\n')
                    print('save weights of epoch %d' % (epoch + 1) + '\n')

            print('\n')
            log_file.write('\n')
        log_file.close()


    def eval(self, test_loader):

        self.load_weight()
        self.gen_a.eval()
        self.gen_b.eval()
        if not os.path.exists(self.config.inference_path):
            os.makedirs(self.config.inference_path)
        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        with torch.no_grad():
            ma_ssim, ma_psnr = 0., 0.
            eval_ssim, eval_psnr = 0, 0
            for i, batch in enumerate(test_loader):

                x_a = batch['ma_img'].to(self.config.device)
                x_b = batch['gt_img'].to(self.config.device)

                h_a = self.gen_a.encode_cont(x_a)
                x_ab = self.gen_b.decode_cont(h_a)

                ma_ssim += utils.ssim(x_a, x_b).item()
                ma_psnr += utils.psnr(x_a, x_b).item()
                eval_ssim += utils.ssim(x_ab, x_b).item()
                eval_psnr += utils.psnr(x_ab, x_b).item()

                x_ab = x_ab[0, 0].cpu().numpy()
                np.savez(os.path.join(self.config.inference_path, batch['filename'][0]), pred=x_ab)

        num = len(test_loader)

        print('\nBefore Reduction: ssim=%.4f and psnr=%.4f' % (ma_ssim / num, ma_psnr / num))
        log_file.write('\nBefore Reduction: ssim=%.4f and psnr=%.4f\n' % (ma_ssim / num, ma_psnr / num))

        print('\nEvaluation: ssim=%.4f and psnr=%.4f' % (eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: ssim=%.4f and psnr=%.4f\n' % (eval_ssim / num, eval_psnr / num))

        log_file.close()