import torch.nn as nn
import torch.optim as optim
import torch
import utils
import numpy as np
from torch.nn import functional as F
import os



class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class diff(nn.Module):
    # Calculate the image gradient - the sharpening technique
    def __init__(self):
        super(diff, self).__init__()
        self.ave_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        g1 = self.ave_pool(x)
        out = torch.cat([x, g1], dim=1)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, num_feat=64, model_name='dcgan'):
        super(Discriminator, self).__init__()
        model = []
        if model_name == 'dcgan':
            model += [nn.Conv2d(2 * in_ch, num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(2 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(4 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(8 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.AdaptiveAvgPool2d((1, 1))]

        elif self.model_name == "wgan-gp":
            model += [nn.Conv2d(in_ch, num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.AdaptiveAvgPool2d((1, 1))]
        else: raise ValueError
        self.diff = diff()
        self.features = nn.Sequential(*model)
        self.classifier = nn.Linear(8 * num_feat, 1)

    def forward(self, x):
        # image sharpening technique
        x = self.diff(x)
        # discriminator
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class Generator(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1):
        super(Generator, self).__init__()

        self.conv1 = conv_block(2 * num_in_ch, 32)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(32, 64)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(64, 128)

        self.dropout = nn.Dropout2d(0.5)

        self.up4 = up_conv(128, 64)
        self.up_conv4 = conv_block(128, 64)
        self.up5 = up_conv(64, 32)
        self.up_conv5 = conv_block(64, 32)

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, num_out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_out_ch, num_out_ch, kernel_size=(1, 1)),
            nn.Tanh())

    def forward(self, x, z):
        inputs = torch.cat([x, z], dim=1)

        e1 = self.conv1(inputs)
        e2 = self.conv2(self.Maxpool1(e1))
        e3 = self.dropout(self.conv3(self.Maxpool2(e2)))

        d4 = self.up4(e3)
        d4 = torch.cat([e2, d4], dim=1)
        d4 = self.up_conv4(d4)

        d5 = self.up5(d4)
        d5 = torch.cat([e1, d5], dim=1)
        d5 = self.up_conv5(d5)

        out = self.conv5(d5)
        return out

class Dnn(nn.Module):
    # denoising network
    def __init__(self, num_in_ch=1, num_out_ch=1):
        super(Dnn, self).__init__()
        # 1st layer, Conv + relu
        model = [nn.Conv2d(num_in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.ReLU(inplace=True)]
        # 15 layers, Conv + BN + relu
        for i in range(15):
            model += [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                      nn.BatchNorm2d(64),
                      nn.ReLU(inplace=True)]
        # last layer, Conv
        model += [nn.Conv2d(64, num_out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UIDnet(object):

    def __init__(self, config, model_name='dcgan'):

        # Initialization
        self.lam = 10

        # Models
        self.g_model = Generator(num_in_ch=1, num_out_ch=1).to(config.device)
        self.d_model = Discriminator(in_ch=1, num_feat=64, model_name=model_name).to(config.device)
        self.dnn_model = Dnn(num_in_ch=1, num_out_ch=1).to(config.device)

        # Optimizers
        params_G = list(self.g_model.parameters())
        params_D = list(self.d_model.parameters())
        params_dnn = list(self.dnn_model.parameters())
        self.optimizer_G = optim.Adam(params_G, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_D = optim.Adam(params_D, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_dnn = optim.Adam(params_dnn, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_dnn= optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dnn, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)

        self.config = config
        self.model_name = model_name
        self.start_epoch = 0
        # load weight
        if config.load_weight:
            self.load_weight()


    def load_weight(self):

        path = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.config.device))
        self.g_model.load_state_dict(checkpoint['g_model'])
        self.d_model.load_state_dict(checkpoint['d_model'])
        self.dnn_model.load_state_dict(checkpoint['dnn_model'])
        self.start_epoch = checkpoint['epoch']


    def train(self, data_loader):

        MA_loader = data_loader[0]             # MA: motion artifact
        MF_loader = data_loader[1]             # MF: motion free
        val_loader = data_loader[2]

        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        max_ssim = 0.0

        for epoch in range(self.start_epoch, self.config.epochs):

            loss_D, loss_G, loss_dnn = 0., 0., 0.
            # generator
            self.g_model.train()
            # discriminator
            self.d_model.train()
            # denoising network
            self.dnn_model.train()

            for step, ((A), (B)) in enumerate(zip(MA_loader, MF_loader)):

                ########################
                #       data load      #
                ########################
                noisy = A['ma_img'].to(self.config.device)
                clean = B['gt_img'].to(self.config.device)

                z = torch.tensor(utils.get_random_sample(clean.shape)).float().to(self.config.device)

                real_label = torch.full((self.config.batch_size, 1), 1, dtype=noisy.dtype).to(self.config.device)
                fake_label = torch.full((self.config.batch_size, 1), 0, dtype=noisy.dtype).to(self.config.device)
                ###########################
                # (1) Update Disc network #
                ###########################
                self.d_model.zero_grad()
                # generate pseudo-noisy data
                g_noise = self.g_model(clean, z)
                G = clean + g_noise

                D_real = self.d_model(noisy.detach())
                D_fake = self.d_model(G.detach())

                if self.model_name == 'dcgan':
                    # dcgan discriminator loss function
                    d_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D_real, real_label))
                    d_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D_fake, fake_label))
                    loss_D_total = d_loss_real + d_loss_fake
                elif self.model_name == 'wgan-gp':
                    # wgan discriminator loss function
                    gradient_penalty = utils.calc_gradient_penalty(self.d_model, noisy.data, G.data, self.config.batch_size)
                    d_loss_real = -torch.mean(D_real)
                    d_loss_fake = torch.mean(D_fake)
                    d_loss_gradient = gradient_penalty
                    loss_D_total = d_loss_real + d_loss_fake + self.lam * d_loss_gradient
                else:
                    raise ValueError

                loss_D_total.backward()
                self.optimizer_D.step()

                loss_D += loss_D_total.item()
                ##########################
                # (2) Update Gen network #
                ##########################
                self.g_model.zero_grad()
                # generate pseudo-noisy data
                g_noise = self.g_model(clean, z)
                G = clean + g_noise
                D_fake = self.d_model(G)

                if self.model_name == 'dcgan':
                    # dcgan generator loss function
                    loss_G_total = torch.mean(F.binary_cross_entropy_with_logits(D_fake, real_label))
                elif self.model_name == 'wgan-gp':
                    # wgan generator loss function
                    loss_G_total = -torch.mean(D_fake)
                else:
                    raise ValueError

                loss_G_total.backward()
                self.optimizer_G.step()

                loss_G += loss_G_total.item()
                ################################
                # (3) Update denoising network #
                ################################
                self.dnn_model.zero_grad()
                # generate pseudo-noisy data
                g_noise = self.g_model(clean, z)
                G = clean + g_noise
                # denoising
                dnn_pred = self.dnn_model(G.detach())
                out = G - dnn_pred
                # dnn model loss function
                loss_dnn_total = F.mse_loss(out, clean)  # L2 Loss
                loss_dnn_total.backward()
                self.optimizer_dnn.step()

                loss_dnn += loss_dnn_total.item()
                ########################
                #     record loss      #
                ########################
                if (step + 1) % self.config.log_step == 0:
                    log_file.write("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] dnn_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_dnn_total={:.5f}\n"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_dnn.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step,
                                  loss_dnn / self.config.log_step))
                    print("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] dnn_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_dnn_total={:.5f}"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_dnn.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step,
                                  loss_dnn / self.config.log_step))
                    loss_D, loss_G, loss_dnn = 0., 0., 0.

            self.scheduler_D.step()
            self.scheduler_G.step()
            self.scheduler_dnn.step()

            ########################
            #     validation       #
            ########################
            self.dnn_model.eval()
            with torch.no_grad():
                ssim_eval, psnr_eval = 0, 0
                for i, batch in enumerate(val_loader):
                    noisy = batch['ma_img'].to(self.config.device)
                    clean = batch['gt_img'].to(self.config.device)
                    dnn_out = noisy - self.dnn_model(noisy)
                    ssim_eval += utils.ssim(dnn_out, clean).item()
                    psnr_eval += utils.psnr(dnn_out, clean).item()
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                log_file.write("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))
                print("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))

                if epoch == 0 or max_ssim < ssim_eval:
                    max_ssim = ssim_eval
                    weights_file = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
                    torch.save({
                        'epoch': epoch,
                        'g_model': self.g_model.state_dict(),
                        'd_model': self.d_model.state_dict(),
                        'dnn_model': self.dnn_model.state_dict()
                    }, weights_file)
                    log_file.write('save weights of epoch %d' % (epoch + 1) + '\n')
                    print('save weights of epoch %d' % (epoch + 1) + '\n')

            print("\n")
            log_file.write("\n")
        log_file.close()


    def eval(self, test_loader):
        self.load_weight()
        self.dnn_model.eval()
        if not os.path.exists(self.config.inference_path):
            os.makedirs(self.config.inference_path)
        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        with torch.no_grad():
            ma_ssim, ma_psnr = 0., 0.
            eval_ssim, eval_psnr = 0, 0
            for i, batch in enumerate(test_loader):

                noisy = batch['ma_img'].to(self.config.device)
                clean = batch['gt_img'].to(self.config.device)

                dnn_out = noisy - self.dnn_model(noisy)

                ma_ssim += utils.ssim(noisy, clean).item()
                ma_psnr += utils.psnr(noisy, clean).item()
                eval_ssim += utils.ssim(dnn_out, clean).item()
                eval_psnr += utils.psnr(dnn_out, clean).item()

                dnn_out = dnn_out[0, 0].cpu().numpy()
                np.savez(os.path.join(self.config.inference_path, batch['filename'][0]), pred=dnn_out)

        num = len(test_loader)

        print('\nBefore Reduction: ssim=%.4f and psnr=%.4f' % (ma_ssim / num, ma_psnr / num))
        log_file.write('\nBefore Reduction: ssim=%.4f and psnr=%.4f\n' % (ma_ssim / num, ma_psnr / num))

        print('\nEvaluation: ssim=%.4f and psnr=%.4f' % (eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: ssim=%.4f and psnr=%.4f\n' % (eval_ssim / num, eval_psnr / num))

        log_file.close()