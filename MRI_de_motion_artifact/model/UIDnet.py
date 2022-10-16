import torch.nn as nn
import torch.optim as optim
import torch
from tools import utils, visualize
import numpy as np
from torch.nn import functional as F
import os


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
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
            model += [nn.Conv2d(2 * in_ch, num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.BatchNorm2d(2 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.BatchNorm2d(4 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.BatchNorm2d(8 * num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.AdaptiveAvgPool2d((1, 1))]

        elif self.model_name == "wgan-gp":
            model += [nn.Conv2d(in_ch, num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),

                      nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(5, 5), stride=(2, 2), padding=2),
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
            nn.Conv2d(32, num_out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_out_ch, num_out_ch, kernel_size=(1, 1)),
            nn.Tanh())

    def forward(self, x, z):
        input = torch.cat([x, z], dim=1)

        e1 = self.conv1(input)
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
        model = [nn.Conv2d(num_in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                 nn.ReLU(inplace=True)]
        # 15 layers, Conv + BN + relu
        for i in range(15):
            model += [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                      nn.BatchNorm2d(64),
                      nn.ReLU(inplace=True)]
        # last layer, Conv
        model += [nn.Conv2d(64, num_out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UIDnet(object):

    def __init__(self, args, data_loader, model_name='dcgan'):
        # Initialization
        self.lam = 10

        # Models
        self.g_model = Generator(num_in_ch=args.input_channels, num_out_ch=args.output_channels).cuda()
        self.d_model = Discriminator(in_ch=args.input_channels, num_feat=64, model_name=model_name).cuda()
        self.dnn_model = Dnn(num_in_ch=args.input_channels, num_out_ch=args.output_channels).cuda()

        self.ssim = utils.SSIM_loss().cuda()

        # dataset loder
        self.train_loader_A = data_loader[0]        # MA
        self.train_loader_B = data_loader[1]        # GT
        self.val_loader = data_loader[2]
        self.len_data_loader = min(len(self.train_loader_A), len(self.train_loader_B))

        # Optimizers
        params_G = list(self.g_model.parameters())
        params_D = list(self.d_model.parameters())
        params_dnn = list(self.dnn_model.parameters())
        self.optimizer_G = optim.Adam(params_G, lr=args.UID_lr_G, betas=(args.UID_beta1, args.UID_beta2), weight_decay=args.UID_weight_decay, amsgrad=True)
        self.optimizer_D = optim.Adam(params_D, lr=args.UID_lr_D, betas=(args.UID_beta1, args.UID_beta2), weight_decay=args.UID_weight_decay, amsgrad=True)
        self.optimizer_dnn = optim.Adam(params_dnn, lr=args.UID_lr_dnn, betas=(args.UID_beta1, args.UID_beta2), weight_decay=args.UID_weight_decay, amsgrad=True)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=args.UID_epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=args.UID_epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_dnn= optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dnn, T_max=args.UID_epochs, eta_min=1e-6, last_epoch=-1)

        # load weight
        if args.load_weight:
            self.load_weight()
        else:
            self.start_epoch = 0
        self.args = args
        self.model_name = model_name

    def load_weight(self):

        path = os.path.join(self.args.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path)
        self.g_model.load_state_dict(checkpoint['g_model'])
        self.d_model.load_state_dict(checkpoint['d_model'])
        self.dnn_model.load_state_dict(checkpoint['dnn_model'])
        self.start_epoch = checkpoint['epoch']

    def train(self):

        utils.create_training_log(os.path.join(self.args.snap_path, 'log.txt'))
        max_ssim = 0.0
        # for store visualization validation result during training
        if not os.path.exists(os.path.join(self.args.snap_path, 'visualize')):
            os.makedirs(os.path.join(self.args.snap_path, 'visualize'))

        for epoch in range(self.start_epoch, self.args.UID_epochs):

            # running [loss_D_total, loss_G_total, loss_dnn_total] record
            loss_list = [0.0, 0.0, 0.0]
            # generator
            self.g_model.train()
            # discriminator
            self.d_model.train()
            # denoising network
            self.dnn_model.train()

            for step, ((A), (B)) in enumerate(zip(self.train_loader_A, self.train_loader_B)):

                ########################
                #       data load      #
                ########################
                noisy = A.float().cuda(non_blocking=True)
                clean = B.float().cuda(non_blocking=True)

                z = torch.tensor(utils.get_random_sample(clean.shape)).float().cuda(non_blocking=True)

                real_label = torch.full((self.args.batch_size, 1), 1, dtype=noisy.dtype).cuda(non_blocking=True)
                fake_label = torch.full((self.args.batch_size, 1), 0, dtype=noisy.dtype).cuda(non_blocking=True)
                ###########################
                # (1) Update Disc network #
                ###########################
                self.d_model.zero_grad()
                for i in range(self.args.UID_n_disc):
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
                        gradient_penalty = utils.calc_gradient_penalty(self.d_model, noisy.data, G.data, self.args.batch_size)

                        d_loss_real = -torch.mean(D_real)
                        d_loss_fake = torch.mean(D_fake)
                        d_loss_gradient = gradient_penalty
                        loss_D_total = d_loss_real + d_loss_fake + self.lam * d_loss_gradient
                    else:
                        raise ValueError

                    loss_D_total.backward()
                    self.optimizer_D.step()
                ##########################
                # (2) Update Gen network #
                ##########################
                self.g_model.zero_grad()
                for i in range(self.args.UID_n_gen):
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
                ########################
                #     compute loss     #
                ########################
                loss_list[0] += loss_D_total.item()
                loss_list[1] += loss_G_total.item()
                loss_list[2] += loss_dnn_total.item()

                if (step + 1) % self.args.UID_log_step == 0:
                    print("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] dnn_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_dnn_total={:.5f}"
                          .format(epoch + 1,
                                  self.args.UID_epochs,
                                  step + 1,
                                  self.len_data_loader,
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_dnn.param_groups[0]['lr'],
                                  loss_list[0] / self.args.UID_log_step,
                                  loss_list[1] / self.args.UID_log_step,
                                  loss_list[2] / self.args.UID_log_step))
                    loss_list = [0.0, 0.0, 0.0]

            if self.args.UID_apply_scheduler:
                self.scheduler_D.step()
                self.scheduler_G.step()
                self.scheduler_dnn.step()

            mean_ssim, mean_psnr = self.validation(self.val_loader, epoch)
            print("Avg SSIM = {}, Avg PSNR = {}\n".format(mean_ssim, mean_psnr))

            if epoch == 0 or max_ssim < mean_ssim:
                max_ssim = mean_ssim
                weights_file_name = 'best_ssim_network_parameter.pth'
                weights_file = os.path.join(self.args.snap_path, weights_file_name)
                torch.save({
                    'epoch': epoch,
                    'g_model': self.g_model.state_dict(),
                    'd_model': self.d_model.state_dict(),
                    'dnn_model': self.dnn_model.state_dict()
                }, weights_file)
                print('save weights of epoch %d' % (epoch + 1) + '\n')
            print("\n")

    def validation(self, val_loader, epoch):

        self.dnn_model.eval()
        # random choose one for visualization and save it
        idx = np.random.choice(np.arange(len(val_loader)))
        with torch.no_grad():
            print('Validation:')
            ssim_eval, psnr_eval = 0, 0
            for i, data in enumerate(val_loader):
                noisy, clean = data[0].float().cuda(), data[1].float().cuda()
                dnn_out = noisy - self.dnn_model(noisy)
                ssim_eval += self.ssim(dnn_out, clean).item()
                psnr_eval += utils.calc_psnr_for_mri_image(dnn_out, clean).item()

                if i == idx:
                    image_A = noisy[0].permute(1, 2, 0).cpu().numpy()
                    image_B = clean[0].permute(1, 2, 0).cpu().numpy()
                    image_out = dnn_out[0].permute(1, 2, 0).cpu().numpy()
                    visualize.display_images(
                        images=[image_B, image_A, image_A - image_B, image_out - image_B, image_A - image_out,
                                image_out],
                        titles=['motion free', 'motion simulation', 'error map between 1 and 2',
                                'error map between 1 and 6', 'error map between 2 and 6', 'after correction'],
                        dir_save=os.path.join(self.args.snap_path, 'visualize', 'epoch={}.jpg'.format(epoch + 1)))
            mean_ssim = ssim_eval / len(val_loader)
            mean_psnr = psnr_eval / len(val_loader)
        return mean_ssim, mean_psnr

    def eval(self, test_set):

        self.load_weight()
        self.dnn_model.eval()
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        with torch.no_grad():
            for i in range(len(test_set)):
                noisy = torch.tensor(np.array([test_set[i]])).float().cuda()
                dnn_out = noisy - self.dnn_model(noisy)
                dnn_out = dnn_out[0].permute(1, 2, 0).cpu().numpy()
                np.savez(os.path.join(self.args.result_path, test_set.filename[i]), pred=dnn_out)
