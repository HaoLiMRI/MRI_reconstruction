import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.base_layer.radam import RAdam
from model.base_layer.BatchInstanceNorm import BatchInstanceNorm2d
from tools import utils, visualize
import numpy as np
import os

class ResBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResBlock, self).__init__()
        conv_block = [nn.Conv2d(num_feat, num_feat, (3, 3), padding=1),
                      BatchInstanceNorm2d(num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                      nn.Conv2d(num_feat, num_feat, (3, 3), padding=1),
                      BatchInstanceNorm2d(num_feat)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Extractor(nn.Module):
    def __init__(self, num_in_ch=1, num_feat=16, num_out_ch=1, n_layers=20):
        super(Extractor, self).__init__()

        model = [nn.Conv2d(num_in_ch, num_feat, kernel_size=(3,3), padding=1),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(0, n_layers - 2):
            model += [nn.Conv2d(num_feat, num_feat, kernel_size=(3,3), padding=1),
                      BatchInstanceNorm2d(num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        model += [nn.Conv2d(num_feat, num_out_ch, kernel_size=(3,3), padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=1, num_feat=16):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
            BatchInstanceNorm2d(2 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
            BatchInstanceNorm2d(4 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
            BatchInstanceNorm2d(8 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(8 * num_feat, 16 * num_feat, kernel_size=(4, 4), stride=(1, 1)),
            BatchInstanceNorm2d(16 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(16 * num_feat, 1)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class Generator(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=32, nres=3):
        super(Generator, self).__init__()

        model = [nn.Conv2d(num_in_ch, num_feat, kernel_size=(7, 7), stride=(1, 1), padding=3),
                 BatchInstanceNorm2d(num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True),

                 nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
                 BatchInstanceNorm2d(2 * num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True),

                 nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=1),
                 BatchInstanceNorm2d(4 * num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for i in range(nres):
            model.append(ResBlock(4 * num_feat))

        model += [nn.ConvTranspose2d(4 * num_feat, 2 * num_feat, kernel_size=(2, 2), stride=(2, 2)),
                  BatchInstanceNorm2d(2 * num_feat),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                  nn.ConvTranspose2d(2 * num_feat, num_feat, kernel_size=(2, 2), stride=(2, 2)),
                  BatchInstanceNorm2d(num_feat),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                  nn.Conv2d(num_feat, num_out_ch, kernel_size=(7, 7), stride=(1, 1), padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ISCL(object):

    def __init__(self, args, data_loader):
        # Initialization
        self.gamma = 0.5
        self.l = 30

        # Models
        self.G = Generator(num_in_ch=args.input_channels, num_out_ch=args.output_channels, num_feat=32).cuda() # Clean image -> Noisy image
        self.F = Generator(num_in_ch=args.input_channels, num_out_ch=args.output_channels, num_feat=32).cuda() # Noisy image -> Clean image
        self.H = Extractor(num_in_ch=args.input_channels, num_out_ch=args.output_channels, num_feat=32).cuda() # Noisy image -> Noise map
        self.DX = Discriminator(in_ch=args.input_channels, num_feat=32).cuda() # Noisy image discriminator
        self.DY = Discriminator(in_ch=args.input_channels, num_feat=32).cuda() # Clean image discriminator

        self.ssim = utils.SSIM_loss().cuda()

        # dataset loder
        self.train_loader_A = data_loader[0]        # MA
        self.train_loader_B = data_loader[1]        # GT
        self.val_loader = data_loader[2]
        self.len_data_loader = min(len(self.train_loader_A), len(self.train_loader_B))

        # Optimizers
        params_G = list(self.G.parameters()) + list(self.F.parameters())
        params_D = list(self.DX.parameters()) + list(self.DY.parameters())
        params_H = list(self.H.parameters())
        self.optimizer_G = RAdam(params_G, lr=args.ISCL_lr_G, betas=(args.ISCL_beta1, args.ISCL_beta2), weight_decay=args.ISCL_weight_decay)
        self.optimizer_D = RAdam(params_D, lr=args.ISCL_lr_D, betas=(args.ISCL_beta1, args.ISCL_beta2), weight_decay=args.ISCL_weight_decay)
        self.optimizer_H = RAdam(params_H, lr=args.ISCL_lr_H, betas=(args.ISCL_beta1, args.ISCL_beta2), weight_decay=args.ISCL_weight_decay)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=args.ISCL_epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=args.ISCL_epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_H= optim.lr_scheduler.CosineAnnealingLR(self.optimizer_H, T_max=args.ISCL_epochs, eta_min=1e-6, last_epoch=-1)

        # load weight
        if args.load_weight:
            self.load_weight()
        else:
            self.start_epoch = 0
        self.args = args

    def load_weight(self):

        path = os.path.join(self.args.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G'])
        self.F.load_state_dict(checkpoint['F'])
        self.H.load_state_dict(checkpoint['H'])
        self.DX.load_state_dict(checkpoint['DX'])
        self.DY.load_state_dict(checkpoint['DY'])
        self.start_epoch = checkpoint['epoch']

    def train(self):

        utils.create_training_log(os.path.join(self.args.snap_path, 'log.txt'))
        max_ssim = 0.0
        # for store visualization validation result during training
        if not os.path.exists(os.path.join(self.args.snap_path, 'visualize')):
            os.makedirs(os.path.join(self.args.snap_path, 'visualize'))

        for epoch in range(self.start_epoch, self.args.ISCL_epochs):

            # running [loss_D_total, loss_G_total, loss_H_total] record
            loss_list = [0.0, 0.0, 0.0]
            # generator
            self.G.train()
            self.F.train()
            # discriminator
            self.DX.train()
            self.DY.train()
            # extractor
            self.H.train()

            for step, ((A), (B)) in enumerate(zip(self.train_loader_A, self.train_loader_B)):

                ########################
                #       data load      #
                ########################
                noisy = A.float().cuda(non_blocking=True)
                clean = B.float().cuda(non_blocking=True)

                ###########################
                # (1) Update Gen network #
                ###########################
                self.G.zero_grad()
                self.F.zero_grad()
                for i in range(self.args.ISCL_n_gen):
                    y_hat_i = self.F(noisy)     # trainable
                    x_hat_j = self.G(clean)     # trainable
                    y_bar_i = noisy - self.H(noisy)    # untrainable
                    x_bar_j = clean + self.H(noisy)    # untrainable
                    y_hat_j = self.F(x_bar_j)       # trainable
                    x_tilda_i = self.G(y_hat_i)     # trainable
                    y_tilda_j = self.F(x_hat_j)     # trainable

                    # Generator loss
                    fake_clean = self.DY(y_hat_i)      # untrainable
                    fake_noisy = self.DX(x_hat_j)      # untrainable
                    noisy_gloss = -torch.mean(fake_noisy)
                    clean_gloss = -torch.mean(fake_clean)
                    gloss = noisy_gloss + clean_gloss
                    # Cycle loss
                    cycle_loss = torch.mean(torch.abs(noisy - x_tilda_i)) + torch.mean(torch.abs(clean - y_tilda_j))
                    # Bypass loss
                    bypass_loss = torch.mean(torch.abs(y_hat_i - y_bar_i)) + torch.mean(torch.abs(clean - y_hat_j))
                    # Nested loss
                    nested_loss = cycle_loss + bypass_loss
                    # Total loss
                    loss_G_total = gloss + self.l * nested_loss
                    loss_G_total.backward()
                    self.optimizer_G.step()
                ###########################
                # (2) Update Disc network #
                ###########################
                self.DX.zero_grad()
                self.DY.zero_grad()
                for i in range(self.args.ISCL_n_disc):
                    y_hat_i = self.F(noisy)         # untrainable
                    y_bar_i = noisy - self.H(noisy) # untrainable
                    x_hat_j = self.G(clean)         # untrainable
                    x_bar_j = clean + self.H(noisy) # untrainable

                    real_noisy = self.DX(noisy)     # trainable
                    real_clean = self.DY(clean)     # trainable
                    fake_noisy = self.DX(x_hat_j.detach())   # trainable
                    fake_clean = self.DY(y_hat_i.detach())   # trainable
                    fake_noisy2 = self.DX(x_bar_j.detach())  # trainable
                    fake_clean2 = self.DY(y_bar_i.detach())  # trainable

                    # Discriminator loss
                    noisy_dloss = torch.mean(F.relu(1.0 - real_noisy)) + torch.mean(F.relu(fake_noisy))
                    clean_dloss = torch.mean(F.relu(1.0 - real_clean)) + torch.mean(F.relu(fake_clean))
                    # Boosting loss
                    bst_loss = torch.mean(F.relu(fake_noisy2)) + torch.mean(F.relu(fake_clean2))
                    # Total loss
                    loss_D_total = noisy_dloss + clean_dloss + bst_loss
                    loss_D_total.backward()
                    self.optimizer_D.step()
                ################################
                # (3) Update Extractor network #
                ################################
                self.H.zero_grad()
                for i in range(self.args.ISCL_n_extractor):
                    n_hat_i = self.H(noisy)         # trainable
                    n_bar_i = noisy - self.F(noisy) # untrainable
                    x_hat_j = self.G(clean)         # untrainable
                    n_tilda_j = self.H(x_hat_j)     # trainable
                    # Pseudo noise loss
                    pseudo_loss = torch.mean(torch.abs(n_hat_i - n_bar_i))
                    noise_consistency = torch.mean(torch.abs(x_hat_j - clean - n_tilda_j))
                    loss_H_total = pseudo_loss + noise_consistency
                    loss_H_total.backward()
                    self.optimizer_H.step()
                ########################
                #     compute loss     #
                ########################
                loss_list[0] += loss_D_total.item()
                loss_list[1] += loss_G_total.item()
                loss_list[2] += loss_H_total.item()

                if (step + 1) % self.args.ISCL_log_step == 0:
                    print("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] H_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_H_total={:.5f}"
                          .format(epoch + 1,
                                  self.args.ISCL_epochs,
                                  step + 1,
                                  self.len_data_loader,
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_H.param_groups[0]['lr'],
                                  loss_list[0] / self.args.ISCL_log_step,
                                  loss_list[1] / self.args.ISCL_log_step,
                                  loss_list[2] / self.args.ISCL_log_step))
                    loss_list = [0.0, 0.0, 0.0]

            if self.args.ISCL_apply_scheduler:
                self.scheduler_D.step()
                self.scheduler_G.step()
                self.scheduler_H.step()

            mean_ssim, mean_psnr = self.validation(self.val_loader, epoch)
            print("Avg SSIM = {}, Avg PSNR = {}\n".format(mean_ssim, mean_psnr))

            if epoch == 0 or max_ssim < mean_ssim:
                max_ssim = mean_ssim
                weights_file_name = 'best_ssim_network_parameter.pth'
                weights_file = os.path.join(self.args.snap_path, weights_file_name)
                torch.save({
                    'epoch': epoch,
                    'G': self.G.state_dict(),
                    'F': self.F.state_dict(),
                    'H': self.H.state_dict(),
                    'DX': self.DX.state_dict(),
                    'DY': self.DY.state_dict(),
                }, weights_file)
                print('save weights of epoch %d' % (epoch + 1) + '\n')
            print("\n")

    def validation(self, val_loader, epoch):

        self.F.eval()
        self.H.eval()
        # random choose one for visualization and save it
        idx = np.random.choice(np.arange(len(val_loader)))
        with torch.no_grad():
            print('Validation:')
            ssim_eval, psnr_eval = 0, 0
            for i, data in enumerate(val_loader):
                noisy, clean = data[0].float().cuda(), data[1].float().cuda()
                y_hat = self.F(noisy)
                y_bar = noisy-self.H(noisy)
                pred = self.gamma * y_hat + (1 - self.gamma) * y_bar
                ssim_eval += self.ssim(pred, clean).item()
                psnr_eval += utils.calc_psnr_for_mri_image(pred, clean).item()

                if i == idx:
                    image_A = noisy[0].permute(1, 2, 0).cpu().numpy()
                    image_B = clean[0].permute(1, 2, 0).cpu().numpy()
                    image_out = pred[0].permute(1, 2, 0).cpu().numpy()
                    visualize.display_images(
                        images = [image_B, image_A, image_A-image_B, image_out-image_B, image_A-image_out, image_out],
                        titles = ['motion free', 'motion simulation', 'error map between 1 and 2', 'error map between 1 and 6', 'error map between 2 and 6', 'after correction'],
                        dir_save = os.path.join(self.args.snap_path, 'visualize', 'epoch={}.jpg'.format(epoch+1)))
            mean_ssim = ssim_eval / len(val_loader)
            mean_psnr = psnr_eval / len(val_loader)
        return mean_ssim, mean_psnr

    def eval(self, test_set):

        self.load_weight()
        self.F.eval()
        self.H.eval()
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        with torch.no_grad():
            for i in range(len(test_set)):
                noisy = torch.tensor(np.array([test_set[i]])).float().cuda()
                y_hat = self.F(noisy)
                y_bar = noisy-self.H(noisy)
                pred = self.gamma * y_hat + (1 - self.gamma) * y_bar
                pred = pred[0].permute(1, 2, 0).cpu().numpy()
                np.savez(os.path.join(self.args.result_path, test_set.filename[i]), pred=pred)