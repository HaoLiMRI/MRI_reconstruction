import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
import os



class ResBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResBlock, self).__init__()
        conv_block = [nn.Conv2d(num_feat, num_feat, (3, 3), padding=(1, 1)),
                      # BatchInstanceNorm2d(num_feat),
                      nn.BatchNorm2d(num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                      nn.Conv2d(num_feat, num_feat, (3, 3), padding=(1, 1)),
                      # BatchInstanceNorm2d(num_feat)
                      nn.BatchNorm2d(num_feat)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Extractor(nn.Module):
    def __init__(self, num_in_ch=1, num_feat=16, num_out_ch=1, n_layers=20):
        super(Extractor, self).__init__()

        model = [nn.Conv2d(num_in_ch, num_feat, kernel_size=(3,3), padding=(1, 1)),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(0, n_layers - 2):
            model += [nn.Conv2d(num_feat, num_feat, kernel_size=(3,3), padding=(1, 1)),
                      # BatchInstanceNorm2d(num_feat),
                      nn.BatchNorm2d(num_feat),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        model += [nn.Conv2d(num_feat, num_out_ch, kernel_size=(3,3), padding=(1, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, num_feat=16):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # BatchInstanceNorm2d(2 * num_feat),
            nn.BatchNorm2d(2 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # BatchInstanceNorm2d(4 * num_feat),
            nn.BatchNorm2d(4 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(4 * num_feat, 8 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            # BatchInstanceNorm2d(8 * num_feat),
            nn.BatchNorm2d(8 * num_feat),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(8 * num_feat, 16 * num_feat, kernel_size=(4, 4), stride=(1, 1)),
            # BatchInstanceNorm2d(16 * num_feat),
            nn.BatchNorm2d(16 * num_feat),
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

        model = [nn.Conv2d(num_in_ch, num_feat, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                 # BatchInstanceNorm2d(num_feat),
                 nn.BatchNorm2d(num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True),

                 nn.Conv2d(num_feat, 2 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                 # BatchInstanceNorm2d(2 * num_feat),
                 nn.BatchNorm2d(2 * num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True),

                 nn.Conv2d(2 * num_feat, 4 * num_feat, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                 # BatchInstanceNorm2d(4 * num_feat),
                 nn.BatchNorm2d(4 * num_feat),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for i in range(nres):
            model.append(ResBlock(4 * num_feat))

        model += [nn.ConvTranspose2d(4 * num_feat, 2 * num_feat, kernel_size=(2, 2), stride=(2, 2)),
                  # BatchInstanceNorm2d(2 * num_feat),
                  nn.BatchNorm2d(2 * num_feat),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                  nn.ConvTranspose2d(2 * num_feat, num_feat, kernel_size=(2, 2), stride=(2, 2)),
                  # BatchInstanceNorm2d(num_feat),
                  nn.BatchNorm2d(num_feat),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                  nn.Conv2d(num_feat, num_out_ch, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ISCL(object):

    def __init__(self, config):

        # Initialization
        self.gamma = 0.5
        self.l = 30

        # Models
        self.G = Generator(num_in_ch=1, num_out_ch=1, num_feat=32).to(config.device) # Clean image -> Noisy image
        self.F = Generator(num_in_ch=1, num_out_ch=1, num_feat=32).to(config.device) # Noisy image -> Clean image
        self.H = Extractor(num_in_ch=1, num_out_ch=1, num_feat=32).to(config.device) # Noisy image -> Noise map
        self.DX = Discriminator(in_ch=1, num_feat=32).to(config.device) # Noisy image discriminator
        self.DY = Discriminator(in_ch=1, num_feat=32).to(config.device) # Clean image discriminator

        # Optimizers
        params_G = list(self.G.parameters()) + list(self.F.parameters())
        params_D = list(self.DX.parameters()) + list(self.DY.parameters())
        params_H = list(self.H.parameters())
        # self.optimizer_G = RAdam(params_G, lr=config.lr_G, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        # self.optimizer_D = RAdam(params_D, lr=config.lr_D, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        # self.optimizer_H = RAdam(params_H, lr=config.lr_H, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.optimizer_G = optim.Adam(params_G, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_D = optim.Adam(params_D, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        self.optimizer_H = optim.Adam(params_H, lr=1e-4, betas=(0.9, 0.99), amsgrad=True)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)
        self.scheduler_H = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_H, T_max=config.epochs, eta_min=1e-6, last_epoch=-1)

        self.config = config
        self.start_epoch = 0
        # load weight
        if config.load_weight:
            self.load_weight()


    def load_weight(self):

        path = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path, map_location=torch.device(self.config.device))
        self.G.load_state_dict(checkpoint['G'])
        self.F.load_state_dict(checkpoint['F'])
        self.H.load_state_dict(checkpoint['H'])
        self.DX.load_state_dict(checkpoint['DX'])
        self.DY.load_state_dict(checkpoint['DY'])
        self.start_epoch = checkpoint['epoch']


    def train(self, data_loader):

        MA_loader = data_loader[0]             # MA: motion artifact
        MF_loader = data_loader[1]             # MF: motion free
        val_loader = data_loader[2]

        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        max_ssim = 0.0

        for epoch in range(self.start_epoch, self.config.epochs):

            loss_D, loss_G, loss_H = 0., 0., 0.
            # generator
            self.G.train()
            self.F.train()
            # discriminator
            self.DX.train()
            self.DY.train()
            # extractor
            self.H.train()

            for step, ((A), (B)) in enumerate(zip(MA_loader, MF_loader)):
                ########################
                #       data load      #
                ########################
                noisy = A['ma_img'].to(self.config.device)
                clean = B['gt_img'].to(self.config.device)

                ###########################
                # (1) Update Gen network #
                ###########################
                self.optimizer_G.zero_grad()

                y_hat_i = self.F(noisy)     # trainable
                x_hat_j = self.G(clean)     # trainable
                y_bar_i = noisy - self.H(noisy)    # untrainable
                x_bar_j = clean + self.H(noisy)    # untrainable
                y_hat_j = self.F(x_bar_j)       # trainable
                x_tilda_i = self.G(y_hat_i)     # trainable
                y_tilda_j = self.F(x_hat_j)     # trainable

                # Generator loss
                fake_clean = self.DY(y_hat_i)  # untrainable
                fake_noisy = self.DX(x_hat_j)  # untrainable
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

                loss_G += loss_G_total.item()
                ###########################
                # (2) Update Disc network #
                ###########################
                self.optimizer_D.zero_grad()

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

                loss_D += loss_D_total.item()
                ################################
                # (3) Update Extractor network #
                ################################
                self.H.zero_grad()

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

                loss_H += loss_H_total.item()
                ########################
                #     record loss      #
                ########################
                if (step + 1) % self.config.log_step == 0:
                    log_file.write("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] H_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_H_total={:.5f}\n"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_H.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step,
                                  loss_H / self.config.log_step))
                    print("Epoch [{}/{}] Step [{}/{}] G_lr [{:.8f}] H_lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  loss_H_total={:.5f}"
                          .format(epoch + 1,
                                  self.config.epochs,
                                  step + 1,
                                  len(MA_loader),
                                  self.optimizer_G.param_groups[0]['lr'],
                                  self.optimizer_H.param_groups[0]['lr'],
                                  loss_D / self.config.log_step,
                                  loss_G / self.config.log_step,
                                  loss_H / self.config.log_step))
                    loss_D, loss_G, loss_H = 0., 0., 0.

            self.scheduler_D.step()
            self.scheduler_G.step()
            self.scheduler_H.step()

            ########################
            #     validation       #
            ########################
            self.F.eval()
            self.H.eval()
            with torch.no_grad():
                print('Validation:')
                ssim_eval, psnr_eval = 0, 0
                for i, batch in enumerate(val_loader):
                    noisy = batch['ma_img'].to(self.config.device)
                    clean = batch['gt_img'].to(self.config.device)
                    y_hat = self.F(noisy)
                    y_bar = noisy - self.H(noisy)
                    pred = self.gamma * y_hat + (1 - self.gamma) * y_bar
                    ssim_eval += utils.ssim(pred, clean).item()
                    psnr_eval += utils.psnr(pred, clean).item()
                ssim_eval = ssim_eval / len(val_loader)
                psnr_eval = psnr_eval / len(val_loader)
                log_file.write("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))
                print("Avg SSIM = {}, Avg PSNR = {}\n".format(ssim_eval, psnr_eval))

                if epoch == 0 or max_ssim < ssim_eval:
                    max_ssim = ssim_eval
                    weights_file = os.path.join(self.config.snap_path, 'best_ssim_network_parameter.pth')
                    torch.save({
                        'epoch': epoch,
                        'G': self.G.state_dict(),
                        'F': self.F.state_dict(),
                        'H': self.H.state_dict(),
                        'DX': self.DX.state_dict(),
                        'DY': self.DY.state_dict(),
                    }, weights_file)
                    log_file.write('save weights of epoch %d' % (epoch + 1) + '\n')
                    print('save weights of epoch %d' % (epoch + 1) + '\n')

            print("\n")
            log_file.write('\n')
        log_file.close()


    def eval(self, test_loader):

        self.load_weight()
        self.F.eval()
        self.H.eval()
        if not os.path.exists(self.config.inference_path):
            os.makedirs(self.config.inference_path)
        log_file = open(os.path.join(self.config.snap_path, 'log.txt'), "a+")
        with torch.no_grad():
            ma_ssim, ma_psnr = 0., 0.
            eval_ssim, eval_psnr = 0, 0
            for i, batch in enumerate(test_loader):

                noisy = batch['ma_img'].to(self.config.device)
                clean = batch['gt_img'].to(self.config.device)

                y_hat = self.F(noisy)
                y_bar = noisy-self.H(noisy)
                pred = self.gamma * y_hat + (1 - self.gamma) * y_bar

                ma_ssim += utils.ssim(noisy, clean).item()
                ma_psnr += utils.psnr(noisy, clean).item()
                eval_ssim += utils.ssim(pred, clean).item()
                eval_psnr += utils.psnr(pred, clean).item()

                pred = pred[0, 0].cpu().numpy()
                np.savez(os.path.join(self.config.inference_path, batch['filename'][0]), pred=pred)

        num = len(test_loader)

        print('\nBefore Reduction: ssim=%.4f and psnr=%.4f' % (ma_ssim / num, ma_psnr / num))
        log_file.write('\nBefore Reduction: ssim=%.4f and psnr=%.4f\n' % (ma_ssim / num, ma_psnr / num))

        print('\nEvaluation: ssim=%.4f and psnr=%.4f' % (eval_ssim / num, eval_psnr / num))
        log_file.write('\nEvaluation: ssim=%.4f and psnr=%.4f\n' % (eval_ssim / num, eval_psnr / num))

        log_file.close()