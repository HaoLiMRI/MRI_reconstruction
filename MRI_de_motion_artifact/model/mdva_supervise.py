
import torch.nn as nn
import torch
import torch.optim as optim
from model import generator, discriminator
from tools import utils, visualize
import numpy as np
import os


class mdvaGAN(object):

    def __init__(self, args, data_loader):

        # network
        self.gen_A2B = generator.Decoder_Id_RCAN(num_in_ch=args.input_channels, num_out_ch=args.output_channels, num_feat=args.n_hidden_feats).cuda()
        self.gen_B2A = generator.Decoder_Id_RCAN(num_in_ch=args.output_channels, num_out_ch=args.input_channels, num_feat=args.n_hidden_feats).cuda()
        self.disc_A = discriminator.DiscriminatorVGG(in_ch=args.input_channels).cuda()
        self.disc_B = discriminator.DiscriminatorVGG(in_ch=args.output_channels).cuda()

        # Loss criterion
        self.loss_L1 = nn.L1Loss().cuda()
        self.loss_MSE = nn.MSELoss().cuda()
        self.loss_adversarial = nn.BCEWithLogitsLoss().cuda()
        self.loss_ssim = utils.SSIM_loss().cuda()
        self.gp = utils.calc_gradient_penalty

        # dataset loder
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]
        self.len_data_loader = len(self.train_loader)

        # optimizer
        params_G = list(self.gen_A2B.parameters()) + list(self.gen_B2A.parameters())
        self.optimizer_G = optim.Adam(params_G, lr=args.lr_G, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, amsgrad=True)
        params_D = list(self.disc_A.parameters()) + list(self.disc_B.parameters())
        self.optimizer_D = optim.Adam(params_D, lr=args.lr_D, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, amsgrad=True)

        # Scheduler
        # self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=args.epochs, eta_min=1e-8, last_epoch=-1)
        # self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=args.epochs, eta_min=1e-8, last_epoch=-1)
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=10, gamma=0.5)

        # load weight
        if args.load_weight:
            self.load_weight()
        else:
            self.start_epoch = 0
        self.args = args

    def load_weight(self):

        path = os.path.join(self.args.snap_path, 'best_ssim_network_parameter.pth')
        checkpoint = torch.load(path)
        self.gen_A2B.load_state_dict(checkpoint['gen_A2B'])
        self.gen_B2A.load_state_dict(checkpoint['gen_B2A'])
        self.disc_A.load_state_dict(checkpoint['disc_A'])
        self.disc_B.load_state_dict(checkpoint['disc_B'])
        self.start_epoch = checkpoint['epoch']

        #    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        #    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        #    scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        #    scheduler_G.load_state_dict(checkpoint['scheduler_G'])

    def train(self):

        utils.create_training_log(os.path.join(self.args.snap_path, 'log.txt'))
        max_ssim = 0.0
        # for store visualization validation result during training
        if not os.path.exists(os.path.join(self.args.snap_path, 'visualize')):
            os.makedirs(os.path.join(self.args.snap_path, 'visualize'))

        for epoch in range(self.start_epoch, self.args.epochs):

            # running [loss_D_total, loss_G_total, gp_A, gp_B] record
            loss_list = [0.0, 0.0, 0.0, 0.0]
            # generator
            self.gen_A2B.train()
            self.gen_B2A.train()
            # discriminator
            self.disc_A.train()
            self.disc_B.train()

            for step, train_data in enumerate(self.train_loader):

                ########################
                #       data load      #
                ########################
                A, B = train_data
                input_A = A.float().cuda(non_blocking=True)
                input_B = B.float().cuda(non_blocking=True)

                real_label = torch.full((self.args.batch_size, 1), 1, dtype=input_A.dtype).cuda(non_blocking=True)
                fake_label = torch.full((self.args.batch_size, 1), 0, dtype=input_A.dtype).cuda(non_blocking=True)
                ###########################
                # (1) Update Disc network #
                ###########################
                self.disc_A.zero_grad()
                self.disc_B.zero_grad()
                for i in range(self.args.n_disc):
                    # generator output (feature domain)
                    gen_B = input_A - self.gen_A2B(input_A)
                    cyclic_A = self.gen_B2A(gen_B)

                    # output of discriminator (image domain)
                    disc_input_A = self.disc_A(input_A)
                    disc_cyclic_A = self.disc_A(cyclic_A.detach())
                    disc_input_B = self.disc_B(input_B)
                    disc_gen_B = self.disc_B(gen_B.detach())

                    # reconstruction loss
                    loss_disc_A_cycle = (self.loss_MSE(disc_cyclic_A, fake_label) + self.loss_MSE(disc_input_A, real_label)) / 2
                    # style loss
                    loss_disc_B_sty = (self.loss_MSE(disc_gen_B, fake_label) + self.loss_MSE(disc_input_B, real_label)) / 2

                    # discriminator weight update
                    loss_D_total = loss_disc_A_cycle + loss_disc_B_sty
                    if self.args.GPloss:
                        loss_disc_A_gp = self.gp(self.disc_A, input_A.data, cyclic_A.data, self.args.batch_size) * self.args.lambda_gp
                        loss_disc_B_gp = self.gp(self.disc_B, input_B.data, gen_B.data, self.args.batch_size) * self.args.lambda_gp
                        loss_D_total += loss_disc_A_gp + loss_disc_B_gp
                    loss_D_total.backward()
                    self.optimizer_D.step()
                ##########################
                # (2) Update Gen network #
                ###########################
                self.gen_A2B.zero_grad()
                self.gen_B2A.zero_grad()
                for i in range(self.args.n_gen):
                    # generator output (feature domain)
                    gen_B = input_A - self.gen_A2B(input_A)
                    cyclic_A = self.gen_B2A(gen_B)

                    # output of discriminator (image domain)
                    disc_cyclic_A = self.disc_A(cyclic_A)
                    disc_gen_B = self.disc_B(gen_B)

                    # L1 loss
                    loss_L1_A = self.loss_L1(input_A, cyclic_A)
                    loss_L1_B = self.loss_L1(input_B, gen_B)
                    # SSIM loss
                    loss_ssim_A = (1 - self.loss_ssim(input_A, cyclic_A) ** 2)
                    loss_ssim_B = (1 - self.loss_ssim(input_B, gen_B) ** 2)
                    # adversatial loss
                    loss_adv_A = self.loss_MSE(disc_cyclic_A, real_label)
                    # reconstruction loss
                    loss_gen_A_cycle = loss_L1_A + self.args.lambda_ssim * loss_ssim_A + self.args.lambda_adv * loss_adv_A
                    loss_gen_B_cycle = loss_L1_B + self.args.lambda_ssim * loss_ssim_B
                    # generator style loss (feature domain)
                    loss_gen_B_sty = self.loss_MSE(disc_gen_B, real_label)

                    # generator weight update
                    loss_G_total = self.args.lambda_cyc * loss_gen_A_cycle +  self.args.lambda_sty * loss_gen_B_sty + \
                                   self.args.lambda_cyc * loss_gen_B_cycle
                    loss_G_total.backward()
                    self.optimizer_G.step()
                ########################
                #     compute loss     #
                ########################
                loss_list[0] += loss_D_total.item()
                loss_list[1] += loss_G_total.item()

                if self.args.GPloss:
                    loss_list[2] += loss_disc_A_gp.item()
                    loss_list[3] += loss_disc_B_gp.item()

                if (step + 1) % self.args.log_step == 0:
                    print("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]:"
                          "loss_D_total={:.5f}  loss_G_total={:.5f}  gp_A={:.5f}  gp_B={:.5f}"
                          .format(epoch + 1,
                                  self.args.epochs,
                                  step + 1,
                                  self.len_data_loader,
                                  self.optimizer_G.param_groups[0]['lr'],
                                  loss_list[0] / self.args.log_step,
                                  loss_list[1] / self.args.log_step,
                                  loss_list[2] / self.args.log_step,
                                  loss_list[3] / self.args.log_step))
                    loss_list = [0.0, 0.0, 0.0, 0.0]

            if self.args.apply_scheduler:
                self.scheduler_D.step()
                self.scheduler_G.step()

            mean_ssim, mean_psnr = self.validation(self.val_loader, epoch)
            print("Avg SSIM = {}, Avg PSNR = {}\n".format(mean_ssim, mean_psnr))

            if epoch == 0 or max_ssim < mean_ssim:
                max_ssim = mean_ssim
                weights_file_name = 'best_ssim_network_parameter.pth'
                weights_file = os.path.join(self.args.snap_path, weights_file_name)
                torch.save({
                    'epoch': epoch,
                    'gen_A2B': self.gen_A2B.state_dict(),
                    'gen_B2A': self.gen_B2A.state_dict(),
                    'disc_A': self.disc_A.state_dict(),
                    'disc_B': self.disc_B.state_dict(),
                    #            'optimizer_D': optimizer_D.state_dict(),
                    #            'optimizer_G': optimizer_G.state_dict(),
                    #            'scheduler_D': scheduler_D.state_dict(),
                    #            'scheduler_G': scheduler_G.state_dict(),
                    }, weights_file)
                print('save weights of epoch %d' % (epoch + 1) + '\n')
            print("\n")

    def validation(self, val_loader, epoch):

        self.gen_A2B.eval()
        # random choose one for visualization and save it
        idx = np.random.choice(np.arange(len(val_loader)))
        with torch.no_grad():
            print('Validation:')
            ssim_eval, psnr_eval = 0, 0
            for i, data in enumerate(val_loader):
                A, B = data
                input_A, input_B = A.float().cuda(), B.float().cuda()
                gen_B = input_A - self.gen_A2B(input_A)
                ssim_eval += self.loss_ssim(gen_B, input_B).item()
                psnr_eval += utils.calc_psnr_for_mri_image(gen_B, input_B).item()

                if i == idx:
                    image_A = input_A[0].permute(1, 2, 0).cpu().numpy()
                    image_B = input_B[0].permute(1, 2, 0).cpu().numpy()
                    image_out = gen_B[0].permute(1, 2, 0).cpu().numpy()
                    visualize.display_images(
                        images = [image_B, image_A, image_A-image_B, image_out-image_B, image_A-image_out, image_out],
                        titles = ['motion free', 'motion simulation', 'error map between 1 and 2', 'error map between 1 and 6', 'error map between 2 and 6', 'after correction'],
                        dir_save = os.path.join(self.args.snap_path, 'visualize', 'epoch={}.jpg'.format(epoch+1)),
                    )
            mean_ssim = ssim_eval / len(val_loader)
            mean_psnr = psnr_eval / len(val_loader)

        return mean_ssim, mean_psnr

    def eval(self, test_set):

        self.load_weight()
        self.gen_A2B.eval()
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        with torch.no_grad():
            for i in range(len(test_set)):
                input_A = torch.tensor(np.array([test_set[i]])).float().cuda()
                gen_B = input_A - self.gen_A2B(input_A)
                gen_B = gen_B[0].permute(1, 2, 0).cpu().numpy()
                np.savez(os.path.join(self.args.result_path, test_set.filename[i]), pred=gen_B)

