import os

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

from opt.option import args
# from data.LQGT_dataset import LQGTDataset
# from util.utils import RandCrop, RandHorizontalFlip, RandRotate, ToTensor, VGG19PerceptualLoss
from util.utils import init_random_seed, calc_psnr_for_mri_image
from model import encoder, decoder, discriminator
import pytorch_ssim_l1_org
from tqdm import tqdm
from datasets import get_dataloader, get_eval_dataloader
import scipy.io


# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()


if args.manual_seed is not None:
    init_random_seed(args.manual_seed)
    
args_file = open(os.path.join(args.snap_path, 'args.txt'), 'w')
for k, v in vars(args).items():
    args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')
args_file.close()

"""
# make directory for saving weights
if not os.path.exists(args.snap_path):
    os.mkdir(args.snap_path)

# load training dataset
train_dataset = LQGTDataset(
    db_path=args.dir_data,
    transform=transforms.Compose([RandCrop(args.patch_size, args.scale), RandHorizontalFlip(), RandRotate(), ToTensor()])
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True
)
"""

# Load datasets
src_training_loader, tgt_training_loader, validation_loader = get_dataloader(args.dir_source_data, args.dir_target_data, args.batch_size)
len_data_loader = min(len(src_training_loader), len(tgt_training_loader))

# define model (generator)
model_Downsample = decoder.Resnet_DS(num_in_ch=args.output_channels, num_out_ch=args.input_channels, num_feat=args.n_hidden_feats).cuda()
model_Upsample = decoder.Resnet_US(num_in_ch=args.input_channels, num_out_ch=args.output_channels, num_feat=args.n_hidden_feats).cuda()

# define model (discriminator)
model_Disc_img_LR = discriminator.DiscriminatorVGG(in_ch=args.input_channels, image_size=args.patch_size).cuda()
model_Disc_img_HR = discriminator.DiscriminatorVGG(in_ch=args.output_channels, image_size=args.scale*args.patch_size).cuda()
# model_Disc_feat = discriminator.UNetDiscriminator(num_in_ch=64).cuda()
# model_Disc_img_LR = discriminator.UNetDiscriminator(num_in_ch=3).cuda()
# model_Disc_img_HR = discriminator.UNetDiscriminator(num_in_ch=3).cuda()


# loss
loss_L1 = nn.L1Loss().cuda()
loss_MSE = nn.MSELoss().cuda()
loss_adversarial = nn.BCEWithLogitsLoss().cuda()
# loss_percept = VGG19PerceptualLoss().cuda()
loss_ssim = pytorch_ssim_l1_org.SSIM().cuda()


# optimizer 
params_G = list(model_Downsample.parameters()) + list(model_Upsample.parameters())
optimizer_G = optim.Adam(
    params_G,
    lr=args.lr_G,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)
params_D = list(model_Disc_img_LR.parameters()) + list(model_Disc_img_HR.parameters())
optimizer_D = optim.Adam(
    params_D,
    lr=args.lr_D,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay,
    amsgrad=True
)

# Scheduler
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max = args.epochs, eta_min = 1e-8, last_epoch = -1)
scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max = args.epochs, eta_min = 1e-8, last_epoch = -1)
"""
iter_indices = [args.interval1, args.interval2, args.interval3]
scheduler_G = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_G,
    milestones=iter_indices,
    gamma=0.5
)
scheduler_D = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer_D,
    milestones=iter_indices,
    gamma=0.5
)
"""

# load model weights & optimzer % scheduler
if args.checkpoint:
    checkpoint = torch.load(args.weights)

    model_Downsample.load_state_dict(checkpoint['model_Downsample'])
    model_Upsample.load_state_dict(checkpoint['model_Upsample'])
    model_Disc_img_LR.load_state_dict(checkpoint['model_Disc_img_LR'])
    model_Disc_img_HR.load_state_dict(checkpoint['model_Disc_img_HR'])

    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    scheduler_D.load_state_dict(checkpoint['scheduler_D'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G'])

    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0

log_file = open(os.path.join(args.snap_path, 'log.txt'), 'w')
print('Training starts:')
max_ssim = 0.0
# training
for epoch in range(start_epoch, args.epochs):
    # generator
    model_Downsample.train()
    model_Upsample.train()

    # discriminator
    model_Disc_img_LR.train()
    model_Disc_img_HR.train()
    
    running_loss_D_total = 0.0
    running_loss_G_total = 0.0

    running_loss_rec_src = 0.0
    running_loss_rec_tgt = 0.0
    running_loss_sty_tgt = 0.0
    running_loss_sty_src = 0.0

    
    data_zip = enumerate(zip(src_training_loader, tgt_training_loader))
    for step, ((src_HR), (tgt_LR)) in data_zip:

        ########################
        #       data load      #
        ########################
        X_t = tgt_LR[0].float().cuda(non_blocking=True)
        Y_s = src_HR[0].float().cuda(non_blocking=True)


        # real label and fake label
        batch_size = X_t.size(0)
        real_label = torch.full((batch_size, 1), 1, dtype=X_t.dtype).cuda(non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=X_t.dtype).cuda(non_blocking=True)


        ########################
        # (1) Update D network #
        ########################
        model_Disc_img_LR.zero_grad()
        model_Disc_img_HR.zero_grad()

        for i in range(args.n_disc):
            # generator output (feature domain)
            Y_t = model_Upsample(X_t)
            X_s = model_Downsample(Y_s)

            # 1. SR reconstruction loss (discriminator)
            # generator output (image domain)
            Y_s_s = model_Upsample(X_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_s_s_rec = loss_MSE(output_Disc_Y_s_s, fake_label)
            loss_Disc_Y_s_rec = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_src_rec = (loss_Disc_Y_s_s_rec + loss_Disc_Y_s_rec) / 2
            
            # 2. LR reconstruction loss (discriminator)
            # generator output (image domain)
            X_t_t = model_Downsample(Y_t)
            # output of discriminator (image domain)
            output_Disc_X_t_t = model_Disc_img_LR(X_t_t.detach())
            output_Disc_X_t = model_Disc_img_LR(X_t)
            # discriminator loss (image domain)
            loss_Disc_X_t_t_rec = loss_MSE(output_Disc_X_t_t, fake_label)
            loss_Disc_X_t_rec = loss_MSE(output_Disc_X_t, real_label)
            loss_Disc_tgt_rec = (loss_Disc_X_t_t_rec + loss_Disc_X_t_rec) / 2

            # 3. Target style loss
            # generator output (image domain)
            # output of discriminator (image domain)
            output_Disc_X_s = model_Disc_img_LR(X_s.detach())
            output_Disc_X_t = model_Disc_img_LR(X_t)
            # discriminator loss (image domain)
            loss_Disc_X_s_sty = loss_MSE(output_Disc_X_s, fake_label)
            loss_Disc_X_t_sty = loss_MSE(output_Disc_X_t, real_label)
            loss_Disc_tgt_sty = (loss_Disc_X_s_sty + loss_Disc_X_t_sty) / 2

            # 4. Source style loss
            # generator output (image domain)
            # output of discriminator (image domain)
            output_Disc_Y_t = model_Disc_img_HR(Y_t.detach())
            output_Disc_Y_s = model_Disc_img_HR(Y_s)
            # discriminator loss (image domain)
            loss_Disc_Y_t_sty = loss_MSE(output_Disc_Y_t, fake_label)
            loss_Disc_Y_s_sty = loss_MSE(output_Disc_Y_s, real_label)
            loss_Disc_src_sty = (loss_Disc_Y_t_sty + loss_Disc_Y_s_sty) / 2

            # discriminator weight update
            loss_D_total = loss_Disc_src_rec + loss_Disc_tgt_rec + loss_Disc_tgt_sty + loss_Disc_src_sty
            loss_D_total.backward()
            optimizer_D.step()
        # scheduler_D.step()


        ########################
        # (2) Update G network #
        ########################
        model_Upsample.zero_grad()
        model_Downsample.zero_grad()

        for i in range(args.n_gen):
            # generator output (feature domain)
            Y_t = model_Upsample(X_t)
            X_s = model_Downsample(Y_s)

            # 1. Source reconstruction loss
            # generator output (image domain)
            Y_s_s = model_Upsample(X_s)
            # output of discriminator (image domain)
            output_Disc_Y_s_s = model_Disc_img_HR(Y_s_s)
            # L1 loss
            loss_L1_rec_src = loss_L1(Y_s.detach(), Y_s_s)
            # SSIM loss
            loss_ssim_rec_src = (1-loss_ssim(Y_s.detach(), Y_s_s)**2)
            # adversatial loss
            loss_G_Y_s_s = loss_MSE(output_Disc_Y_s_s, real_label)
            L_rec_G_src = loss_L1_rec_src + args.lambda_ssim*loss_ssim_rec_src + args.lambda_adv*loss_G_Y_s_s

            # 2. SR reconstruction loss
            # generator output (image domain)
            X_t_t = model_Downsample(Y_t)
            # output of discriminator (image domain)
            output_Disc_X_t_t = model_Disc_img_LR(X_t_t)
            # L1 loss
            loss_L1_rec_tgt = loss_L1(X_t.detach(), X_t_t)
            # SSIM loss
            loss_ssim_rec_tgt = (1-loss_ssim(X_t.detach(), X_t_t)**2)
            # adversatial loss
            loss_G_X_t_t = loss_MSE(output_Disc_X_t_t, real_label)
            L_rec_G_tgt = loss_L1_rec_tgt + args.lambda_ssim*loss_ssim_rec_tgt + args.lambda_adv*loss_G_X_t_t

            # 3. Target style loss
            # output of discriminator (img domain)
            output_Disc_X_s = model_Disc_img_LR(X_s)
            # generator loss (feature domain)
            loss_G_X_s = loss_MSE(output_Disc_X_s, real_label)
            L_sty_G_tgt = loss_G_X_s

            # 4. Source style loss
            # output of discriminator (img domain)
            output_Disc_Y_t = model_Disc_img_HR(Y_t)
            # generator loss (feature domain)
            loss_G_Y_t = loss_MSE(output_Disc_Y_t, real_label)
            L_sty_G_src = loss_G_Y_t

            # generator weight update
            loss_G_total = args.lambda_rec*L_rec_G_src + args.lambda_rec*L_rec_G_tgt + args.lambda_sty*L_sty_G_tgt + args.lambda_sty*L_sty_G_src
            loss_G_total.backward()
            optimizer_G.step()
        # scheduler_G.step()


        ########################
        #     compute loss     #
        ########################
        running_loss_D_total += loss_D_total.item()
        running_loss_G_total += loss_G_total.item()

        running_loss_rec_src += L_rec_G_src.item()
        running_loss_rec_tgt += L_rec_G_tgt.item()
        running_loss_sty_tgt += L_sty_G_tgt.item()
        running_loss_sty_src += L_sty_G_src.item()
        
        if ((step + 1) % args.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}] lr [{:.8f}]:"
                      "loss_D_total={:.5f} loss_G_total={:.5f} loss_rec_src={:.5f} loss_sty_src={:.5f} loss_rec_tgt={:.5f} loss_sty_tgt={:.5f}"
                      .format(epoch + 1,
                              args.epochs,
                              step + 1,
                              len_data_loader,
                              optimizer_G.param_groups[0]['lr'],
                              running_loss_D_total/args.log_step,
                              running_loss_G_total/args.log_step,
                              running_loss_rec_src/args.log_step,
                              running_loss_sty_src/args.log_step,
                              running_loss_rec_tgt/args.log_step,
                              running_loss_sty_tgt/args.log_step))
            running_loss_D_total = 0
            running_loss_G_total = 0
            running_loss_rec_src = 0
            running_loss_sty_src = 0
            running_loss_rec_tgt = 0
            running_loss_sty_tgt = 0


    scheduler_D.step()
    scheduler_G.step()
    
    model_Upsample.eval()
    with torch.no_grad():
        print('Validation:')
        ssim_eval = 0
        psnr_eval = 0
        for data in validation_loader:
            X_s, Y_s = data
            X_s, Y_s = X_s.float().cuda(), Y_s.float().cuda()
            X_SR = model_Upsample(X_s)
            ssim_eval += loss_ssim(X_SR, Y_s).item()
            psnr_eval += calc_psnr_for_mri_image(X_SR, Y_s).item()
        mean_ssim = ssim_eval/len(validation_loader)
        mean_psnr = psnr_eval/len(validation_loader)
        print("Avg SSIM = {}, Avg PSNR = {}".format(mean_ssim, mean_psnr))
        log_file.write("Epoch: %d, SSIM: %f, PSNR: %f \n" % (epoch+1, mean_ssim, mean_psnr))

    
    if epoch == 0 or max_ssim<ssim_eval:
        max_ssim = ssim_eval
        weights_file_name = 'best_ssim_network_parameter.pth'
        weights_file = os.path.join(args.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,

            'model_Upsample': model_Upsample.state_dict(),
            'model_Downsample': model_Downsample.state_dict(),
            'model_Disc_img_LR': model_Disc_img_LR.state_dict(),
            'model_Disc_img_HR': model_Disc_img_HR.state_dict(),

            'optimizer_D': optimizer_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),

            'scheduler_D': scheduler_D.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))
        log_file.write("Network saved! \n")
    log_file.write("\n")
log_file.close()        
        

if args.perform_inference:
    "Evaluation(Test)"
    checkpoint = torch.load(args.weights)
    model_Upsample.load_state_dict(checkpoint['model_Upsample'])
    model_Upsample.eval()
    filenames = sorted(os.listdir(os.path.join(args.dir_test, 'Evaluation')))
    for filename in tqdm(filenames):
        if '.mat' in filename:
            eval_loader = get_eval_dataloader(args.dir_test, filename, args.batch_size)
            with torch.no_grad():    
                for i, tgt_LR in enumerate(eval_loader, 0):
                    X_t = tgt_LR[0].float().cuda()
                    X_SR = model_Upsample(X_t)
                    if i==0:
                        SR_img_eval_tensor = X_SR.data
                    else:
                        SR_img_eval_tensor = torch.cat((SR_img_eval_tensor, X_SR.data), 0)
    
            SR_images_test = SR_img_eval_tensor.cpu().numpy()
            "save the .mat files for SR, HR and LR training images"
            scipy.io.savemat(os.path.join(args.results, os.path.splitext(filename)[0]+'_SR_test_image.mat'), mdict = {'SR_test_image' : SR_images_test})
