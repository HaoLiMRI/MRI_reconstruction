import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from math import exp
import torch
import random
import numpy as np
import os, sys


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

############################################################
#  calculate psnr
############################################################
def calc_psnr_for_mri_image(img1, img2):
    ### args:
        # img1: pytorch tensor, shape is [N, C, H, W]
        # img2: pytorch tensor, shape is [N, C, H, W]

    diff = torch.add(img1, -img2)
    mse = torch.pow(diff, 2).mean(2).mean(2)
    return -10 * torch.log10(mse).mean(0).mean(0)

############################################################
#  ssim
############################################################
def _ssim(img1, img2, window, window_size, channel, size_average=True, luminance_weight=1., contrast_weight=1.,
          structure_weight=1.):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.clamp((F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq), 0, 1)
    sigma2_sq = torch.clamp((F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq), 0, 1)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_loss(nn.Module):
    def __init__(self, window_size=11, size_average=True, luminance_weight=1., contrast_weight=1., structure_weight=1.):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.luminance_weight = luminance_weight
        self.contrast_weight = contrast_weight
        self.structure_weight = structure_weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, self.luminance_weight,
                     self.contrast_weight, self.structure_weight)


def ssim(img1, img2, window_size=11, size_average=True, luminance_weight=1, contrast_weight=1, structure_weight=1):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, luminance_weight, contrast_weight,
                 structure_weight)

############################################################
#  calculate gradient penalty
############################################################
def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

############################################################
#  get all filepath and filename under main_path
############################################################
def get_filepath(main_path):
    # read all file path in main_path
    file_path = []
    filename = []
    for root, dirs, files in os.walk(main_path):
        if len(files) != 0:
            for i in files:
                file_path.append(os.path.join(root, i))
                filename.append(i)

    return file_path, filename

##################################################################
#  calculate the ssim and psnr between motion free MRImage and MA
##################################################################
def cal_original_ssim_psnr(val_loader):
    ssim_eval, psnr_eval = 0, 0
    ssim = SSIM_loss().cuda()
    for data in val_loader:
        A, B = data
        input_A, input_B = A.float().cuda(), B.float().cuda()
        ssim_eval += ssim(input_A, input_B).item()
        psnr_eval += calc_psnr_for_mri_image(input_A, input_B).item()
    mean_ssim = ssim_eval / len(val_loader)
    mean_psnr = psnr_eval / len(val_loader)
    return mean_ssim, mean_psnr

##################################################################
#  多端日志打印
##################################################################
class logger(object):
    def __init__(self, name='default.log', stream = sys.stdout):
        self.terminal = stream
        self.log = open(name, 'a+')

    def write(self, message):
        """同时向终端和日志中写入打印信息"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def create_training_log(log_path):

    sys.stdout = logger(log_path, sys.stdout)
    print('Record training logs!')

##################################################################
#  image normalize
##################################################################
def normalize(img):

    vmin = np.min(img)
    vmax = np.max(img)
    volume = (img - vmin) / (vmax - vmin)
    volume = volume.astype("float32")

    return volume

##################################################################
# random noise z
##################################################################
def get_random_sample(shape, method = 'normal'):
    if method == 'uniform':
        sample_z = np.random.uniform(-1, 1, size = shape).astype(np.float32)
    elif method == 'random':
        sample_z = 2.0 * np.random.random(size = shape) - 1.0
    else:
        sample_z = np.random.normal(size = shape)
        sample_z = (sample_z - np.min(sample_z)) / (np.max(sample_z) - np.min(sample_z))
        sample_z = 2.0 * sample_z - 1.0
    return sample_z
