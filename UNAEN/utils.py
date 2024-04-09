import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from math import exp
import torch
import random
import numpy as np
import os


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
    else:
        torch.manual_seed(seed)


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
def psnr(img1, img2):
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
    alpha = alpha.to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(real_data.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
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
#  image normalize
##################################################################
def normalize(img, max_value=None, min_value=None, clip=True):

    if min_value is None:
        min_value = np.min(img)
    if max_value is None:
        max_value = np.max(img)
    if max_value == min_value:
        return img

    img = (img - min_value) / (max_value - min_value)
    if clip:
        img = np.clip(img, 0, 1)
    return img.astype("float32")

##################################################################
#
##################################################################
def image_concatenation(sub_images, shape, crop_size, stride):
    # overlap region line
    overlap_line = crop_size - stride
    # fully image
    image = np.zeros(shape)
    for i in range((shape[0] - crop_size) // stride + 1):
        # concatenated patch of one line
        img = np.zeros([crop_size, shape[1]])
        for j in range((shape[1] - crop_size) // stride + 1):
            if j != 0:
                sub_images[0][:, :overlap_line] = np.mean([img[:, j * stride:j * stride + overlap_line], sub_images[0][:, :overlap_line]], axis=0)
            img[:, j * stride:j * stride + crop_size] = sub_images[0]
            del sub_images[0]
        if i != 0:
            img[:overlap_line] = np.mean([image[i * stride:i * stride + overlap_line], img[:overlap_line]], axis=0)
        image[i * stride:i * stride + crop_size] = img
    return image


def crop_img(gt_img, ma_img, crop_size=128, stride=None):

    if stride is None:
        stride = int(crop_size * 3/4)

    gt_crop = []
    ma_crop = []

    for i in range((gt_img.shape[0] - crop_size) // stride + 1):
        for j in range((gt_img.shape[1] - crop_size) // stride + 1):
            gt_crop.append(gt_img[i * stride:i * stride + crop_size, j * stride:j * stride + crop_size])
            ma_crop.append(ma_img[i * stride:i * stride + crop_size, j * stride:j * stride + crop_size])

    return np.array(gt_crop).astype(np.float32), np.array(ma_crop).astype(np.float32)

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
