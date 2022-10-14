import torch
#import torchvision

import math
#import cv2
import numpy as np
#from scipy.ndimage import rotate
import pytorch_ssim_l1_org
import random
from torchvision.models import vgg19
#from receptive_cal import *
import torch.nn as nn
from pytorch_wavelets import DWTForward

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    
class L1_Charbonnier_Loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_Loss,self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


SSIM_function = pytorch_ssim_l1_org.SSIM()
loss_function_Charbonnier = L1_Charbonnier_Loss()
    

def get_rec_loss_function(output, label):
    loss = loss_function_Charbonnier(output, label) + 0.5 * (1 - SSIM_function(output, label)**2)
    return loss


def calc_psnr_for_mri_image(img1, img2):
    ### args:
        # img1: pytorch tensor, shape is [N, C, H, W]
        # img2: pytorch tensor, shape is [N, C, H, W]

    diff = torch.add(img1, -img2)
    mse = torch.pow(diff, 2).mean(2).mean(2)
    return -10 * torch.log10(mse).mean(0).mean(0)




"""
class RandCrop(object):
    def __init__(self, crop_size, scale):
        # if output size is tuple -> (height, width)
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        
        self.scale = scale

    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        h, w, c = img_LQ.shape
        new_h, new_w = self.crop_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img_LQ_crop = img_LQ[top: top+new_h, left: left+new_w, :]

        h, w, c = img_GT.shape
        top = np.random.randint(0, h - self.scale*new_h)
        left = np.random.randint(0, w - self.scale*new_w)
        img_GT_crop = img_GT[top: top + self.scale*new_h, left: left + self.scale*new_w, :]

        sample = {'img_LQ': img_LQ_crop, 'img_GT': img_GT_crop}
        return sample


class RandRotate(object):
    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        prob_rotate = np.random.random()
        if prob_rotate < 0.25:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        elif prob_rotate < 0.5:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        elif prob_rotate < 0.75:
            img_LQ = rotate(img_LQ, 90).copy()
            img_GT = rotate(img_GT, 90).copy()
        
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample


class RandHorizontalFlip(object):
    def __call__(self, sample):
        # img_LQ: H x W x C (numpy array)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        prob_lr = np.random.random()
        if prob_lr < 0.5:
            img_LQ = np.fliplr(img_LQ).copy()
            img_GT = np.fliplr(img_GT).copy()
        
        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # img_LQ : H x W x C (numpy array) -> C x H x W (torch tensor)
        img_LQ, img_GT = sample['img_LQ'], sample['img_GT']

        img_LQ = img_LQ.transpose((2, 0, 1))
        img_GT = img_GT.transpose((2, 0, 1))

        img_LQ = torch.from_numpy(img_LQ)
        img_GT = torch.from_numpy(img_GT)

        sample = {'img_LQ': img_LQ, 'img_GT': img_GT}
        return sample
"""
     

class VGG19PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=36):
        super(VGG19PerceptualLoss, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer]).eval().cuda()
        # Freeze parameters
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
    
    def forward(self, source, target):
        vgg_loss = self.mse_loss(self.features(source), self.features(target))

        return vgg_loss


def domain_distance_map_handler(fake_img, D_out, convnet=[[5, 1, 2], [5, 1, 2], [5, 1, 2], [5, 1, 2]]):
    ddm_shape = (fake_img.shape[0], 1, fake_img.shape[2], fake_img.shape[3])
    ddm = torch.zeros(ddm_shape)
    currentLayer_h, currentLayer_w = receptive_cal(ddm.shape[2], convnet), receptive_cal(ddm.shape[3], convnet)
    ddm = getWeights(D_out, ddm, currentLayer_h, currentLayer_w)
    return ddm



def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    # pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3]))


def weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start):
    # B, C, H, W = patch.shape
    wm = np.zeros(img.shape)
    for i in range(n_f_h):
        for j in range(n_f_w):
            val = patch[:, :, i, j]
            hf, ht = int(max(0, start + i*jump - rf//2)), int(start + i*jump + rf - rf//2)
            wf, wt = int(max(0, start + j*jump - rf//2)), int(start + j*jump + rf - rf//2)    
            wm[:, :, hf:ht, wf:wt] = wm[:, :, hf:ht, wf:wt] + val
    return wm


def receptive_cal(imsize, convnet=[[5, 1, 2], [5, 1, 2], [5, 1, 2], [5, 1, 2]]):
    convnet = [[4, 1, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1]]
    # layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    currentLayer = [imsize, 1, 1, 0.5]
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        # layerInfos.append(currentLayer)
    return currentLayer


def getWeights(patch, img, currentLayer_h, currentLayer_w):
    n_f_h, jump, rf, start = currentLayer_h[0], currentLayer_h[1], currentLayer_h[2], currentLayer_h[3]
    n_f_w, jump, rf, start = currentLayer_w[0], currentLayer_w[1], currentLayer_w[2], currentLayer_w[3]
    s = weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start)
    count = weights_matrix(np.ones_like(patch), img, n_f_h, n_f_w, jump, rf, start)
    return s / count


def discriminator_loss(reals, fakes, weights=None):
    if not isinstance(reals, list):
        reals = (reals,)
    if not isinstance(fakes, list):
        fakes = (fakes,)
    if weights is None:
        weights = [1.0 / len(fakes)] * len(fakes)
    loss = 0.0
    for real, fake, weight in zip(reals, fakes, weights):
        loss += weight * (-torch.log(real + 1e-8).mean() - torch.log(1 - fake + 1e-8).mean())
    return loss


def generator_loss(labels, weights=None):
    if not isinstance(labels, list):
        labels = (labels,)
    if weights is None:
        weights = [1.0 / len(labels)] * len(labels)
    loss = 0.0
    for label, weight in zip(labels, weights):
        loss += weight * torch.mean(-torch.log(label + 1e-8))
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, use_perceptual_loss=True, w_col=0.01, w_tex=0.005, w_per=1.0):
        super(GeneratorLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.color_filter = self.filter_wavelet_LL
        
        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()
        
        self.perceptual_loss = VGG19PerceptualLoss()
        
        self.use_perceptual_loss = use_perceptual_loss
        self.w_col = w_col
        self.w_tex = w_tex
        self.w_per = w_per
        self.last_tex_loss = 0
        self.last_per_loss = 0
        self.last_col_loss = 0
        self.last_mean_loss = 0

    def forward(self, tex_labels, out_images, target_images, ddm=None):
        # Adversarial Texture Loss
        self.last_tex_loss = generator_loss(tex_labels)
        # Perception Loss
        if out_images.size(1) == 6:
            self.last_per_loss = (self.perceptual_loss(out_images[:,0:3,:,:], target_images[:,0:3,:,:]) + self.perceptual_loss(out_images[:,3:6,:,:], target_images[:,3:6,:,:])) / 2
        else:
            self.last_per_loss = self.perceptual_loss(out_images, target_images)
        # Color Loss
        self.last_col_loss = self.rgb_loss(out_images, target_images, ddm)
        loss = self.w_col * self.last_col_loss + self.w_tex * self.last_tex_loss
        if self.use_perceptual_loss:
            loss += self.w_per * self.last_per_loss
        return loss

    def color_loss(self, x, y):
        return self.pixel_loss(self.color_filter(x), self.color_filter(y))

    def rgb_loss(self, x, y, ddm=None):
        if ddm==None:
            return self.pixel_loss(x, y)
        else:
            return torch.mean(ddm * torch.abs(x - y))

    def mean_loss(self, x, y):
        return self.pixel_loss(x.view(x.size(0), -1).mean(1), y.view(y.size(0), -1).mean(1))

    def filter_wavelet_LL(self, x, norm=True):
        DWT2 = DWTForward(J=1, wave='haar', mode='reflect').cuda()
        LL, Hc = DWT2(x)

        return LL * 0.5 if norm else LL
    
    
class L1_Charbonnier_Loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_Loss,self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight


    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    