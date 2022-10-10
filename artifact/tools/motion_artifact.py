import numpy as np
from scipy import ndimage
from tools.utils import normalize


gap = 6
etl = 10
angle = 5       # rotation angle in plane
time_stay = 5       # Rotation duration
time_rot = 2        # Rotation time consumption

crop_size = 128


def motion_simulation_2D(img):

    # img = img / np.max(img)
    img_R1 = ndimage.rotate(img, angle/(time_rot+1), reshape=False, order=3, mode='nearest')
    img_R2 = ndimage.rotate(img, 2*angle/(time_rot+1), reshape=False, order=3, mode='nearest')
    img_R = ndimage.rotate(img, angle, reshape=False, order=3, mode='nearest')
    img_L1 = ndimage.rotate(img, -angle/(time_rot+1), reshape=False, order=3, mode='nearest')
    img_L2 = ndimage.rotate(img, -2*angle/(time_rot+1), reshape=False, order=3, mode='nearest')
    img_L = ndimage.rotate(img, -angle, reshape=False, order=3, mode='nearest')

    img_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    img_fft_R1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_R1)))
    img_fft_R2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_R2)))
    img_fft_R = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_R)))
    img_fft_L1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_L1)))
    img_fft_L2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_L2)))
    img_fft_L = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_L)))

    center = int(img_fft.shape[1] / 2)
    index = gap / 2

    while center - index * etl > 0:

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_L1[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_L1[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_L2[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_L2[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + time_stay / 2) * etl < 0:
            break

        img_fft[:, int(center-(index+time_stay/2)*etl) : int(center-index*etl)] = img_fft_L[:, int(center-(index+time_stay/2)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+time_stay/2)*etl)] = img_fft_L[:, int(center+index*etl) : int(center+(index+time_stay/2)*etl)]
        index = index + time_stay / 2

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_L2[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_L2[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_L1[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_L1[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5 + gap / 2

        if center - index * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_R1[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_R1[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_R2[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_R2[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + time_stay / 2) * etl < 0:
            break

        img_fft[:, int(center-(index+time_stay/2)*etl) : int(center-index*etl)] = img_fft_R[:, int(center-(index+time_stay/2)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+time_stay/2)*etl)] = img_fft_R[:, int(center+index*etl) : int(center+(index+time_stay/2)*etl)]
        index = index + time_stay / 2

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)]=img_fft_R2[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)]=img_fft_R2[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5

        if center - (index + 0.5) * etl < 0:
            break

        img_fft[:, int(center-(index+0.5)*etl) : int(center-index*etl)] = img_fft_R1[:, int(center-(index+0.5)*etl) : int(center-index*etl)]
        img_fft[:, int(center+index*etl) : int(center+(index+0.5)*etl)] = img_fft_R1[:, int(center+index*etl) : int(center+(index+0.5)*etl)]
        index = index + 0.5 + gap / 2

    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img_fft)))
    img = np.sqrt(img.real**2 + img.imag**2)

    img = normalize(img)

    return img


def crop_img_2D(gt_img, ma_img):

    stride = int(crop_size * 3/4)

    gt_crop = []
    ma_crop = []

    for i in range((gt_img.shape[0] - crop_size) // stride + 1):
        for j in range((gt_img.shape[1] - crop_size) // stride + 1):
            gt_crop.append(gt_img[i * stride:i * stride + crop_size, j * stride:j * stride + crop_size])
            ma_crop.append(ma_img[i * stride:i * stride + crop_size, j * stride:j * stride + crop_size])

    return np.array(gt_crop), np.array(ma_crop)
