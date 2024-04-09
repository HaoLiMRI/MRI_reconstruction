from scipy import ndimage
import numpy as np


def fft(img):
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    return kspace


def ifft(kspace):
    img = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace)))
    img = np.sqrt(img.real ** 2 + img.imag ** 2)
    return img


def replace_kspcae_line(kspace_org, kspace_R1, kspace_R2, kspace_R, kspace_L1, kspace_L2, kspace_L,
                        etl, time_stay, gap):

    center = int(kspace_org.shape[1] / 2)
    index = gap / 2

    while center - index * etl > 0:

        # first gap: stay at center
        # kspace_org[:, int(center - index * etl) : int(center + index * etl)] = kspace_org[:, int(center - index * etl) : int(center + index * etl)]

        # rotate to left: first rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_L1[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_L1[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # rotate to left: second rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_L2[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_L2[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # stay at left
        if center - (index + time_stay / 2) * etl < 0:
            break
        kspace_org[:, int(center - (index + time_stay / 2) * etl): int(center - index * etl)] = kspace_L[:, int(center - (index + time_stay / 2) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + time_stay / 2) * etl)] = kspace_L[:, int(center + index * etl): int(center + (index + time_stay / 2) * etl)]
        index = index + time_stay / 2

        # back to center: first rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_L2[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_L2[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # back to center: second rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_L1[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_L1[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # stay at center
        if center - (index + gap / 2) * etl < 0:
            break
        # kspace_org[:, int(center - (index + gap / 2) * etl) : int(center - index * etl)] = kspace_org[:, int(center - (index + gap / 2) * etl) : int(center - index * etl)]
        # kspace_org[:, int(center + index * etl) : int(center + (index + gap / 2) * etl)] = kspace_org[:, int(center + index * etl) : int(center + (index + gap / 2) * etl)]
        index = index + gap / 2

        # rotate to right: first rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_R1[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_R1[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # rotate to right: second rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_R2[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_R2[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # stay at right
        if center - (index + time_stay / 2) * etl < 0:
            break
        kspace_org[:, int(center - (index + time_stay / 2) * etl): int(center - index * etl)] = kspace_R[:, int(center - (index + time_stay / 2) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + time_stay / 2) * etl)] = kspace_R[:, int(center + index * etl): int(center + (index + time_stay / 2) * etl)]
        index = index + time_stay / 2

        # back to center: first rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_R2[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_R2[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # back to center: second rotation etl
        if center - (index + 0.5) * etl < 0:
            break
        kspace_org[:, int(center - (index + 0.5) * etl): int(center - index * etl)] = kspace_R1[:, int(center - (index + 0.5) * etl): int(center - index * etl)]
        kspace_org[:, int(center + index * etl): int(center + (index + 0.5) * etl)] = kspace_R1[:, int(center + index * etl): int(center + (index + 0.5) * etl)]
        index = index + 0.5

        # stay at center
        if center - (index + gap / 2) * etl<0:
            break
        # kspace_org[:, int(center - (index + gap / 2) * etl): int(center - index * etl)] = kspace_org[:, int(center - (index + gap / 2) * etl): int(center - index * etl)]
        # kspace_org[:, int(center + index * etl): int(center + (index + gap / 2) * etl)] = kspace_org[:, int(center + index * etl): int(center + (index + gap / 2) * etl)]
        index = index + gap / 2

    return kspace_org


def motion_simulation_2D(img, gap=3):

    # some parameters
    etl = 10                    # EG
    angle = 5                   # rotation angle in plane
    time_stay = 5               # Rotation duration
    time_rot = 2                # Rotation time consumption

    # rotation
    img_R1 = ndimage.rotate(img, angle / (time_rot + 1), reshape=False, order=3, mode='nearest')
    img_R2 = ndimage.rotate(img, 2 * angle/(time_rot + 1), reshape=False, order=3, mode='nearest')
    img_R = ndimage.rotate(img, angle, reshape=False, order=3, mode='nearest')
    img_L1 = ndimage.rotate(img, -angle / (time_rot + 1), reshape=False, order=3, mode='nearest')
    img_L2 = ndimage.rotate(img, -2 * angle / (time_rot + 1), reshape=False, order=3, mode='nearest')
    img_L = ndimage.rotate(img, -angle, reshape=False, order=3, mode='nearest')
    # fft
    kspace_org = fft(img)
    kspace_R1 = fft(img_R1)
    kspace_R2 = fft(img_R2)
    kspace_R = fft(img_R)
    kspace_L1 = fft(img_L1)
    kspace_L2 = fft(img_L2)
    kspace_L = fft(img_L)
    # motion simulation
    kspace = replace_kspcae_line(kspace_org, kspace_R1, kspace_R2, kspace_R, kspace_L1, kspace_L2, kspace_L,
                                  etl, time_stay, gap)
    # ifft
    img = ifft(kspace)
    img = np.sqrt(img.real ** 2 + img.imag ** 2)

    # img = utils.normalize(img)

    return img.astype(np.float32)


def motion_simulation_3D(img, gap=18):

    # some parameters
    H, W, C = img.shape
    time_rot = 2                # Rotation time consumption
    time_stay = 5               # Rotation duration
    in_plane_rot = 5            # rotation angle in plane
    through_plane_rot = 5       # rotation angle through plane
    etl = 80                    # EG

    # in plane rotation
    img_R1 = ndimage.rotate(img, in_plane_rot / (time_rot + 1),      axes=(0, 1), reshape=False, order=3, mode='nearest')
    img_R2 = ndimage.rotate(img, 2 * in_plane_rot / (time_rot + 1),  axes=(0, 1), reshape=False, order=3, mode='nearest')
    img_R  = ndimage.rotate(img, in_plane_rot,                       axes=(0, 1), reshape=False, order=3, mode='nearest')
    img_L1 = ndimage.rotate(img, -in_plane_rot / (time_rot + 1),     axes=(0, 1), reshape=False, order=3, mode='nearest')
    img_L2 = ndimage.rotate(img, -2 * in_plane_rot / (time_rot + 1), axes=(0, 1), reshape=False, order=3, mode='nearest')
    img_L  = ndimage.rotate(img, -in_plane_rot,                      axes=(0, 1), reshape=False, order=3, mode='nearest')
    # through plane rotation
    img_R1 = ndimage.rotate(img_R1, -through_plane_rot / (time_rot + 1),     axes=(0, 2), reshape=False, order=3, mode='nearest')
    img_R2 = ndimage.rotate(img_R2, -2 * through_plane_rot / (time_rot + 1), axes=(0, 2), reshape=False, order=3, mode='nearest')
    img_R  = ndimage.rotate(img_R,  -through_plane_rot,                      axes=(0, 2), reshape=False, order=3, mode='nearest')
    img_L1 = ndimage.rotate(img_L1, -through_plane_rot / (time_rot + 1),     axes=(0, 2), reshape=False, order=3, mode='nearest')
    img_L2 = ndimage.rotate(img_L2, -2 * through_plane_rot / (time_rot + 1), axes=(0, 2), reshape=False, order=3, mode='nearest')
    img_L  = ndimage.rotate(img_L,  -through_plane_rot,                      axes=(0, 2), reshape=False, order=3, mode='nearest')
    # fft
    kspace_org = fft(img).reshape([H, W * C])
    kspace_R1 = fft(img_R1).reshape([H, W * C])
    kspace_R2 = fft(img_R2).reshape([H, W * C])
    kspace_R = fft(img_R).reshape([H, W * C])
    kspace_L1 = fft(img_L1).reshape([H, W * C])
    kspace_L2 = fft(img_L2).reshape([H, W * C])
    kspace_L = fft(img_L).reshape([H, W * C])
    # motion simulation
    kspace = replace_kspcae_line(kspace_org, kspace_R1, kspace_R2, kspace_R, kspace_L1, kspace_L2, kspace_L,
                                  etl, time_stay, gap).reshape([H, W, C])
    # ifft
    img = ifft(kspace)
    img = np.sqrt(img.real ** 2 + img.imag ** 2)

    # img = utils.normalize(img)

    return img.astype(np.float32)