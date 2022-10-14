import matplotlib.pyplot as plt
import numpy as np
import os
from tools.utils import get_filepath

os.environ['KMP_DUPLICATE_LIB_OK']='True'


############################################################
#  Visualize
############################################################
def display_images(images, titles, dir_save, cols=3, auto_show=False):

    if len(images) % cols == 0:
        rows = len(images) // cols
    else:
        rows = len(images) // cols + 1

    plt.figure(figsize=(5 * cols, 5 * rows), dpi=100)
    for i, (img, title) in enumerate(zip(images, titles)):

        if img is None:
            continue

        plt.subplot(rows, cols, i+1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.savefig(dir_save, bbox_inches='tight', pad_inches=0.1)
    if auto_show:
        plt.show()
    plt.close()


############################################################
#  draw prediction results after training
############################################################
def image_concatenation(sub_images, shapes, crop_size, stride):

    image = np.zeros(shapes)
    for i in range((shapes[0] - crop_size) // stride + 1):
        for j in range((shapes[1] - crop_size) // stride + 1):
            image[i * stride:i * stride + crop_size, j * stride:j * stride + crop_size] = sub_images[0]
            del sub_images[0]

    return image


def draw_test_result(pred_path, test_set_path, auto_show=False):

    filepath1, filename1 = get_filepath(pred_path)
    filepath2, filename2 = get_filepath(test_set_path)

    img_path = os.path.join(pred_path, 'visualize')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in range(len(filename2)):

        assert filename1[i] == filename2[i]
        pred_dat = np.load(filepath1[i])
        test_dat = np.load(filepath2[i])
        pred_img = pred_dat['pred']
        gt_img, ma_img = test_dat['gt_img'], test_dat['ma_img']

        titles = ['motion free', 'motion simulation', 'after correction', 'residual map1', 'residual map2', 'residual map3']
        images = [gt_img, ma_img, pred_img, gt_img - gt_img, ma_img - gt_img, pred_img - gt_img]

        display_images(images, titles, os.path.join(img_path, filename2[i].replace('.npz', '.jpg')), auto_show=auto_show)


def draw_test_result_crop(pred_path, test_set_path, shapes=(320, 320, 1), crop_size=128, auto_show=False):

    filepath1, filename1 = get_filepath(pred_path)
    filepath2, filename2 = get_filepath(test_set_path)

    img_path = os.path.join(pred_path, 'visualize')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    stride = int(crop_size * 3/4)
    sub_num = ((shapes[0] - crop_size) // stride + 1) * ((shapes[1] - crop_size) // stride + 1)

    for i in range(len(filename2) // sub_num):
        pred_imgs, gt_imgs, ma_imgs = [], [], []
        for j in range(sub_num):
            assert filename1[i * 9 + j] == filename2[i * 9 + j]
            pred_dat = np.load(filepath1[i * 9 + j])
            test_dat = np.load(filepath2[i * 9 + j])
            pred_imgs.append(pred_dat['pred'])
            gt_imgs.append(test_dat['gt_img'])
            ma_imgs.append(test_dat['ma_img'])

        pred_img = image_concatenation(pred_imgs, shapes, crop_size, stride)
        gt_img = image_concatenation(gt_imgs, shapes, crop_size, stride)
        ma_img = image_concatenation(ma_imgs, shapes, crop_size, stride)

        titles = ['motion free', 'motion simulation', 'after correction', 'residual map1', 'residual map2', 'residual map3']
        images = [gt_img, ma_img, pred_img, gt_img - gt_img, ma_img - gt_img, pred_img - gt_img]

        display_images(images, titles, os.path.join(img_path, filename2[i * 9].replace('.npz', '.jpg')), auto_show=auto_show)


