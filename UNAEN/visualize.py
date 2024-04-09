import matplotlib.pyplot as plt
import numpy as np
import os, utils
import cv2, torch


os.environ['KMP_DUPLICATE_LIB_OK']='True'


image_params = {'fastMRI_brain': {'shape': [320, 320], 'crop_size': 128, 'stride': 96},
                'BraTS2020': {'shape': [240, 240], 'crop_size': 128, 'stride': 112}}


############################################################
#  visualize all prediction results after training
############################################################
def visualize_all(dataset='BraTS2020', method='UNAEN', gap=3, simu_type='in_plane', concat=True):

    test_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\test'.format(dataset, gap, simu_type)
    inference_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\Inference\\{}'.format(dataset, gap, simu_type, method)
    filenames = os.listdir(inference_path)

    img_path = inference_path.replace('Inference', 'Visualize')
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    titles = ['ground truth', 'before correction', 'after correction']
    for i in titles:
        if not os.path.exists(os.path.join(img_path, i)):
            os.makedirs(os.path.join(img_path, i))

    shape = image_params[dataset]['shape']
    crop_size = image_params[dataset]['crop_size']
    stride = image_params[dataset]['stride']
    patch_num = ((shape[0] - crop_size) // stride + 1) * ((shape[1] - crop_size) // stride + 1)

    metrics_file = open(os.path.join(img_path, 'metrics log.txt'), 'a+')
    if not concat:
        for i in range(len(filenames)):
            pred_dat = np.load(os.path.join(inference_path, filenames[i]))
            test_dat = np.load(os.path.join(test_path, filenames[i]))
            images = [test_dat['gt_img'], test_dat['ma_img'], pred_dat['pred']]
            for j in range(len(images)):
                cv2.imwrite(os.path.join(img_path, titles[j], filenames[i].replace('npz', 'bmp')), images[j]*255)

            gt_tensor = torch.tensor([[np.squeeze(test_dat['gt_img'])]])
            pred_tensor = torch.tensor([[np.squeeze(pred_dat['pred'])]])
            ssim = utils.ssim(gt_tensor, pred_tensor).item()
            psnr = utils.psnr(gt_tensor, pred_tensor).item()
            metrics_file.write('restored image %s : ssim=%.4f and psnr=%.4f\n' % (filenames[i].split('.')[0], ssim, psnr))
    else:
        for i in range(len(filenames) // patch_num):
            pred_imgs, gt_imgs, ma_imgs = [], [], []
            for j in range(patch_num):
                pred_dat = np.load(os.path.join(inference_path, filenames[i * patch_num + j]))
                test_dat = np.load(os.path.join(test_path, filenames[i * patch_num + j]))
                pred_imgs.append(pred_dat['pred'])
                gt_imgs.append(test_dat['gt_img'])
                ma_imgs.append(test_dat['ma_img'])

            pred_img = utils.image_concatenation(pred_imgs, shape, crop_size, stride)
            gt_img = utils.image_concatenation(gt_imgs, shape, crop_size, stride)
            ma_img = utils.image_concatenation(ma_imgs, shape, crop_size, stride)

            images = [gt_img, ma_img, pred_img]
            for j in range(len(images)):
                cv2.imwrite(os.path.join(img_path, titles[j], filenames[i * patch_num].replace('npz', 'bmp')), images[j] * 255)

            gt_tensor = torch.tensor([[np.squeeze(gt_img)]])
            pred_tensor = torch.tensor([[np.squeeze(pred_img)]])
            ssim = utils.ssim(gt_tensor, pred_tensor).item()
            psnr = utils.psnr(gt_tensor, pred_tensor).item()
            metrics_file.write('restored image %s : ssim=%.4f and psnr=%.4f\n' % (filenames[i * patch_num].split('.')[0], ssim, psnr))


############################################################
#  visualize selection
############################################################
# def visualize_selection(dataset='fastMRI_brain', name=39, gap=3, simu_type='in_plane', crop_w=(130, 110, 230, 210)):
def visualize_selection(dataset='BraTS2020', name='BraTS20_Training_024_t1ce-42', gap=3, simu_type='in_plane', crop_w=(110, 140, 180, 210)):

    test_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\test'.format(dataset, gap, simu_type)
    inference_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\Inference'.format(dataset, gap, simu_type)
    methods = os.listdir(inference_path)
    save_path = os.path.join(inference_path.replace('Inference', 'Visualize'), '{}'.format(name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape = image_params[dataset]['shape']
    crop_size = image_params[dataset]['crop_size']
    stride = image_params[dataset]['stride']
    patch_num = ((shape[0] - crop_size) // stride + 1) * ((shape[1] - crop_size) // stride + 1)

    gt_imgs, ma_imgs = [], []
    for i in range(patch_num):
        test_data = np.load(os.path.join(test_path, '{}-{}.npz'.format(name, i)))
        gt_imgs.append(test_data['gt_img'])
        ma_imgs.append(test_data['ma_img'])

    gt_img = utils.image_concatenation(gt_imgs, shape, crop_size, stride)
    ma_img = utils.image_concatenation(ma_imgs, shape, crop_size, stride)
    cv2.imwrite(os.path.join(save_path, 'ground true.bmp'), gt_img * 255)
    cv2.imwrite(os.path.join(save_path, 'motion artifact.bmp'), ma_img * 255)
    heat_img = utils.normalize(np.fabs(ma_img - gt_img), 0.15, 0, True)
    heat_img = (heat_img * 255).astype(np.uint8)
    heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_path, 'ma-heat.bmp'), heat_img)
    if crop_w is not None:
        pic = cv2.rectangle(gt_img * 255, (crop_w[1], crop_w[0]), (crop_w[3], crop_w[2]), [255, 255, 255], thickness=1)
        cv2.imwrite(os.path.join(save_path, 'ground true-location.bmp'), pic)
        cv2.imwrite(os.path.join(save_path, 'gt-crop.bmp'), gt_img[crop_w[0]:crop_w[2], crop_w[1]:crop_w[3]] * 255)
        cv2.imwrite(os.path.join(save_path, 'ma-crop.bmp'), ma_img[crop_w[0]:crop_w[2], crop_w[1]:crop_w[3]] * 255)
        cv2.imwrite(os.path.join(save_path, 'ma-heat-crop.bmp'), heat_img[crop_w[0]:crop_w[2], crop_w[1]:crop_w[3]])

    metrics_file = open(os.path.join(save_path, 'metrics log.txt'), 'a+')
    if crop_w is not None:
        metrics_file.write('window: [{}, {}, {}, {}]\n'.format(crop_w[0], crop_w[1], crop_w[2], crop_w[3]))
    gt_tensor = torch.tensor([[gt_img]])
    ma_tensor = torch.tensor([[ma_img]])
    metrics_file.write('ma-image: ssim=%.4f and psnr=%.4f\n' % (utils.ssim(gt_tensor, ma_tensor).item(),
                                                                utils.psnr(gt_tensor, ma_tensor).item()))

    for method in methods:
        pred_imgs = []
        for i in range(patch_num):
            pred_imgs.append(np.load(os.path.join(inference_path, method, '{}-{}.npz'.format(name, i)))['pred'])
        pred_img = utils.image_concatenation(pred_imgs, shape, crop_size, stride)
        cv2.imwrite(os.path.join(save_path, '{}.bmp'.format(method)), pred_img * 255)
        heat_img = utils.normalize(np.fabs(pred_img - gt_img), 0.15, 0, True)
        heat_img = (heat_img * 255).astype(np.uint8)
        heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, '{}-heat.bmp'.format(method)), heat_img)
        if  crop_w is not None:
            cv2.imwrite(os.path.join(save_path, '{}-crop.bmp'.format(method)), pred_img[crop_w[0]:crop_w[2], crop_w[1]:crop_w[3]] * 255)
            cv2.imwrite(os.path.join(save_path, '{}-heat-crop.bmp'.format(method)), heat_img[crop_w[0]:crop_w[2], crop_w[1]:crop_w[3]])
        pred_img = torch.tensor([[pred_img]])
        metrics_file.write(method + ' artifact reduction result: ssim=%.4f and psnr=%.4f\n' % (utils.ssim(gt_tensor, pred_img).item(),
                                                                                               utils.psnr(gt_tensor, pred_img).item()))


def visualize_extracted_map(dataset='fastMRI_brain', name=437, gap=6, simu_type='in_plane'):

    test_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\test'.format(dataset, gap, simu_type)
    inference_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\Inference'.format(dataset, gap, simu_type)
    methods = os.listdir(inference_path)
    save_path = os.path.join(inference_path.replace('Inference', 'Visualize'), '{}'.format(name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shape = image_params[dataset]['shape']
    crop_size = image_params[dataset]['crop_size']
    stride = image_params[dataset]['stride']
    patch_num = ((shape[0] - crop_size) // stride + 1) * ((shape[1] - crop_size) // stride + 1)

    gt_imgs, ma_imgs = [], []
    for i in range(patch_num):
        test_data = np.load(os.path.join(test_path, '{}-{}.npz'.format(name, i)))
        gt_imgs.append(test_data['gt_img'])
        ma_imgs.append(test_data['ma_img'])

    gt_img = utils.image_concatenation(gt_imgs, shape, crop_size, stride)
    ma_img = utils.image_concatenation(ma_imgs, shape, crop_size, stride)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(shape[0] / 100.0, shape[1] / 100.0)  # 输出原始图像width*height的像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(utils.normalize(gt_img - ma_img, 0.15, -0.15, True), cmap='gray')
    plt.savefig(os.path.join(save_path, 'extraction', 'ma-extraction.jpg'))

    for method in methods:
        pred_imgs = []
        for i in range(patch_num):
            pred_imgs.append(np.load(os.path.join(inference_path, method, '{}-{}.npz'.format(name, i)))['pred'])
        pred_img = utils.image_concatenation(pred_imgs, shape, crop_size, stride)

        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(shape[0] / 100.0, shape[1] / 100.0)  # 输出原始图像width*height的像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(utils.normalize(gt_img - pred_img, 0.15, -0.15, True), cmap='gray')
        plt.savefig(os.path.join(save_path, 'extraction', '{}-extraction.jpg'.format(method)))


if __name__ == '__main__':

    visualize_selection()