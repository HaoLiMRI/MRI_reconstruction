import numpy as np
import os, torch
import utils


def metric_evaluation(dataset='fastMRI_brain', gap='9', simu_type='in_plane', method='UNAEN'):

    test_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\test'.format(dataset, gap, simu_type)
    inference_path = 'E:\\dataset\\{}\\Motion_Artifact_Reduction\\size=128x128_gap={}_{}\\Inference\\{}'.format(dataset, gap, simu_type, method)
    files = os.listdir(test_path)

    crop_size = 128
    patch_params = dict(fastMRI_brain=dict(shape=[320, 320], stride=96) if simu_type=='in_plane' else dict(shape=[384, 384], stride=128),
                        BraTS2020=dict(shape=[240, 240], stride=112))[dataset]
    patch_num = ((patch_params['shape'][0] - crop_size) // patch_params['stride'] + 1) *\
                ((patch_params['shape'][1] - crop_size) // patch_params['stride'] + 1)

    eval_ssim, eval_psnr = 0., 0.
    ma_ssim, ma_psnr = 0., 0.
    num = len(files) // patch_num
    for i in range(num):
        pred_imgs, gt_imgs, ma_imgs = [], [], []
        for j in range(patch_num):
            pred_dat = np.load(os.path.join(inference_path, files[i * patch_num + j]))
            test_dat = np.load(os.path.join(test_path, files[i * patch_num + j]))
            pred_imgs.append(pred_dat['pred'])
            gt_imgs.append(test_dat['gt_img'])
            ma_imgs.append(test_dat['ma_img'])

        pred_img = utils.image_concatenation(pred_imgs, patch_params['shape'], crop_size, patch_params['stride'])
        gt_img = utils.image_concatenation(gt_imgs, patch_params['shape'], crop_size, patch_params['stride'])
        ma_img = utils.image_concatenation(ma_imgs, patch_params['shape'], crop_size, patch_params['stride'])

        pred_img = torch.tensor([[pred_img]])
        gt_img = torch.tensor([[gt_img]])
        ma_img = torch.tensor([[ma_img]])

        eval_ssim += utils.ssim(pred_img, gt_img).item()
        eval_psnr += utils.psnr(pred_img, gt_img).item()
        ma_ssim += utils.ssim(ma_img, gt_img).item()
        ma_psnr += utils.psnr(ma_img, gt_img).item()

    print('The Evaluation Results of {} on {} dataset with gap={} and simulation {}:'.format(method, dataset, gap, simu_type))
    print('Before Reduction: ssim={:.4f} and psnr={:.4f}.'.format(ma_ssim / num, ma_psnr / num))
    print('After Reduction : ssim={:.4f} and psnr={:.4f}.'.format(eval_ssim / num, eval_psnr / num))



if __name__ == '__main__':

    metric_evaluation()

