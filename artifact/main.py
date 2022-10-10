from model.cycle_gan import cycleGAN
from model.mdva_gan import mdvaGAN
from dataset.data_loader import get_dataloader, Dataset
from option import args
from tools import utils, visualize
import os


if __name__ == '__main__':

    # device setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)

    if args.manual_seed is not None:
        utils.init_random_seed(args.manual_seed)

    if not os.path.exists(args.snap_path):
        os.makedirs(args.snap_path)

    args_file = open(os.path.join(args.snap_path, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30, ' ') + '\t' + str(v) + '\n')
    args_file.close()

    # Load datasets
    data_loader = get_dataloader(args.main_datapath, args.batch_size)
    # mean_ssim, mean_psnr = utils.cal_original_ssim_psnr(data_loader[-1])     # 0.8824   30.41
    # print(mean_ssim, mean_psnr)

    # define model
    model = cycleGAN(args, data_loader)

    model.train()

    if args.inference:
        test_set = Dataset(os.path.join(args.main_datapath, 'test'), load_gt=False, load_ma=True)
        print('Inference:')
        model.eval(test_set)
        print('Done!\nVisualize:')
        visualize.draw_test_result_crop(args.result_path, os.path.join(args.main_datapath, 'test'))
        print('Done!')