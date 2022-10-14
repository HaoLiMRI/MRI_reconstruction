import argparse


parser = argparse.ArgumentParser(description='gan')

# Hardware specifications
parser.add_argument('--gpu_id', type=str, default='0', help='specify GPU ID to use')
parser.add_argument('--manual_seed', type=int, default=1)

# data specifications
parser.add_argument('--patch_size', type=int, default=64, help='patch size') # default = 128 (in the paper)
parser.add_argument('--main_datapath', type=str, default='E:\\dataset\\Motion Artifact\\crop-128x128', help='dataset root directory')
parser.add_argument('--result_path', type=str, default='E:\\dataset\\Motion Artifact\\mdva-gan-results', help='directory of test results')

# Train specificaions
parser.add_argument('--epochs', type=int, default=50, help='total epochs')
parser.add_argument('--batch_size', type=int, default=4, help='size of each batch') # default = 8 (in the paper)        # 4 / 8
parser.add_argument('--n_disc', type=int, default=1, help='number of iteration for discriminator update in one epoch')
parser.add_argument('--n_gen', type=int, default=4, help='number of iteration for generator update in one epoch')
parser.add_argument('--log_step', type=str, default=1000, help='log frequency')

# Optimizer specificaions
parser.add_argument('--lr_G', type=float, default=1e-4, help='initial learning rate of generator')
parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate of discriminator')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.99, help='ADAM beta2')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--apply_scheduler', type=bool, default=True, help='apply scheduler or not during training')

# loss specificaions
parser.add_argument('--lambda_ssim', type=float, default=0.5, help='SSIM loss weight')          # 0.5 - 0
parser.add_argument('--lambda_adv', type=float, default=0.1, help='adversarial loss weight')      # 0.1 -- 0.01
parser.add_argument('--lambda_cyc', type=float, default=1, help='cycle loss weight')
parser.add_argument('--lambda_sty', type=float, default=0.1, help='style loss weight')            # down 0.1
parser.add_argument('--lambda_gp', type=float, default=0, help='gradient penalty loss weight')
parser.add_argument('--GPloss', type=bool, default=False, help='whether to use GPloss')      # false

# model specificaions
parser.add_argument('--model', type=str, default='cycle_gan')
parser.add_argument('--n_hidden_feats', type=int, default=64, help='number of feature vectors in hidden layer')
parser.add_argument('--input_channels', type=int, default=1, help='number of input slice numbers')
parser.add_argument('--output_channels', type=int, default=1, help='number of output slice numbers')

# base specificaions
parser.add_argument('--inference', type=bool, default=False, help='whether performing inference')
parser.add_argument('--snap_path', type=str, default='results\\mdva-gan', help='path to save model weights')
parser.add_argument('--load_weight', type=bool, default=False, help='load checkpoint')


args = parser.parse_args()
