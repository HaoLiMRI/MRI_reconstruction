from torch.utils import data
import numpy as np
import nibabel as nib
import os, utils, pydicom, h5py
from artifact_simulation import motion_simulation_2D, motion_simulation_3D, ifft


############################################################
# get the distribution of MRI types
############################################################
def get_MRI_distribution(filepath, record_path):

    MRI_type = ['T1', 'T2', 'T1 POST', 'T2 POST', 'FLAIR']
    shapes = {'1.5T': [[] for _ in range(len(MRI_type))],
              '3T': [[] for _ in range(len(MRI_type))]}
    paths = {'1.5T': [[] for _ in range(len(MRI_type))],
             '3T': [[] for _ in range(len(MRI_type))]}
    FieldStrength = ['1.5T', '3T']
    # sort filepath by MRImage type and shape
    for p in filepath:
        ds = pydicom.read_file(p)
        img = ds.pixel_array
        # remove background
        std = np.std(img)
        mean = np.mean(img)
        if mean < 150 or std < 100:
            continue
        h, w = img.shape

        def mri_type(contrast):
            # search MRI contrasts type
            if 'FLAIR' in contrast: return 4
            elif 'T1' in contrast and ('POST' in contrast or 'Post' in contrast or 'post' in contrast): return 2
            elif 'T2' in contrast and ('POST' in contrast or 'Post' in contrast or 'post' in contrast): return 3
            elif 'T1' in contrast: return 0
            elif 'T2' in contrast or 't2' in contrast: return 1
            else: return 5

        idx = mri_type(ds.SeriesDescription)
        if ds.MagneticFieldStrength < 2:
            if [h, w] not in shapes['1.5T'][idx]:
                shapes['1.5T'][idx].append([h, w])
                paths['1.5T'][idx].append([])
            paths['1.5T'][idx][shapes['1.5T'][idx].index([h, w])].append(p)
        else:
            if [h, w] not in shapes['3T'][idx]:
                shapes['3T'][idx].append([h, w])
                paths['3T'][idx].append([])
            paths['3T'][idx][shapes['3T'][idx].index([h, w])].append(p)
    # Save all filepath of each MRI type and each shape.
    for fs in FieldStrength:
        for i in range(len(MRI_type)):
            # create folder
            path = os.path.join(record_path, fs, MRI_type[i])
            if not os.path.exists(path):
                os.makedirs(path)
            # write dicom path to to txt
            for j in range(len(shapes[fs][i])):
                file = open(path + '\\{}x{}.txt'.format(shapes[fs][i][j][0], shapes[fs][i][j][1]), 'w')
                for k in paths[fs][i][j]:
                    file.write(k + '\n')
                file.close()

############################################################
# get MRI types
############################################################
def get_MRI_type(filepath):

    record = []
    count = []
    # sort filepath by MRImage shape
    for p in filepath:
        ds = pydicom.read_file(p)
        if ds.SeriesDescription not in record:
            record.append(ds.SeriesDescription)
            count.append(1)
        else:
            count[record.index(ds.SeriesDescription)] += 1

    for i in range(len(record)):
        print('type: {} \t-- num: {}'.format(record[i], count[i]))

############################################################
# fastMRI brain dataset preprocess
# motion simulation
############################################################
def process_fastMRI(MRI_distribution_path, save_path, crop=False, gap=3):
    # read file path
    filepath = []
    file = open(MRI_distribution_path, 'r')
    for line in file:
        filepath.append(line.strip("\n"))

    save_path = os.path.join(save_path, 'size=128x128_gap={}_in_plane'.format(gap))

    # get subset index
    index = np.arange(len(filepath))
    train_idx = np.random.choice(index, 4000, replace=False)
    index = np.delete(index, train_idx)
    val_idx = np.random.choice(index, 500, replace=False)
    index = np.delete(np.arange(len(filepath)), np.concatenate([train_idx, val_idx]))
    test_idx = np.random.choice(index, 500, replace=False)

    idx_list = [train_idx, val_idx, test_idx]
    sets = ['train', 'val', 'test']
    for i in range(3):
        idx = idx_list[i]
        # create folder if not exists
        if not os.path.exists(os.path.join(save_path, sets[i])):
            os.makedirs(os.path.join(save_path, sets[i]))
        # read original data, simulate motion and save as npz
        for j in idx:
            if '.dcm' in filepath[j]:
                ds = pydicom.read_file(filepath[j])
                img = utils.normalize(ds.pixel_array)
                ma_img = motion_simulation_2D(img, gap=gap)
                if not crop:
                    np.savez(os.path.join(save_path, sets[i], str(j)+'.npz'), gt_img=img, ma_img=ma_img)
                else:
                    gt_crop, ma_crop = utils.crop_img(img, ma_img, crop_size=128, stride=96)
                    for k in range(len(gt_crop)):
                        np.savez(os.path.join(save_path, sets[i], '{}-{}.npz'.format(j, k)), gt_img=gt_crop[k], ma_img=ma_crop[k])

############################################################
# BraTS2020 dataset preprocess
# motion simulation
############################################################
def process_BraTS2020(data_path, save_path, crop=False, gap=3):

    filepath = []
    filename = []
    for root, dirs, files in os.walk(data_path):
        if len(files) != 0:
            for i in files:
                if 't1ce' not in i:
                    continue
                filepath.append(os.path.join(root, i))
                filename.append(i)

    save_path = os.path.join(save_path, 'size=128x128_gap={}_in_plane'.format(gap))

    filepath = np.tile(filepath, [50, 1]).T.reshape([-1, ])
    filename = np.tile(filename, [50, 1]).T.reshape([-1, ])

    index = np.arange(len(filepath))
    train_idx = np.random.choice(index, 240 * 50, replace=False)
    index = np.delete(index, train_idx)
    val_idx = np.random.choice(index, 30 * 50, replace=False)
    index = np.delete(np.arange(len(filepath)), np.concatenate([train_idx, val_idx]))
    test_idx = np.random.choice(index, 30 * 50, replace=False)

    idx_list = [train_idx, val_idx, test_idx]
    sets = ['train', 'val', 'test']
    for i in range(3):
        idx = idx_list[i]
        if not os.path.exists(os.path.join(save_path, sets[i])):
            os.makedirs(os.path.join(save_path, sets[i]))

        for j in idx:
            scan = nib.load(filepath[j]).get_fdata()[:, :, j % 50 + 50]
            if (scan == 0).all():
                continue
            img = utils.normalize(scan)
            ma_img = motion_simulation_2D(img, gap=gap)
            if not crop:
                np.savez(os.path.join(save_path, sets[i], filename[j].replace('.nii.gz', '-{}.npz'.format(j % 50))), gt_img=img, ma_img=ma_img)
            else:
                gt_crop, ma_crop = utils.crop_img(img, ma_img, crop_size=128, stride=112)
                for k in range(len(gt_crop)):
                    np.savez(os.path.join(save_path, sets[i], filename[j].replace('.nii.gz', '-{}-{}.npz'.format(j % 50, k))), gt_img=gt_crop[k], ma_img=ma_crop[k])

############################################################
# fastMRI brain dataset preprocess
# 3D motion simulation
############################################################
def process_fastMRI_3D(data_path, save_path, crop=False, gap=6):

    images = []
    ma_images = []
    files = os.listdir(data_path)
    save_path = os.path.join(save_path, 'size=128x128_gap={}_through_plane'.format(gap))
    for file in files:
        f = h5py.File(os.path.join(data_path, file), mode='r')
        kspace = np.array(f.get('kspace'))
        image = np.array([np.sqrt(np.sum([ifft(k) ** 2 for k in kspace[i]], axis=0))[192:-192, 6:-6] for i in range(len(kspace))])
        image = utils.normalize(image, 2e-6, 1e-8)
        images.append(image[:5])
        ma_image = np.transpose(motion_simulation_3D(np.transpose(image, [1, 2, 0]), gap), [2, 0, 1])
        ma_images.append(ma_image[:5])
    images = np.concatenate(images)
    ma_images = np.concatenate(ma_images)

    # get subset index
    index = np.arange(len(images))
    train_idx = np.random.choice(index, 800, replace=False)
    index = np.delete(index, train_idx)
    val_idx = np.random.choice(index, 100, replace=False)
    index = np.delete(np.arange(len(images)), np.concatenate([train_idx, val_idx]))
    test_idx = np.random.choice(index, 100, replace=False)

    idx_list = [train_idx, val_idx, test_idx]
    sets = ['train', 'val', 'test']
    for i in range(3):
        idx = idx_list[i]
        if not os.path.exists(os.path.join(save_path, sets[i])):
            os.makedirs(os.path.join(save_path, sets[i]))
        for j in idx:
            if not crop:
                np.savez(os.path.join(save_path, sets[i], files[j // 5].replace('.h5', '-{}.npz'.format(j % 5))), gt_img=images[j], ma_img=ma_images[j])
            else:
                gt_crop, ma_crop = utils.crop_img(images[j], ma_images[j], crop_size=128, stride=128)
                for k in range(len(gt_crop)):
                    np.savez(os.path.join(save_path, sets[i], files[j // 5].replace('.h5', '-{}-{}.npz'.format(j % 5, k))), gt_img=gt_crop[k], ma_img=ma_crop[k])

############################# data loader ##################################
class Dataset(data.Dataset):

    def __init__(self, main_path):

        self.filename = os.listdir(main_path)
        self.filepath = [os.path.join(main_path, filename) for filename in self.filename]

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, ind):

        img = np.load(self.filepath[ind])
        gt_img = np.expand_dims(img['gt_img'], axis=0).astype(np.float32)
        ma_img = np.expand_dims(img['ma_img'], axis=0).astype(np.float32)

        return {'gt_img': gt_img,
                'ma_img': ma_img,
                'filename': self.filename[ind]}


def get_dataloader(config, shuffle=True, evaluation=False):

    if not evaluation:
        gt_train_data = Dataset(os.path.join(config.datapath, 'train'))
        ma_train_data = Dataset(os.path.join(config.datapath, 'train'))
        val_data = Dataset(os.path.join(config.datapath, 'val'))

        gt_train_loader = data.DataLoader(dataset=gt_train_data, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle, drop_last=True)
        ma_train_loader = data.DataLoader(dataset=ma_train_data, batch_size=config.batch_size, pin_memory=True, shuffle=shuffle, drop_last=True)
        val_loader = data.DataLoader(dataset=val_data, batch_size=config.batch_size, pin_memory=True, shuffle=False)

        return ma_train_loader, gt_train_loader, val_loader
    else:
        test_data = Dataset(os.path.join(config.datapath, 'test'))
        test_loader = data.DataLoader(dataset=test_data, batch_size=1, pin_memory=True, shuffle=False)
        return test_loader


if __name__ == '__main__':

    # main_path = 'F:\\dataset\\fastMRI_brain_DICOM'
    # record_path ='E:\\dataset\\fastMRI_brain\\MRI_distribution'
    # filepath, _ = get_filepath(main_path)
    # get_MRI_distribution(filepath, record_path)

    # MRI_distribution_path = 'E:\\dataset\\fastMRI_brain\\MRI_distribution\\3T\\T1\\320x320.txt'
    # save_path = 'E:\\dataset\\fastMRI_brain\\Motion_Artifact_Reduction'
    # process_fastMRI(MRI_distribution_path, save_path, crop=True, gap=3)

    # main_path = 'E:\\dataset\\MICCAI_BraTS2020_TrainingData'
    # save_path = 'E:\\dataset\\BraTS2020\\Motion_Artifact_Reduction'
    # process_BraTS2020(main_path, save_path, crop=True, gap=3)

    data_path = 'G:\\dataset\\fastMRI_brain\\brain_multicoil_train'
    save_path = 'E:\\dataset\\fastMRI_brain\\Motion_Artifact_Reduction'
    process_fastMRI_3D(data_path, save_path, crop=True, gap=9)