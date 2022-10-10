import pydicom
import numpy as np
import os
from tools.motion_artifact import motion_simulation_2D, crop_img_2D
from tools.utils import get_filepath, normalize


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def MRI_distribution(filepath):

    shapes = []
    path = []
    # sort filepath by MRImage shape
    for p in filepath:
        ds = pydicom.read_file(p)
        img = ds.pixel_array
        h, w = img.shape
        if [h, w] not in shapes:
            shapes.append([h, w])
            path.append([])
        path[shapes.index([h, w])].append(p)
    # Save all filepath of each shape.
    for i in range(len(shapes)):
        file = open('E:\\dataset\\MRI_distribution\\shape\\{}x{}.txt'.format(shapes[i][0], shapes[i][1]), 'w')
        for j in path[i]:
            file.write(j + '\n')
        file.close()

def MRI_distribution_1(filepath):

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
            path = 'E:\\dataset\\MRI_distribution\\{}\\{}'.format(fs, MRI_type[i])
            if not os.path.exists(path):
                os.makedirs(path)
            # write dicom path to to txt
            for j in range(len(shapes[fs][i])):
                file = open(path + '\\{}x{}.txt'.format(shapes[fs][i][j][0], shapes[fs][i][j][1]), 'w')
                for k in paths[fs][i][j]:
                    file.write(k + '\n')
                file.close()


def grt_MRI_type(filepath):

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


def create_MA_set(MRI_distribution_path, new_path, crop=False):
    # read file path
    filepath = []
    file = open(MRI_distribution_path, 'r')
    for line in file:
        filepath.append(line.strip("\n"))

    # get subset index
    index = np.arange(len(filepath))
    train_idx = np.random.choice(index, 4000, replace=False)
    index = np.delete(index, train_idx)
    val_idx = np.random.choice(index, 500, replace=False)
    index = np.delete(np.arange(len(filepath)), np.concatenate([train_idx, val_idx]))
    test_idx = np.random.choice(index, 500, replace=False)

    idx_list = [train_idx, val_idx, test_idx]
    filename = ['train', 'val', 'test']
    for i in range(3):
        idx = idx_list[i]
        # create folder if not exists
        if not os.path.exists(os.path.join(new_path, filename[i])):
            os.makedirs(os.path.join(new_path, filename[i]))
        # read original data, simulate motion and save as npz
        for j in idx:
            if '.dcm' in filepath[j]:
                ds = pydicom.read_file(filepath[j])
                img = normalize(ds.pixel_array)
                ma_img = motion_simulation_2D(img)
                if not crop:
                    np.savez(os.path.join(new_path, filename[i], str(j)+'.npz'), gt_img=img[:, :, np.newaxis], ma_img=ma_img[:, :, np.newaxis])
                else:
                    gt_crop, ma_crop = crop_img_2D(img, ma_img)
                    for k in range(len(gt_crop)):
                        np.savez(os.path.join(new_path, filename[i], '{}-{}.npz'.format(j, k)), gt_img=gt_crop[k][:, :, np.newaxis], ma_img=ma_crop[k][:, :, np.newaxis])


if __name__ == '__main__':

    # main_path = 'F:\\dataset\\fastMRI_brain_DICOM'
    # filepath, _ = get_filepath(main_path)
    # MRI_distribution_1(filepath)


    MRI_distribution_path = 'E:\\dataset\\MRI_distribution\\3T\\T1\\320x320.txt'
    new_path = 'E:\\dataset\\Motion Artifact\\crop-128x128'
    create_MA_set(MRI_distribution_path, new_path, crop=True)

