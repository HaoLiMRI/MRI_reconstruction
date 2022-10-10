from tools.utils import get_filepath
from torch.utils import data
import numpy as np
import os


class Dataset(data.Dataset):

    def __init__(self, main_path, load_gt, load_ma):

        self.load_gt = load_gt
        self.load_ma = load_ma
        self.filepath, self.filename = get_filepath(main_path)

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, ind):

        img = np.load(self.filepath[ind])
        gt_img, ma_img = img['gt_img'], img['ma_img']

        if self.load_gt and self.load_ma:
            return np.transpose(ma_img, (2, 0, 1)), np.transpose(gt_img, (2, 0, 1))
        if self.load_gt:
            return np.transpose(gt_img, (2, 0, 1))
        if self.load_ma:
            return np.transpose(ma_img, (2, 0, 1))


def get_dataloader(main_path, batch_size, shuffle=True):

    gt_train_data = Dataset(os.path.join(main_path, 'train'), load_gt=True, load_ma=False)
    ma_train_data = Dataset(os.path.join(main_path, 'train'), load_gt=False, load_ma=True)
    val_data = Dataset(os.path.join(main_path, 'val'), load_gt=True, load_ma=True)

    gt_train_loader = data.DataLoader(dataset=gt_train_data, batch_size=batch_size, pin_memory=True, shuffle=shuffle, drop_last=True)
    ma_train_loader = data.DataLoader(dataset=ma_train_data, batch_size=batch_size, pin_memory=True, shuffle=shuffle, drop_last=True)
    val_loader = data.DataLoader(dataset=val_data, batch_size=batch_size, pin_memory=True, shuffle=False)

    return ma_train_loader, gt_train_loader, val_loader


def get_dataloader_supervise(main_path, batch_size, shuffle=True):

    train_data = Dataset(os.path.join(main_path, 'train'), load_gt=True, load_ma=True)
    val_data = Dataset(os.path.join(main_path, 'val'), load_gt=True, load_ma=True)

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, pin_memory=True, shuffle=shuffle, drop_last=True)
    val_loader = data.DataLoader(dataset=val_data, batch_size=batch_size, pin_memory=True, shuffle=False)

    return train_loader, val_loader

