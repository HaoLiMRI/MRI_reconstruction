""" The code to organize the MRI SR target domain dataset, which contains only the measured MRI LR data for testing purpose. In the other word, this file is 
the dataloader for test dataset. """


# import gzip
# import os
# import pickle
# import urllib

import numpy as np
import torch as tc
# import torch.utils.data as data
# from torchvision import datasets, transforms
import h5py
import os


def EVAL_LOADER(root, load_hr, load_lr):
    """
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

#    def __init__(self, root):
    """Init USPS dataset."""
        # init params
#    self.root = os.path.expanduser(root)
#    self.forward()
#    def forward(self):
    print(root)
    print('One more low resolution image set exist')
    file_data = h5py.File(root, 'r')
    if load_lr:
        data_low_resolution = file_data['LR'][:] #----- numpy array
        print(np.shape(data_low_resolution))
        print(data_low_resolution.dtype)
        torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
        print(np.shape(torch_data_low_resolution))
        torch_data_low_resolution_sequence = torch_data_low_resolution
        print(np.shape(torch_data_low_resolution_sequence))
        torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
        print(np.shape(torch_data_low_resolution_sequence))
        
        return torch_data_low_resolution_sequence
    
    if load_hr:
        data_high_resolution = file_data['HRGT'][:] #----- numpy array
        print(np.shape(data_high_resolution))
        print(data_high_resolution.dtype)
        torch_data_high_resolution = tc.from_numpy(data_high_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
        "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
        torch_data_high_resolution = torch_data_high_resolution.permute(0, 1, 3, 2)
        print(np.shape(torch_data_high_resolution))
        torch_data_high_resolution_sequence = torch_data_high_resolution
        print(np.shape(torch_data_high_resolution_sequence))
        torch_data_high_resolution_sequence = torch_data_high_resolution_sequence.float()
        print(np.shape(torch_data_high_resolution_sequence))
        
        return torch_data_high_resolution_sequence


def get_eval_dataloader(root, filename, batch_size, ref=False, shuffle=False):
    """Get Target dataset loader."""
    # dataset and data loader
    lr_dataset = EVAL_LOADER(os.path.join(root, 'Evaluation', filename), load_hr=False, load_lr=True)
    
    if ref:
        ref_dataset = EVAL_LOADER(os.path.join(root, 'Ref_evaluation', filename), load_hr=True, load_lr=False)
        evaluation_dataset = tc.utils.data.TensorDataset(lr_dataset, ref_dataset)
    else:
        evaluation_dataset = tc.utils.data.TensorDataset(lr_dataset)

    evaluation_loader = tc.utils.data.DataLoader(
        dataset=evaluation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle)

    return evaluation_loader
