""" The dataloader in this file is used to get the LR and HR which belong to the different tissues (i.e. they are unpaired LR and HR), for training and validation purpose. """

# import gzip
import os
# import pickle
# import urllib

import numpy as np
import torch as tc
# import torch.utils.data as data
# from torchvision import datasets, transforms
import h5py


def GET_LOADER(root, load_lr=True, load_hr=True):
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

#     def __init__(self, root):
    """Init Source dataset."""
    # init params
#    root = os.path.expanduser(root)
    # self.forward()
#     def forward(self):
    file_names = sorted(os.listdir(root))
    num_low_resolution_mat_file = 0
    num_high_resolution_groundtruth_mat_file = 0
    for idx_file in file_names:
        print(idx_file)
        if '.mat' in os.path.join(root, idx_file):
            print('One more low resolution image set exist')
            num_low_resolution_mat_file = num_low_resolution_mat_file + 1
            print(os.path.join(root, idx_file))
            file_data = h5py.File(os.path.join(root, idx_file), 'r')
            if load_lr:
                data_low_resolution = file_data['LR'][:] #----- numpy array
                print('data_low_resolution', np.shape(data_low_resolution))
                print(data_low_resolution.dtype)
                torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
    # =============================================================================
    #         torch_data_low_resolution = tc.t(torch_data_low_resolution)
    # =============================================================================
    # =============================================================================
    #         torch_data_low_resolution = torch_data_low_resolution.type(tc.DoubleTensor)
    # =============================================================================
                print(np.shape(torch_data_low_resolution))
                if num_low_resolution_mat_file == 1:
                    torch_data_low_resolution_sequence = torch_data_low_resolution
                elif num_low_resolution_mat_file > 1:
                    print(num_low_resolution_mat_file)
                    torch_data_low_resolution_sequence = tc.cat((torch_data_low_resolution_sequence, torch_data_low_resolution), 0)
                print('torch_data_low_resolution_sequence', np.shape(torch_data_low_resolution_sequence))
            if load_hr:
                num_high_resolution_groundtruth_mat_file = num_high_resolution_groundtruth_mat_file + 1
                print(os.path.join(root, idx_file))
                data_high_resolution_groundtruth = file_data['HRGT'][:] #----- numpy array
                print(np.shape(data_high_resolution_groundtruth))
                torch_data_high_resolution_groundtruth = tc.from_numpy(data_high_resolution_groundtruth) #----- torch type data could be read by tc.utils.data.TensorDataset
                "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
                torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.permute(0, 1, 3, 2)
    # =============================================================================
    #         torch_data_high_resolution_groundtruth = tc.t(torch_data_high_resolution_groundtruth)
    # =============================================================================
    # =============================================================================
    #         torch_data_high_resolution_groundtruth = torch_data_high_resolution_groundtruth.type(tc.DoubleTensor)
    # =============================================================================
                print(np.shape(torch_data_high_resolution_groundtruth))
                if num_high_resolution_groundtruth_mat_file == 1:
                    torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth
                elif num_high_resolution_groundtruth_mat_file > 1:
                    print(num_high_resolution_groundtruth_mat_file)
                    torch_data_high_resolution_groundtruth_sequence = tc.cat((torch_data_high_resolution_groundtruth_sequence, torch_data_high_resolution_groundtruth), 0)
                print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    if load_lr:
        torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
        print(np.shape(torch_data_low_resolution_sequence))
    if load_hr:
        torch_data_high_resolution_groundtruth_sequence = torch_data_high_resolution_groundtruth_sequence.float()
        print(np.shape(torch_data_high_resolution_groundtruth_sequence))
    if load_lr and load_hr:
        return tc.utils.data.TensorDataset(torch_data_low_resolution_sequence, torch_data_high_resolution_groundtruth_sequence)
    elif load_lr:
        return tc.utils.data.TensorDataset(torch_data_low_resolution_sequence)
    elif load_hr:
        return tc.utils.data.TensorDataset(torch_data_high_resolution_groundtruth_sequence)

def get_dataloader(source_root, target_root, batch_size, shuffel=True):
    """Get Target dataset loader."""
    # dataset and data loader
    source_training_dataset = GET_LOADER(os.path.join(source_root, 'Source'), load_hr=True, load_lr=False)
    target_training_dataset = GET_LOADER(os.path.join(target_root, 'Target'), load_hr=False, load_lr=True)
    validation_dataset = GET_LOADER(os.path.join(target_root, 'Validation'), load_hr=True, load_lr=True)
    # evaluation_dataset = GET_LOADER(os.path.join(root, 'Evaluation'), load_hr=False, load_lr=True)

    source_training_loader = tc.utils.data.DataLoader(
        dataset=source_training_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffel,
        drop_last=True)

    target_training_loader = tc.utils.data.DataLoader(
        dataset=target_training_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffel,
        drop_last=True)
    
    validation_loader = tc.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)
    """
    evaluation_loader = tc.utils.data.DataLoader(
        dataset=evaluation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)
    """    
    return source_training_loader, target_training_loader, validation_loader
