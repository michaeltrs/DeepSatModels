# MTLCC_prev dataset
#   eval: [[2, 2], [3, 197], [4, 21], [5, 23], [6, 3], [7, 11], [8, 30],
#          [9, 74], [10, 19], [11, 4704], [12, 18], [13, 17], [14, 30], [15, 11510]]
from __future__ import print_function, division
import os
import torch
import pandas as pd
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
# from data.MTLCC.data_transforms import MTLCC_transform
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_paths = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name
        
        return sample

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample
    
    
def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    import numpy as np


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #paths_file = "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/MTLCC_demo/MTLCC/data_IJGI18/datasets/full/240pkl/train_paths.csv"
    #root_dir = "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/MTLCC_demo/MTLCC"

    paths_file = "/media/michaeltrs/184c4903-8b24-4311-a391-84e554e0d240/SatelliteImagery/MTLCC/pkl_full/paths/240/eval_pkl_24x24_paths.csv"
    root_dir = "/media/michaeltrs/184c4903-8b24-4311-a391-84e554e0d240/SatelliteImagery/MTLCC/pkl_full"
    
    transform = None  # MTLCC_transform(img_res=24, max_seq_len=51, n_class=18, is_training=False)
    
    pkl_dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform)
    
    paths = pd.read_csv(paths_file, header=None)

    # i = 24989  # 0
    # sample = pkl_dataset[i]  # 33596]
    #
    # print(sample.keys())
    # print(paths.iloc[i].values[0])
    # for key in sample.keys():
    #     print(key)
    #     print("min: ", sample[key].min(), "max: ", sample[key].max())
    #     print("-----------------------------------------------------------")
    #
    # sample['x10'].shape
    #
    # plt.figure()
    # plt.imshow(sample['x10'][5, :, :, 0])
    #
    Nt = []
    for i in range(len(paths)):
        print(i)
        sample = pkl_dataset[i]

        Nt.append([sample['x10'].min(), sample['x10'].max(), sample['x20'].min(), sample['x20'].max(),
                   sample['x60'].min(), sample['x60'].max(), sample['day'].min(), sample['day'].max(),
                   sample['labels'].min(), sample['labels'].max()])

    out = np.array(Nt)

    # out[:5]
    oe = pd.DataFrame(out)
    oe.drop_duplicates().shape
    #o.shape

    s = oe.values[1]
    for s in oe.values:
        print(o[(o[0] == s[0]) & (o[1] == s[1]) & (o[2] == s[2]) & (o[3] == s[3]) & (o[4] == s[4])].shape)

    # o = out[(out[:, 0]==347) & (out[:, 1]==10110)]
    # Nt = np.array(Nt)
    # print(Nt.min(), Nt.max())
    # # dataloader = DataLoader(pkl_dataset, batch_size=10, shuffle=False, num_workers=2)#,
    # dataiter = iter(dataloader)
    #
    # sample = next(dataiter)
    # for sample in dataiter:
    
    #a = next(dataiter)
    #print(a["seq_lengths"])
    #p = pkl_dataset.data_paths
    # sample = pkl_dataset[0]
    #
    # for i in range(10):
    #
    #     plt.figure()
    #
    #     inp = sample['inputs'][i, :, :, :3].numpy()
    #     scfact = inp.max()
    #     inp = inp / scfact
    #
    #     plt.imshow(inp)
