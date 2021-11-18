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
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        # print(self.data_paths.tail())
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
    from utils.config_files_utils import read_yaml, copy_yaml
    import numpy as np
    import matplotlib.pyplot as plt


    root_dir = "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries"
    paths_files = ["/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries/paths/100_3/train_paths.csv",
                   "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries/paths/100_3/eval_paths.csv"]
    transform = False

    pkl_dataset = SatImDataset(csv_file=paths_files[0], root_dir=root_dir, transform=transform, return_paths=True)

    len(pkl_dataset)
    sample = pkl_dataset[0]

    # IDS = []
    # for paths_file in paths_files:
    #     # paths_file = paths_files[0]

    #     pkl_dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=True)
    #
    #     # sample = pkl_dataset[16582]
    #
    #     for i, sample in enumerate(pkl_dataset):
    #         # if i % 1000 == 0:
    #         #     print(i)
    #         sample, path = sample
    #         ids = sample['ids']
    #         unids = np.unique(ids)
    #
    #         if 64367 in unids:
    #             print('found: ', paths_file, i)
    #             break
    #         IDS.append(unids)
    #
    # IDS = np.concatenate(IDS)
    #
    # plt.figure()
    # plt.imshow(sample['B03'][5])
    #
    #
    # plt.figure()
    # ids = sample['ids']
    # ids[ids == 64367] = 1e9
    # plt.imshow(ids)
    # 64367 in IDS
    #
    # ids = np.unique(np.concatenate(IDS))

    # sample[0].keys()
    # sample[0]['B02'].shape
    #
    # plt.figure()
    # plt.imshow(sample[0]['B02'][5])
    #
    # plt.figure()
    # plt.imshow(sample[0]['labels'])


# def vec_translate(x):
    #     d = {v: i for i, v in enumerate(np.unique(x))}
    #     return np.vectorize(d.__getitem__)(x)
    #
    #
    # DATASET_INFO = read_yaml("data/currently_implemented_datasets.yaml")
    #
    # use_cuda = False  # torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    #
    # savedir = 'results/dataset_info/France'
    # machine = 'ic_server'  # 'calculator'  #
    #
    # configs = ['configs/France/moco_tests/cscl_2e0.yaml', 'configs/France/moco_tests/cscl_2e1.yaml', 'configs/France/moco_tests/base_2e2.yaml',
    #            'configs/France/moco_tests/base_2e3.yaml', 'configs/France/moco_tests/base_2e4.yaml', 'configs/France/moco_tests/base_2e5.yaml',
    #            'configs/France/moco_tests/base_2e6.yaml', 'configs/France/moco_tests/cscl_2e7.yaml']
    #
    # labels_dict = {0: 15,
    #          1: 0, 2: 1,
    #          3: 15, 4: 15, 5: 15,
    #          6: 2, 7: 3, 8: 4,
    #          9: 15,
    #          10: 5, 11: 6, 12: 7,
    #          13: 15,
    #          14: 8,
    #          15: 15,
    #          16: 9,
    #          17: 15, 18: 15, 19: 15, 20: 12, 21: 15, 22: 15, 23: 15, 24: 15,
    #          25: 10,
    #          26: 15, 27: 15, 28: 15, 29: 15, 30: 15,
    #          31: 11,
    #          32: 15, 33: 15, 34: 15, 35: 15, 36: 15, 37: 15, 38: 15, 39: 15, 40: 15, 41: 15, 42: 15,
    #          43: 15, 44: 15, 45: 15, 46: 15, 47: 15, 48: 15, 49: 15, 50: 15, 51: 15, 52: 15, 53: 15,
    #          54: 15, 55: 15, 56: 15, 57: 15, 58: 15,
    #          59: 12,
    #          60: 15, 61: 15, 62: 15, 63: 15, 64: 15, 65: 15, 66: 15, 67: 15, 68: 15, 69: 15, 70: 15, 71: 15,
    #          72: 15, 73: 15, 74: 15, 75: 15, 76: 15, 77: 15, 78: 15, 79: 15, 80: 15, 81: 15, 82: 15,
    #          83: 15, 84: 15, 85: 15, 86: 15, 87: 15, 88: 15, 89: 15, 90: 15, 91: 15, 92: 15, 93: 15,
    #          94: 15, 95: 15, 96: 15, 97: 15, 98: 15, 99: 15, 100: 15, 101: 15, 102: 15, 103: 15,
    #          104: 15, 105: 15, 106: 15, 107: 15, 108: 15, 109: 15, 110: 15, 111: 15, 112: 15, 113: 15,
    #          114: 15, 115: 15, 116: 15, 117: 15, 118: 15, 119: 15, 120: 15, 121: 15, 122: 15, 123: 15,
    #          124: 15, 125: 15, 126: 15, 127: 15, 128: 15, 129: 15, 130: 15, 131: 15, 132: 15, 133: 15,
    #          134: 15, 135: 15, 136: 15, 137: 15, 138: 15, 139: 15, 140: 15, 141: 15, 142: 15, 143: 15,
    #          144: 15, 145: 15, 146: 15, 147: 15, 148: 15, 149: 15, 150: 15, 151: 15, 152: 15, 153: 15,
    #          154: 15, 155: 15, 156: 15, 157: 15, 158: 15, 159: 15, 160: 15, 161: 15, 162: 15, 163: 15,
    #          164: 15, 165: 15, 166: 15, 167: 15
    #          }
    #
    # for config_ in configs:
    #     print(config_)
    #     # config_ = configs[0]
    #     config = read_yaml(config_)
    #
    #     # CURRENTLY_IMPLEMENTED = list(DATASET_INFO[machine].keys())
    #
    #     root_dir = DATASET_INFO[machine][config['DATASETS']['train']['dataset']]['basedir']
    #     csv_file = DATASET_INFO[machine][config['DATASETS']['train']['dataset']]['paths_train']
    #
    #     pkl_dataset = SatImDataset(csv_file=csv_file, root_dir=root_dir, transform=None)
    #
    #     unique_labels = []
    #     unique_counts = []
    #     for i, sample in enumerate(pkl_dataset):
    #         if i % 1000 == 0:
    #             print(i)
    #         # sample, path = sample
    #         # year = path[-11:-7]
    #         unlabels, uncounts = np.unique(sample['labels'], return_counts=True)
    #         unlabels = np.array([labels_dict[c] for c in unlabels])
    #         unique_labels.append(unlabels)
    #         unique_counts.append(uncounts)
    #     unique_labels = np.concatenate(unique_labels)
    #     unique_counts = np.concatenate(unique_counts)
    #     labels = pd.DataFrame(np.stack((unique_labels, unique_counts), axis=1), columns=['label', 'counts'])
    #     labgr = labels.groupby(['label']).agg(['count', 'sum']).reset_index()
    #     labgr.columns = ['label', 'count', 'sum']
    #     labgr = labgr.sort_values('sum', ascending=False)
    #     labgr.to_csv(savedir + "/%s_label_counts.csv" % config_.split('/')[-1].split('.')[0], index=False)



    # root_dir = "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries"
    # # paths_file = ["/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/paths/100_3/train_paths.csv",
    # #               "/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/paths/100_3/eval_paths.csv"]
    # savedir = "/home/michaeltrs/Documents/Research/satellite_imagery/github/SatelliteImagery/results/data_stats_france"
    # transform = False
    #
    # # pkl_dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=True)
    #
    # # base_labels = [ 0.,  1.,  2.,  6.,  7.,  8.,  9., 10., 11., 12., 14., 16., 25., 31., 59., 60.]
    # label_names = {1: 'PPH', 2: 'J6S', 3: 'VRC', 6: 'SNE', 7: 'ORH', 8: 'PRL', 9: 'TTH', 10: 'PTC', 11: 'MH7',
    #                12: 'RDF', 13: 'MH6', 14: 'BOP', 20: 'BOR', 25: 'RVI', 33: 'VRG', 34: 'TCR', 35: 'J5M',
    #                36: 'MC7', 48: 'LU6', 73: 'SA7'}
    #
    # for ratio in range(1, 8):
    #     print(ratio)
    #     paths_file = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries/paths/100_3/train_paths_bal_2e%d.csv' % ratio
    #
    #     pkl_dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=True)
    #
    #     # sample = pkl_dataset[0]
    #
    #     labels = []
    #     doy = {'2016': [], '2017': [], '2018': []}
    #     for i, sample in enumerate(pkl_dataset):
    #         if i % 100 == 0:
    #             print(i)
    #         sample, path = sample
    #         year = path[-11:-7]
    #         unlabels, uncounts = np.unique(sample['labels'], return_counts=True)
    #         ldict = {l: c for l, c in zip(unlabels, uncounts)}
    #         for ii in range(len(unlabels)):
    #             if 1.0 in unlabels:
    #                 labels.append([i, unlabels[ii], uncounts[ii]])  # , list(unlabels)==[0.0, 1.0], ldict[1.0]/uncounts.sum()])
    #             else:
    #                 labels.append([i, unlabels[ii], uncounts[ii]])
    #         doy[year].append(sample['doy'])
    #
    #     labels = pd.DataFrame(labels, columns=['id', 'label', 'counts'])
    #
    #     labgr = labels[['label', 'counts']].groupby(['label']).agg(['count', 'sum']).reset_index()
    #     labgr.columns = ['label', 'count', 'sum']
    #     labgr = labgr.sort_values('sum', ascending=False)
    #     labgr.to_csv(savedir + "/label_counts.csv", index=False)
    #
    #     plt.figure()
    #     plt.bar([0], np.log(labgr['sum'].iloc[0]), color='red', alpha=0.5)#, edgecolor='black')
    #     plt.bar(np.arange(1, labgr.shape[0]), np.log(labgr['sum'].iloc[1:]), color='blue', alpha=0.5)#, edgecolor='black')
    #     plt.xlabel('label rank')
    #     plt.ylabel('logN')
    #
    #     labgr20 = labgr[labgr['label'].isin(list(label_names.keys()))]
    #
    #     plt.figure()
    #     plt.bar(np.arange(labgr20.shape[0]), np.log(labgr20['sum']), color='blue',
    #             alpha=0.5)  # , edgecolor='black')
    #     plt.xticks(range(len(label_names)), label_names.values(), rotation='vertical')
    #     plt.xlabel('label name')
    #     plt.ylabel('logN')
    #     plt.tight_layout()
    #
    #     # dates --------------------------------------------------------------------------------------------------------
    #     for key in doy:
    #         doy[key] = np.unique(np.concatenate(doy[key]))
    #
    #     fig, axs = plt.subplots(3, 1, figsize=(12, 6))
    #
    #     axs[0].scatter(doy['2016'], np.zeros(doy['2016'].shape[0]), s=20, c='b')
    #     axs[0].vlines(doy['2016'], -0.25, 1, color='b')
    #     axs[0].set_ylim(0, 1)
    #     axs[0].set_xlim(0, 365)
    #     axs[0].set_yticks([], [])
    #     axs[0].set_ylabel('2016')
    #     axs[1].scatter(doy['2017'], np.zeros(doy['2017'].shape[0]), s=20, c='b')
    #     axs[1].vlines(doy['2017'], -0.25, 1, color='b')
    #     axs[1].set_ylim(0, 1)
    #     axs[1].set_xlim(0, 365)
    #     axs[1].set_yticks([], [])
    #     axs[1].set_ylabel('2017')
    #     axs[2].scatter(doy['2018'], np.zeros(doy['2018'].shape[0]), s=20, c='b')
    #     axs[2].vlines(doy['2018'], -0.25, 1, color='b')
    #     axs[2].set_ylim(0, 1)
    #     axs[2].set_xlim(0, 365)
    #     axs[2].set_yticks([], [])
    #     axs[2].set_ylabel('2018')
    #     axs[2].set_xlabel('day of year')
    #
    #
    #     # plt.legend(bbox_to_anchor=(1.3, 1))
    #
    #
    #
    #
    #
    #     labels_ = labels_[labels_['label'].isin(base_labels)]
    #
    #     cnames = pd.read_csv("/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/France/classnames.txt", encoding='iso-8859-1')
    #
    #
    #
    #
    #




    # for step, sample in enumerate(pkl_dataset):
    #     print(step)
    #     sample = pkl_dataset[0]
    #     sample, path = sample
    #
    #     labels = sample['labels']
    #
    #     # if np.unique(labels).shape[0] < 6:
    #     #     continue
    #     # plt.figure()
    #     # plt.imshow(sample['B02'][0])
    #
    #     plt.ioff()
    #     fig, axs = plt.subplots(6, 6, figsize=(100, 100))
    #     for i, key in enumerate(['B02', 'B03', 'B04', 'B08']):
    #         for j in range(6):
    #             axs[i, j].imshow(sample[key][j+5])
    #             axs[i, j].axis('off')
    #     # for i, key in enumerate(['labels', 'ids']):
    #
    #     for jj, id_ in enumerate(np.unique(labels)):
    #         labels[labels == id_] = jj
    #     axs[4, 0].imshow(labels)
    #     axs[4, 0].axis('off')
    #     ids = sample['ids']
    #     # ids[ids==0] = ids.max() + 1
    #     # ids = ids - ids.min()
    #     for jj, id_ in enumerate(np.unique(ids)):
    #         ids[ids == id_] = jj
    #     ids = np.interp(ids, (ids.min(), ids.max()), (0, +1))
    #     axs[5, 0].imshow(ids)
    #     axs[5, 0].axis('off')
    #     plt.savefig(savedir + "/%d.png" % (step))
    #     plt.close(fig)
    #
    #
    #
    # savedir = "/home/michaeltrs/Documents/Research/satellite_imagery/github/SatelliteImagery/results/iccv_plots/samples/keep"
    #
    # step = 156
    # sample, path = pkl_dataset[step]
    #
    # labels = sample['labels']
    # ids = sample['ids']
    #
    # plt.ioff()
    # fig, axs = plt.subplots(6, 18, figsize=(300, 100))
    # for i, key in enumerate(['B02', 'B03', 'B04', 'B08']):
    #     for j in range(6):
    #         axs[i, j].imshow(sample[key][j + 5])
    #         axs[i, j].axis('off')
    # # for i, key in enumerate(['labels', 'ids']):
    #
    # for jj, id_ in enumerate(np.unique(labels)):
    #     labels[labels == id_] = jj
    # axs[4, 0].imshow(labels)
    # axs[4, 0].axis('off')
    # ids = sample['ids']
    # # ids[ids==0] = ids.max() + 1
    # # ids = ids - ids.min()
    # for jj, id_ in enumerate(np.unique(ids)):
    #     ids[ids == id_] = jj
    # ids = np.interp(ids, (ids.min(), ids.max()), (0, +1))
    # axs[5, 0].imshow(ids)
    # axs[5, 0].axis('off')
    # plt.savefig(savedir + "/%d.png" % (step))
    # plt.close(fig)
    #
    #
    #
    # def get_edge_labels(labels, nb_size, axes=[0, 1]):
    #     pad_size = nb_size // 2
    #     stride = 1
    #     lto = labels.to(torch.float32)
    #     H = lto.shape[axes[0]]
    #     W = lto.shape[axes[1]]
    #     lto = torch.nn.functional.pad(
    #         lto.unsqueeze(0).unsqueeze(0), [pad_size, pad_size, pad_size, pad_size], 'reflect')[
    #         0, 0]
    #     patches = lto.unfold(axes[0], nb_size, stride).unfold(axes[1], nb_size, stride)
    #     patches = patches.reshape(-1, nb_size ** 2)
    #     edge_map = (patches != patches[:, 0].unsqueeze(-1).repeat(1, nb_size ** 2)).any(dim=1).reshape(W, H).to(
    #         torch.bool)
    #     return edge_map
    #
    #
    #
    #
    #
    # im0 = sample['B08'][5]
    #
    # edges = get_edge_labels(torch.tensor(labels), nb_size=9, axes=[0, 1]).numpy()
    # edges2 = get_edge_labels(torch.tensor(labels), nb_size=5, axes=[0, 1]).numpy()
    #
    # plt.figure()
    # plt.imshow(im0)
    # plt.show()
    #
    # unids = np.unique(ids).copy()
    # for jj, id_ in enumerate(unids):
    #     ids[ids == id_] = jj
    #
    # unlabels = np.unique(labels).copy()
    # for jj, id_ in enumerate(unlabels):
    #     labels[labels == id_] = jj
    #
    #
    # plt.figure()
    # plt.imshow(ids)
    # plt.title('ids')
    #
    # plt.figure()
    # plt.imshow(labels)
    # plt.title('labels')
    #
    # # plt.figure()
    # # for id_ in np.unique(ids):
    # #     plt.hist(np.reshape(im0[ids == id_], (-1,)), label=id_, alpha=0.3)
    # # plt.legend()
    # # plt.show()
    #
    # fig, axs = plt.subplots(1, 2, figsize=(100, 50), gridspec_kw={'width_ratios': [1, 2]})
    # axs[0].imshow(im0/im0.max())
    # axs[0].imshow(((labels == lab_) & (ids == id_)).astype(np.float32), alpha=0.5)
    # axs[0].axis('off')
    # axs[1].hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=7, label='all', alpha=0.4, density=True)
    # axs[1].hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=7, label='int', alpha=0.4,
    #          density=True)
    # axs[1].set_xlim([2500, 5500])
    # # axs[1].legend()
    #
    # plt.ioff()
    #
    # plt.figure(figsize=(24, 24))
    # plt.imshow(im0 / im0.max())
    # plt.axis('off')
    # plt.savefig(savedir + "/image.png" , bbox_inches='tight')
    #
    # plt.figure(figsize=(24, 24))
    # plt.imshow(labels)
    # plt.axis('off')
    # plt.savefig(savedir + "/labels.png", bbox_inches='tight')
    #
    # lab_, id_ = 2, 8
    #
    # for lab_, id_ in [[2, 8], [2, 5], [4, 1], [3, 3], [3, 4]]:
    #     plt.figure(figsize=(24, 12))
    #     plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=15, label='all', alpha=0.4, density=True)
    #     plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=15, label='int', alpha=0.4, density=True)
    #     plt.xlim([2500, 5500])
    #     plt.yticks([], [])
    #     plt.savefig(savedir + "/%d_%d_hist.png" % (lab_, id_), bbox_inches='tight')
    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(im0/im0.max())
    #     plt.imshow(((labels == lab_) & (ids == id_)).astype(np.float32), alpha=0.5, cmap='gray')
    #     plt.imshow(((labels == lab_) & (ids == id_) & ~edges2).astype(np.float32), alpha=0.5, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(savedir + "/%d_%d_im.png" % (lab_, id_), bbox_inches='tight')
    #
    #
    #
    # lab_, id_ = 2, 5
    # plt.figure()
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=10, label='all', alpha=0.4, density=True)
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=10, label='int', alpha=0.4,
    #          density=True)
    # plt.legend()
    #
    # lab_, id_ = 4, 1
    # plt.figure()
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=10, label='all', alpha=0.4, density=True)
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=10, label='int', alpha=0.4,
    #          density=True)
    # plt.legend()
    #
    # lab_, id_ = 3, 3
    # plt.figure()
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=10, label='all', alpha=0.4, density=True)
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=10, label='int', alpha=0.4,
    #          density=True)
    # plt.legend()
    #
    # lab_, id_ = 3, 4
    # plt.figure()
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_)], (-1,)), bins=10, label='all', alpha=0.4, density=True)
    # plt.hist(np.reshape(im0[(labels == lab_) & (ids == id_) & ~edges], (-1,)), bins=10, label='int', alpha=0.4,
    #          density=True)
    # plt.legend()
    #
    #
    # for lab_ in [2, 3, 4]:  # np.unique(labels):
    #     plt.hist(np.reshape(im0[labels == lab_], (-1,)), label=lab_, alpha=0.4, density=True)
    # plt.legend()
    # plt.show()
    #
    # for lab_ in np.unique(labels):
    #     plt.figure()
    #     plt.imshow(im0)
    #     plt.imshow((labels == lab_).astype(np.float32), alpha=0.7)
    #     plt.title(lab_)
    # plt.show()







    # base_labels = [ 0.,  1.,  2.,  6.,  7.,  8.,  9., 10., 11., 12., 14., 16., 25., 31., 59., 60.]
    #
    # for ratio in range(1, 8):
    #     print(ratio)
    #     paths_file = '/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries/paths/100_3/train_paths_2e%d.csv' % ratio
    #
    #     pkl_dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=True)
    #
    #     # sample = pkl_dataset[0]
    #
    #     labels = []
    #     paths = []
    #     for i, sample in enumerate(pkl_dataset):
    #         sample, path = sample
    #         unlabels, uncounts = np.unique(sample['labels'], return_counts=True)
    #         ldict = {l: c for l, c in zip(unlabels, uncounts)}
    #         for ii in range(len(unlabels)):
    #             if 1.0 in unlabels:
    #                 labels.append([i, unlabels[ii], uncounts[ii], list(unlabels)==[0.0, 1.0], ldict[1.0]/uncounts.sum()])
    #             else:
    #                 labels.append([i, unlabels[ii], uncounts[ii], False, 0.0])
    #         paths.append(path)
    #     labels = pd.DataFrame(labels, columns=['id', 'label', 'counts', '01only', '1ratio'])
    #
    #     labels_ = labels[~labels['01only']]
    #     labels_ = labels_[labels_['1ratio'] < 0.5]
    #     #
    #     # print(labels_['id'].drop_duplicates().shape)
    #     labels_ = labels_[labels_['label'].isin(base_labels)]
    #
    #     paths_ = pd.DataFrame(np.array(paths)[labels_['id'].drop_duplicates().values])
    #     paths_ = pd.DataFrame(paths_[0].apply(lambda s: '/'.join(s.split("/")[-2:])))
    #     paths_.to_csv('/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2018/pkl_timeseries/paths/100_3/eval_paths_bal.csv', header=None, index=False)

    # gr = labels_[['label', 'counts']].groupby(['label']).agg(['count', 'sum'])
    #
    # gr = gr[gr['counts']['count'] > 10].reset_index()
    # gr['label'].values [ 0.,  1.,  2.,  6.,  7.,  8.,  9., 10., 11., 12., 14., 16., 25., 31., 59., 60.]

    # sample.keys()
    #
    # t = 9
    # savedir = "/home/michaeltrs/Documents/Research/satellite_imagery/github/SatelliteImagery/results/sample_example_plots"
    #
    # plt.ioff()
    # for key in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']:
    #     # print(key, sample[key][0].shape[0])
    #     # key = 'B02'
    #     s = 6 * sample[key][0].shape[0] / 48.
    #     print(key, s)
    #     fig, ax = plt.subplots(figsize=(s, s))
    #     ax.set_aspect('equal')
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #     ax.imshow(sample[key][t])
    #     ax.axis('tight')
    #     ax.axis('off')
    #     fig.savefig(os.path.join(savedir, "%s.png" % key))
    #
    # for key in ['groups', 'labels', 'ratios', 'ids']:
    #     # print(key, sample[key][0].shape[0])
    #     # key = 'B02'
    #     s = 6
    #     print(key, s)
    #     fig, ax = plt.subplots(figsize=(s, s))
    #     ax.set_aspect('equal')
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #     ax.imshow(vec_translate(sample[key]))
    #     ax.axis('tight')
    #     ax.axis('off')
    #     fig.savefig(os.path.join(savedir, "%s.png" % key))
    #
    # for key in ['full_ass', 'part_ass']:
    #     # print(key, sample[key][0].shape[0])
    #     # key = 'B02'
    #     s = 6
    #     print(key, s)
    #     fig, ax = plt.subplots(figsize=(s, s))
    #     ax.set_aspect('equal')
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #     ax.imshow(vec_translate(sample[key]))
    #     ax.axis('tight')
    #     ax.axis('off')
    #     fig.savefig(os.path.join(savedir, "%s.png" % key))
    #
    # for key in ['groups_x4', 'labels_x4', 'ratios_x4', 'ids_x4']:
    #     # print(key, sample[key][0].shape[0])
    #     # key = 'B02'
    #     s = 6
    #     print(key, s)
    #     fig, ax = plt.subplots(figsize=(s, s))
    #     ax.set_aspect('equal')
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #     ax.imshow(vec_translate(sample[key]))
    #     ax.axis('tight')
    #     ax.axis('off')
    #     fig.savefig(os.path.join(savedir, "%s.png" % key))
    #
    # for key in ['part_ass_x4']:
    #     # print(key, sample[key][0].shape[0])
    #     # key = 'B02'
    #     s = 6
    #     print(key, s)
    #     fig, ax = plt.subplots(figsize=(s, s))
    #     ax.set_aspect('equal')
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #     ax.imshow(vec_translate(sample[key]))
    #     ax.axis('tight')
    #     ax.axis('off')
    #     fig.savefig(os.path.join(savedir, "%s.png" % key))
    #     # plt.figure() # size=(4, 4))
    #     # plt.imshow(sample[key][t])
    #     # plt.axis('off')
    #     # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #     #                     hspace=0, wspace=0)
    #     # plt.savefig(os.path.join(savedir, "%s.png" % key))
    #
    #
    #
    # plt.figure()
    # plt.imshow(sample['B02'][9])
    #
    #
    #
    # l = vec_translate(sample['labels_x4'])
    # plt.figure()
    # plt.imshow(l)
    # plt.colorbar()
    #
    # ids = vec_translate(sample['ids_x4'])
    # plt.figure()
    # plt.imshow(ids)
    # plt.colorbar()
    #
    # # import png



    # paths = pd.read_csv(paths_file, header=None)
    #
    # idx = []
    # for i in range(paths.shape[0]):
    #     print(i)
    #     sample = pkl_dataset[i]
    #     idx.append(np.unique(sample['ids'][np.vectorize(remap_label_dict['labels_44'].get)(sample['labels']) != 20]))
    # ids = np.unique(np.concatenate(idx))
    # np.savetxt("/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2017/100_3_train_parcel_ids.csv", ids)
    #     if [sample[key].shape[1:] for key in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09',
    #                                          'B10', 'B11', 'B12']] + \
    #         [sample[key].shape for key in ['labels', 'mask']] == [(8, 8), (48, 48), (48, 48),
    #                                                                    (48, 48), (24, 24),
    #                                                                    (24, 24), (24, 24), (48, 48), (24, 24), (8, 8),
    #                                                                    (8, 8), (24, 24), (24, 24), (48, 48), (48, 48)]:
    #         idx.append(i)
    #
    # paths2 = pd.DataFrame(paths.values[idx])
    # paths2.to_csv("/media/michaeltrs/0a8a5a48-ede5-47d0-8eff-10d11350bf98/Satellite_Data/Sentinel2/PSETAE_repl/2017/classification_set/pkl_timeseries/data_paths.csv", header=None, index=False)

    # dataloader = DataLoader(pkl_dataset, batch_size=50, shuffle=False, num_workers=4)#,
    #
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
