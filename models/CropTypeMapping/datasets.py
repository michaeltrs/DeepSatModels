"""

File that houses the dataset wrappers we have.

"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import h5py
import numpy as np
import os

from skimage.transform import resize as imresize

import preprocess
from constants import *
from random import shuffle
from pprint import pprint

import matplotlib.pyplot as plt
from collections import defaultdict


def get_Xy(dl, country):
    """ 
    Constructs data (X) and labels (y) for pixel-based methods. 
    Args: 
      dl - pytorch data loader
    Returns: 
      X - matrix of data of shape [examples, features] 
      y - vector of labels of shape [examples,] 
    """
    # Populate data and labels of classes we care about
    X = []
    y = []
    num_samples = 0
    for inputs, targets, cloudmasks, hres_inputs in dl:
        if len(hres_inputs.shape) > 1:
            raise ValueError('Planet inputs must be resized to the grid size')
        X, y = get_Xy_batch(inputs, targets, X, y, country)
        if len(y) > 0:
            num_samples += y[-1].shape[0]
        if num_samples > 100000:
            break

    X = np.vstack(X)
    y = np.squeeze(np.vstack(y))    

    # shuffle
    indices = np.array(list(range(y.shape[0])))
    indices = np.random.shuffle(indices)
    X = np.squeeze(X[indices, :])
    y = np.squeeze(y[indices])

    print('X shape input: ', X.shape) 
    print('y shape input: ', y.shape)
    return X, y

def get_Xy_batch(inputs, targets, X, y, country):
    """ 
    Constructs necessary pixel array for pixel based methods. The 
    function takes one case of inputs, targets each time it is called
    and builds up X, y as the dataloader goes through all batches 
    """
    # For each input example and corresponding target,
    for ex_idx in range(inputs.shape[0]):
        for crop_idx in range(targets.shape[1]):
            cur_inputs = np.transpose(np.reshape(inputs[ex_idx, :, :, :, :], (-1, GRID_SIZE[country]*GRID_SIZE[country])), (1, 0))
            cur_targets = np.squeeze(np.reshape(targets[ex_idx, crop_idx, :, :], (-1, GRID_SIZE[country]*GRID_SIZE[country])))
            # Index pixels of desired crop
            valid_inputs = cur_inputs[cur_targets == 1, :]
            if valid_inputs.shape[0] == 0:
                pass
            else:
                # Append valid examples to X
                X.append(valid_inputs)
                # Append valid labels to y
                labels = torch.ones((int(torch.sum(cur_targets).numpy()), 1)) * crop_idx
                y.append(labels)
    return X, y 

def split_and_aggregate(arr, doys, ndays, reduction='avg'):
    """
    Aggregates an array along the time dimension, grouping by every ndays
    
    Args: 
      arr - array of images of dimensions
      doys - vector / list of days of year associated with images stored in arr
      ndays - number of days to aggregate together
    """
    total_days = 364
    
    # Get index of observations corresponding to time bins
    # // ndays gives index from 0 to total_days // ndays
    obs_idxs = list(doys.astype(int) // ndays)
    split_idxs = []
    # Get the locations within the obs_idxs vector that change
    # to a new time bin (will be used for splitting)
    for idx in range(1, int(total_days//ndays)):
        if idx in obs_idxs:
            split_idxs.append(obs_idxs.index(idx))
        # if no observations are in bin 1, append the 0 index
        # to indicate that the bin is empty
        elif idx == 1:
            split_idxs.append(0)
        # if observations are not in the bin, use the same index 
        #  as the previous bin, to indicate the bin is empty
        else:
            split_idxs.append(prev_split_idx)
        prev_split_idx = split_idxs[-1]
        
    # split the array according to the indices of the time bins
    split_arr = np.split(arr, split_idxs, axis=3)
 
    # For each bin, create a composite according to a "reduction"
    #  and append for concatenation into a new reduced array
    composites = []
    for a in split_arr:
        if a.shape[3] == 0:
            a = np.zeros((a.shape[0], a.shape[1], a.shape[2], 1))
        if reduction == 'avg':
            cur_agg = np.mean(a, axis=3)
        elif reduction == 'min':
            cur_agg = np.min(a, axis=3)
        elif reduction == 'max':
            cur_agg = np.max(a, axis=3)
        elif reduction == 'median':
            cur_agg = np.median(a, axis=3)
        composites.append(np.expand_dims(cur_agg, axis=3))

    new_arr = np.concatenate(composites, axis=3)
    new_doys = np.asarray(list(range(0, total_days-ndays+1, ndays)))
    return new_arr, new_doys

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path, split):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = HDF5_PATH[args.country]

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
        
        # Rose debugging line to ignore missing S2 files for Tanzania
        #for my_item in ['004125', '004070', '003356', '004324', '004320', '004322', '003706', '004126', '003701', '003700', '003911', '003716', '004323', '004128', '003485', '004365', '004321', '003910', '004129', '003704', '003486', '003488', '003936', '003823']:
        #    if my_item in self.grid_list:
        #        self.grid_list.remove(my_item)

        self.country = args.country
        self.num_grids = len(self.grid_list)
        self.grid_size = GRID_SIZE[args.country]
        self.agg_days = args.agg_days
        # s1 args
        self.use_s1 = args.use_s1
        self.s1_agg = args.s1_agg
        # s2 args 
        self.use_s2 = args.use_s2
        self.s2_agg = args.s2_agg
        # planet args
        self.resize_planet = args.resize_planet
        self.use_planet = args.use_planet
        self.planet_agg = args.planet_agg
        
        self.num_classes = NUM_CLASSES[args.country]
        self.split = split
        self.apply_transforms = args.apply_transforms
        self.normalize = args.normalize
        self.sample_w_clouds = args.sample_w_clouds
        self.include_clouds = args.include_clouds
        self.include_doy = args.include_doy
        self.include_indices = args.include_indices
        self.num_timesteps = args.num_timesteps
        self.all_samples = args.all_samples
        self.var_length = args.var_length
        ## Timeslice for FCN
        self.timeslice = args.time_slice
        self.least_cloudy = args.least_cloudy
        self.s2_num_bands = args.s2_num_bands
        
        with h5py.File(self.hdf5_filepath, 'r') as data:
            self.combined_lengths = []
            for grid in self.grid_list:
                total_len = 0
                if self.use_s1:
                    total_len += data['s1_length'][grid][()]
                if self.use_s2:
                    total_len += data['s2_length'][grid][()]
                if self.use_planet:
                    total_len += data['planet_length'][grid][()]
                self.combined_lengths.append(total_len)                    

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as data:
            sat_properties = { 's1': {'data': None, 'doy': None, 'use': self.use_s1, 'agg': self.s1_agg,
                                      'agg_reduction': 'avg', 'cloudmasks': None },
                               's2': {'data': None, 'doy': None, 'use': self.use_s2, 'agg': self.s2_agg,
                                      'agg_reduction': 'min', 'cloudmasks': None, 'num_bands': self.s2_num_bands },
                               'planet': {'data': None, 'doy': None, 'use': self.use_planet, 'agg': self.planet_agg,
                                          'agg_reduction': 'median', 'cloudmasks': None, 'num_bands': PLANET_NUM_BANDS } }

            for sat in ['s1', 's2', 'planet']:
                sat_properties = self.setup_data(data, idx, sat, sat_properties)
 
            transform = self.apply_transforms and np.random.random() < .5 and self.split == 'train'
            rot = np.random.randint(0, 4)

            label = data['labels'][self.grid_list[idx]][()]
            label = preprocess.preprocess_label(label, self.model_name, self.num_classes, transform, rot) 
        
            if not self.var_length:
                grid, highres_grid = preprocess.concat_s1_s2_planet(sat_properties['s1']['data'],
                                                      sat_properties['s2']['data'], 
                                                      sat_properties['planet']['data'], self.resize_planet)
                grid = preprocess.preprocess_grid(grid, self.model_name, self.timeslice, transform, rot)
                if highres_grid is not None: 
                    highres_grid = preprocess.preprocess_grid(highres_grid, self.model_name, self.timeslice, transform, rot)           
            else:
                inputs = {}
                if self.use_s1:
                    s1 = preprocess.preprocess_grid(sat_properties['s1']['data'], self.model_name, self.timeslice, transform, rot)
                    inputs['s1'] = s1
                if self.use_s2:
                    s2 = preprocess.preprocess_grid(sat_properties['s2']['data'], self.model_name, self.timeslice, transform, rot)
                    inputs['s2'] = s2
                if self.use_planet:
                    planet = preprocess.preprocess_grid(sat_properties['planet']['data'], self.model_name, self.timeslice, transform, rot)
                    inputs['planet'] = planet
                highres_grid = None      
          
        if sat_properties['s2']['cloudmasks'] is None:
            cloudmasks = False
        else:
            cloudmasks = sat_properties['s2']['cloudmasks']
        if highres_grid is None:
            highres_grid = False

        if self.var_length:
            return inputs, label, cloudmasks, False
        else:
            return grid, label, cloudmasks, highres_grid
    
    def setup_planet(self, data, sat, sat_properties): 
        sat_properties[sat]['data'] = sat_properties[sat]['data'][:, :, :, :].astype(np.double)  
        if self.resize_planet:
            sat_properties[sat]['data'] = imresize(sat_properties[sat]['data'], 
                                                  (sat_properties[sat]['data'].shape[0], self.grid_size, self.grid_size, sat_properties[sat]['data'].shape[3]), 
                                                   anti_aliasing=True, mode='reflect')
                
    def setup_s2(self, data, idx, sat, sat_properties):
        if sat_properties[sat]['num_bands'] == 4:
            sat_properties[sat]['data'] = sat_properties[sat]['data'][[BANDS[sat]['10']['BLUE'], 
                                                                       BANDS[sat]['10']['GREEN'], 
                                                                       BANDS[sat]['10']['RED'],
                                                                       BANDS[sat]['10']['NIR']], 
                                                                       :, :, :] #B, G, R, NIR
        elif sat_properties[sat]['num_bands'] == 10:
            sat_properties[sat]['data'] = sat_properties[sat]['data'][:10, :, :, :]
        elif sat_properties[sat]['num_bands'] != 10:
            raise ValueError('s2_num_bands must be 4 or 10')

        if self.include_clouds:
            sat_properties[sat]['cloudmasks'] = data['cloudmasks'][self.grid_list[idx]][()]
    
    def setup_data(self, data, idx, sat, sat_properties):
        if sat_properties[sat]['use']:
            sat_properties[sat]['data'] = data[sat][self.grid_list[idx]] 
            
            if sat in ['planet']:
                self.setup_planet(data, sat, sat_properties)
            if sat in ['s2']:
                self.setup_s2(data, idx, sat, sat_properties)
            if self.include_doy:
                sat_properties[sat]['doy'] = data[f'{sat}_dates'][self.grid_list[idx]][()]
            if sat_properties[sat]['agg']:
                sat_properties[sat]['data'], sat_properties[sat]['doy'] = split_and_aggregate(sat_properties[sat]['data'], 
                                                                                          sat_properties[sat]['doy'],
                                                                                          self.agg_days, 
                                                                                          reduction=sat_properties[sat]['agg_reduction'])
                
                # Replace the VH/VV band with a cleaner band after aggregation??
                if sat in ['s1']:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sat_properties[sat]['data'][BANDS[sat]['RATIO'],:,:,:] = sat_properties[sat]['data'][BANDS[sat]['VH'],:,:,:] / sat_properties[sat]['data'][BANDS[sat]['VV'],:,:,:]
                        sat_properties[sat]['data'][BANDS[sat]['RATIO'],:,:,:][sat_properties[sat]['data'][BANDS[sat]['VV'],:,:,:] == 0] = 0
            
            else:
                sat_properties[sat]['data'], sat_properties[sat]['doy'], sat_properties[sat]['cloudmasks'] = preprocess.sample_timeseries(sat_properties[sat]['data'],
                                                                                                               self.num_timesteps, sat_properties[sat]['doy'],
                                                                                                               cloud_stack = sat_properties[sat]['cloudmasks'],
                                                                                                               least_cloudy=self.least_cloudy,
                                                                                                               sample_w_clouds=self.sample_w_clouds, 
                                                                                                               all_samples=self.all_samples)

            if sat in ['planet'] and self.resize_planet:
                sat_properties[sat]['data'] = imresize(sat_properties[sat]['data'], 
                                                       (sat_properties[sat]['data'].shape[0], self.grid_size, self.grid_size, sat_properties[sat]['data'].shape[3]), 
                                                       anti_aliasing=True, mode='reflect')

            # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection but AFTER AGGREGATION
            if self.include_indices and sat in ['planet', 's2']:
                with np.errstate(divide='ignore', invalid='ignore'):
                    numbands = str(sat_properties[sat]['num_bands'])
                    ndvi = (sat_properties[sat]['data'][BANDS[sat][numbands]['NIR'], :, :, :] - sat_properties[sat]['data'][BANDS[sat][numbands]['RED'], :, :, :]) / (sat_properties[sat]['data'][BANDS[sat][numbands]['NIR'], :, :, :] + sat_properties[sat]['data'][BANDS[sat][numbands]['RED'], :, :, :])
                    gcvi = (sat_properties[sat]['data'][BANDS[sat][numbands]['NIR'], :, :, :] / sat_properties[sat]['data'][BANDS[sat][numbands]['GREEN'], :, :, :]) - 1 

                ndvi[(sat_properties[sat]['data'][BANDS[sat][numbands]['NIR'], :, :, :] + sat_properties[sat]['data'][BANDS[sat][numbands]['RED'], :, :, :]) == 0] = 0
                gcvi[sat_properties[sat]['data'][BANDS[sat][numbands]['GREEN'], :, :, :] == 0] = 0


            #TODO: Clean this up a bit. No longer include doy/clouds if data is aggregated? 
            if self.normalize:
                sat_properties[sat]['data'] = preprocess.normalization(sat_properties[sat]['data'], sat, self.country)
            
            # Concatenate vegetation indices after normalization
            if sat in ['planet', 's2'] and self.include_indices:
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], np.expand_dims(ndvi, axis=0)), 0)
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], np.expand_dims(gcvi, axis=0)), 0)

            # Concatenate cloud mask bands
            if sat_properties[sat]['cloudmasks'] is not None and self.include_clouds:
                sat_properties[sat]['cloudmasks'] = preprocess.preprocess_clouds(sat_properties[sat]['cloudmasks'], self.model_name, self.timeslice)
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], sat_properties[sat]['cloudmasks']), 0)

            # Concatenate doy bands
            if sat_properties[sat]['doy'] is not None and self.include_doy:
                doy_stack = preprocess.doy2stack(sat_properties[sat]['doy'], sat_properties[sat]['data'].shape)
                sat_properties[sat]['data'] = np.concatenate((sat_properties[sat]['data'], doy_stack), 0)
            
        return sat_properties


class CropTypeBatchSampler(Sampler):
    """
        Groups sequences of similiar length into the same batch to prevent unnecessary computation.
    """
    def __init__(self, dataset, max_batch_size, max_seq_length):
        super(CropTypeBatchSampler, self).__init__(dataset)
        batches = []
        idxs = list(range(len(dataset)))
        
        # shuffle the dataset
        shuffle(idxs)
        lengths = [min(dataset.combined_lengths[i], 2 * max_seq_length) for i in idxs] # 2x since we're measure combined s1 / s2
        
        buckets = defaultdict(list)
        
        for i in idxs:
            buckets[lengths[i] // 10].append(i)
        
        for bucket_length, bucket in buckets.items():
            batch = []
            # TODO: why are grids of length ~60 being grouped with grids of length 100+?
            while len(bucket) > 0:
                grid = bucket.pop()
                batch.append(grid)
                if len(batch) == max_batch_size:
                    batches.append(batch)
                    batch = []
            
            if len(batch) > 0:
                batches.append(batch)
        
        self.batches = batches
        
    def __iter__(self):
        for b in self.batches:
            yield(b)
        
    def __len__(self):
        return len(self.batches)


def pad_to_equal_length(grids):
    # time first
    _, c, h, w = grids[0].shape
    lengths = [grid.shape[0] for grid in grids]
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    
    for i, grid in enumerate(grids):
        t, _, _, _ = grid.shape
        if t < max_len:
            padded = np.zeros((max_len, c, h, w))
            padded[:lengths[i], :, :, :] = grid
            grids[i] = torch.tensor(padded, dtype=torch.float32)
    return grids, lengths
    
    
def collate_var_length(batch):
    """ Collates batch into inputs, label, cloudmasks.
    Batch structured as [(inputs_0, label_0, cloudmasks_0), ..., (inputs_n, label_n, cloudmasks_n)]
    
    Returns:
        inputs, label, cloudmasks
        
        where s1 has all same length (padded to max len)
              s2 has all same length (padded to max len)
              planet has all same length (paddedd to max len)
    """
    batch_size = len(batch)
    labels = [batch[i][1] for i in range(batch_size)]
    labels = torch.stack(labels)
    inputs = {}
    sats = batch[0][0].keys()
    for sat in sats:
        grids = [batch[i][0][sat] for i in range(batch_size)]
        grids, lengths = pad_to_equal_length(grids)
        grids = torch.stack(grids)
        inputs[sat] = grids
        inputs[sat + "_lengths"] = lengths
  
    if 's2' in sats and not isinstance(batch[0][2], bool): # batch[0][2] checks if cloudmasks exist
        cloudmasks = [batch[i][2].transpose(3, 0, 1, 2) for i in range(batch_size)]
        cloudmasks, lengths = pad_to_equal_length(cloudmasks)
        cloudmasks = torch.tensor(np.stack(cloudmasks).transpose(0, 2, 3, 4, 1))
    else:
        cloudmasks = None
        
    return inputs, labels, cloudmasks, False
        
    
class GridDataLoader(DataLoader):

    def __init__(self, args, grid_path, split):
        dataset = CropTypeDS(args, grid_path, split)
        if args.var_length:
            sampler = CropTypeBatchSampler(dataset, max_batch_size=args.batch_size, max_seq_length=args.num_timesteps)
            super(GridDataLoader, self).__init__(dataset,
                                                 batch_sampler=sampler,
                                                 num_workers=args.num_workers,
                                                 collate_fn=collate_var_length,
                                                 pin_memory=True)
        else:
            super(GridDataLoader, self).__init__(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=args.shuffle,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

            
def get_dataloaders(country, dataset, args):
    dataloaders = {}
    for split in SPLITS:
        if country in ['southsudan', 'ghana']:
            grid_path = os.path.join(GRID_DIR[country], f"{country}_{dataset}_final_{split}_32")
        else:
            grid_path = os.path.join(GRID_DIR[country], f"{country}_{dataset}_final_{split}")
        dataloaders[split] = GridDataLoader(args, grid_path, split)

    return dataloaders
