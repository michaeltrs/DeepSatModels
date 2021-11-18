"""

File that houses all functions used to format, preprocess, or manipulate the data.

Consider this essentially a util library specifically for data manipulation.

"""
import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.utils.rnn as rnn
import numpy as np
from constants import *
from util import *
import os
import random


def normalization(grid, satellite, country):
    """ Normalization based on values defined in constants.py
    Args: 
      grid - (tensor) grid to be normalized
      satellite - (str) describes source that grid is from ("s1" or "s2")

    Returns:
      grid - (tensor) a normalized version of the input grid
    """
    num_bands = grid.shape[0]
    means = MEANS[satellite][country]
    stds = STDS[satellite][country]
    grid = (grid-means[:num_bands].reshape(num_bands, 1, 1, 1))/stds[:num_bands].reshape(num_bands, 1, 1, 1)
    
    if satellite not in ['s1', 's2', 'planet']:
        raise ValueError("Incorrect normalization parameters")
    return grid
        
def reshapeForLoss(y):
    """ Reshapes labels or preds for loss fn.
    To get them to the correct shape, we permute: 
      [batch x classes x rows x cols] --> [batch x rows x cols x classes]
      and then reshape to [N x classes], where N = batch*rows*cols
    """
    # [batch x classes x rows x cols] --> [batch x rows x cols x classes]
    y = y.permute(0, 2, 3, 1)
    # [batch x rows x cols x classes] --> [batch*rows*cols x classes]
    y = y.contiguous().view(-1, y.shape[3])
    return y

def maskForLoss(y_pred, y_true):
    """
    Masks y_pred and y_true with valid pixel locations. 

    Args:    
      y_true - (torch tensor) torch.Size([batch_size*img_height*img_width, num_classes]) 
                tensor of ground truth crop classes
      y_pred - (torch tensor) torch.Size([batch_size*img_height*img_width, num_classes])
                tensor of predicted crop classes

    Returns: 
      y_true - (torch tensor) torch.Size([batch_size*img_height*img_width]) 
                tensor of ground truth crop classes, argmaxed
      y_pred - (torch tensor) torch.Size([batch_size*img_height*img_width, num_classes])
                tensor of predicted crop classes
    """
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor)

    loss_mask_repeat = loss_mask.unsqueeze(1).repeat(1,y_pred.shape[1]).type(torch.FloatTensor).cuda()
    y_pred = y_pred * loss_mask_repeat
   
    # take argmax to get true values from one-hot encoding 
    _, y_true = torch.max(y_true, dim=1)
    y_true = y_true * loss_mask

    return y_pred, y_true

def maskForMetric(y_pred, y_true):
    """
    Masks y_pred and y_true with valid pixel locations for metric calculations and returns 
     vectors of only valid locations

    Args:    
      y_true - (torch tensor) torch.Size([batch_size*img_height*img_width, num_classes]) 
                tensor of ground truth crop classes
      y_pred - (torch tensor) torch.Size([batch_size*img_height*img_width, num_classes])
                tensor of predicted crop classes

    Returns: 
      y_true - (torch tensor) torch.Size([valid_pixel_locations]) 
                tensor of ground truth crop classes, argmaxed
      y_pred - (torch tensor) torch.Size([valid_pixel_locations])
                tensor of predicted crop classes, argmaxed
    """
    # Create mask for valid pixel locations
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor)
    # Take argmax for labels and targets
    _, y_true = torch.max(y_true, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)

    # Get only valid locations
    y_true = y_true[loss_mask == 1]
    y_pred = y_pred[loss_mask == 1]
    return y_pred, y_true

def onehot_mask(mask, num_classes):
    """
    Return a one-hot version of the mask for a grid

    Args: 
      mask - (np array) mask for grid that contains crop labels according 
             to '/home/data/crop_dict.npy'
      num_classes - (int) number of classes to be encoded into the one-hot
                    mask, as all classes in crop_dict are in the original 
                    mask. This must include other as one of the classes. For 
                    example, if the 5 main crop types + other are used, 
                    then num_classes = 6.

    Returns: 
      Returns a mask of size [64 x 64 x num_classes]. If a pixel was unlabeled, 
      it has 0's in all channels of the one hot mask at that pixel location.
    """
    if num_classes == 2:
        # TODO: why do we treat this as a separate case?
        mask[(mask != 2) & (mask > 0)] = 1
    else:
        mask[mask > num_classes] = 0
    return np.eye(num_classes+1)[mask][:, :, 1:] 

def doy2stack(doy_vec, in_shp):
    """ Creates input bands for day of year values
    Args:
      doy_vec - (vector) vector of day of year values 
      in_shp - (tuple) shape that doy bands should take on

    Returns: 
      stack - (tensor) stack of doy values now in fature bands, of shape in_shp
    """
    b, r, c, t = in_shp
    assert t == len(doy_vec)

    # normalize
    doy_vec = (doy_vec - 177.5) / 177.5
    doy = torch.from_numpy(doy_vec)

    # create feature bands filled with the doy values
    stack = doy.unsqueeze(0).expand(c, t).unsqueeze(0).expand(r, c, t).unsqueeze(0)
    return stack

def retrieve_best_s2_grid(grid_name, country):
    """ Retrieves the least cloudy s2 image of the grid specified.

    Args:
        grid_name - (string) string representation of the grid number

    Returns:
        grid - (npy array) concatenation of the s1 and s2 values of the grid over time
    """
    s2_path = '{}/{}/{}'.format(GCP_DATA_DIR, country, S2_DIR)
    
    # Read in Sentinel-2 stack
    s2_fname = [f for f in os.listdir(s2_path) if f.endswith('_{}.npy'.format(grid_name.zfill(6)))]
    s2_grid_path = "/".join((s2_path, s2_fname[0]))
    s2_stack = np.load(s2_grid_path)    

    # Read in coresponding cloud stack
    s2_cloudmask_fname = [f for f in os.listdir(s2_path) if f.endswith('_{}_cloudmask.npy'.format(grid_name.zfill(6)))]
    s2_grid_cloudmask_path = "/".join((s2_path, s2_cloudmask_fname[0]))
    cloud_stack = np.load(s2_grid_cloudmask_path)

    # Check the timestamps of the data and cloud masks are the same
    assert s2_stack.shape[-1] == cloud_stack.shape[-1]    
    
    # Get the index of the least cloudy image
    idx = get_least_cloudy_idx(cloud_stack)

    # Take the Sentinel-2 grid at the least cloudy index
    grid = s2_stack[:, :, :, idx]
    return grid

def get_least_cloudy_idx(cloud_stack):
    """ Get index of least cloudy image from a stack of cloud masks
    """
    cloud_stack = remap_cloud_stack(cloud_stack)
    cloudiness = np.mean(cloud_stack, axis=(0, 1))
    least_cloudy_idx = np.argmax(cloudiness)
    return least_cloudy_idx

def preprocess_grid(grid, model_name, time_slice=None, transform=False, rot=None):
    """ Returns a preprocessed version of the grid based on the model.

    Args:
        grid - (npy array) concatenation of the s1 and s2 values of the grid
        model_name - (string) type of model (ex: "bidir_clstm")
        time_slice - (int) which timestamp to be used in FCN
    """

    if model_name in ["bidir_clstm", "fcn_crnn", "fcn", "random_forest", "mi_clstm", "only_clstm_mi"]:
        return preprocessGrid(grid, transform, rot, time_slice)
    
    elif model_name == "unet":
        return preprocessGridForUNet(grid, transform, rot, time_slice)
    
    elif model_name == "unet3d":
        return preprocessGridForUNet3D(grid, transform, rot, time_slice)

    raise ValueError(f'Model: {model_name} unsupported')

def preprocess_clouds(clouds, model_name, time_slice=None, transform=False, rot=None):
    """ Returns a preprocessed version of the cloudmask based on the model.

    Args:
        clouds - (npy array) cloudmasks for s2 imagery of the grid
        model_name - (string) type of model (ex: "C-LSTM")
        time_slice - (int) which timestamp to be used in FCN
    """
    if model_name in ["bidir_clstm", "fcn", "unet", "fcn_crnn", "unet3d", "random_forest", "mi_clstm", "only_clstm_mi"]:
        return preprocessClouds(clouds)
    
    raise ValueError(f'Model: {model_name} unsupported')

def preprocess_label(label, model_name, num_classes=None, transform=False, rot=None):
    """ Returns a preprocess version of the label based on the model.

    Usually this just means converting to a one hot representation and 
    shifting the classes dimension of the mask to be the first dimension.

    Args:
        label - (npy arr) categorical labels for each pixel
        model_name - (str) name of the model
    Returns:
        (npy arr) [num_classes x 64 x 64]
    """
    # TODO: make this into a constant somewhere so we don't have to keep adding models
    if model_name in ["bidir_clstm", "fcn", "fcn_crnn", "unet", "unet3d", "random_forest", "mi_clstm", "only_clstm_mi"]:
        assert not num_classes is None
        return preprocessLabel(label, num_classes, transform, rot)
    
    raise ValueError(f'Model: {model_name} unsupported')
    
def preprocessLabel(label, num_classes, transform, rot):
    """ Converts to onehot encoding and shifts channels to be first dim.

    Args:
        label - (npy arr) [64x64] categorical labels for each pixel
        num_classes - (npy arr) number of classes 
    """
    if transform:
        label = np.fliplr(label)
        label = np.rot90(label, k=rot)
    label = onehot_mask(label, num_classes)
    label = np.transpose(label, [2, 0, 1])
    label = torch.tensor(label.copy(), dtype=torch.float32)
    return label

def saveGridAsImg(grid, fname):
    minval = 1100
    maxval = 2100
    for i in range(grid.shape[0]):
        grid[i] = (grid[i] - minval) / (maxval -minval)
    toImg = transforms.ToPILImage()
    grid_as_img = toImg(torch.squeeze(grid[0, [2, 1, 0]]))
    grid_as_img.save(fname)

def preprocessGrid(grid, transform, rot, time_slice=None):
    grid = moveTimeToStart(grid)
    if transform:
        grid = grid[:, :, :, ::-1]
        grid = np.rot90(grid, k=rot, axes=(2, 3))
    grid = torch.tensor(grid.copy(), dtype=torch.float32)

    if time_slice is not None:
        grid = grid[timeslice, :, :, :]
    return grid

def preprocessGridForUNet3D(grid, transform, rot, time_slice=None):
    grid = preprocessGrid(grid, transform, rot, time_slice) 
    grid = np.transpose(grid, [1, 0, 2, 3])
    return grid 

def preprocessGridForUNet(grid, transform, rot, time_slice=None):
    grid = preprocessGrid(grid, transform, rot, time_slice) 
    if time_slice is None:
        grid = mergeTimeBandChannels(grid)
    return grid 
   
def preprocessClouds(clouds):
    """ Normalize cloud mask input bands
    """
    clouds = np.expand_dims(clouds, 0)
    # normalize to -1, 1
    clouds = (clouds - 1.5)/1.5
    return clouds

def moveTimeToStart(arr):
    """ Moves time axis to the first dim.
        
    Args:
        arr - (npy arr) [bands x rows x cols x timestamps] 
    """
    
    return np.transpose(arr, [3, 0, 1, 2])

def mergeTimeBandChannels(arr):
    """ Merge timestamps and band channels for UNet input [(bands x timestamps) x rows x cols]
    
    Args:
        arr - (npy arr) [timestamps x bands x rows x cols] 
    """
    arr_merge = arr.reshape(-1,arr.shape[2],arr.shape[3])
    return arr_merge

def truncateToSmallestLength(batch):
    """ Truncates len of all sequences to MIN_TIMESTAMPS.

    Args:
        batch - (tuple of list of npy arrs) batch[0] is a list containing the torch versions of grids where each grid is [timestamps x bands x rows x cols]; batch[1] is a list containing the torch version of the labels

    """
    batch_X = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    for i in range(len(batch_X)):
        if len(batch_X[i].shape)>3:
            batch_X[i], _, _ = sample_timeseries(batch_X[i], MIN_TIMESTAMPS, timestamps_first=True)
    
    return [torch.stack(batch_X), torch.stack(batch_y)]


def padToVariableLength(batch):
    """ Pads all sequences to same length (variable per batch).

    Specifically, pads sequences to max length sequence with 0s.

    Args:
        batch - (tuple of list of npy arrs) batch[0] is a list containing the torch versions of grids where each grid is [timestamps x bands x rows x cols]; batch[1] is a list containing the torch version of the labels

    Returns:
        batch_X - (list of torch arrs) padded versions of each grid
    """
    batch_X = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    batch_X.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [x.shape[0] for x in batch_X]
    lengths = torch.tensor(lengths, dtype=torch.float32)
    batch_X = rnn.pad_sequence(batch_X, batch_first=True)
    return [batch_X, lengths, batch_y]


def concat_s1_s2_planet(s1, s2, planet, resize_planet):
    """ Returns a concatenation of s1, s2, and planet data.

    Downsamples the larger series to size of the smaller one and returns the concatenation on the time axis.
    If None, the source is excluded.

    Args:
        s1 - (npy array) [bands x rows x cols x timestamps] Sentinel-1 data
        s2 - (npy array) [bands x rows x cols x timestamps] Sentinel-2 data
        planet - (npy array) [bands x rows x cols x timestamps] Planet data

    Returns:
        (npy array) [bands x rows x cols x min(num s1 timestamps, num s2 timestamps, num planet timestamps) 
         Concatenation of s1, s2, and planet data
    """
    inputs = [s1, s2, planet]

    use_s1 = True if inputs[0] is not None else False
    use_s2 = True if inputs[1] is not None else False
    use_planet = True if inputs[2] is not None else False

    # Get indices that are not none and index inputs and ntimes
    not_none = [i for i in range(len(inputs)) if inputs[i] is not None]
    inputs = [inputs[i] for i in not_none]
    ntimes = [i.shape[-1] for i in inputs]
 
    if len(np.unique(ntimes)) == 1:
        if not use_planet or (use_planet and resize_planet) or (use_planet and not resize_planet and len(inputs)==1):
            return np.concatenate(inputs, axis=0), None
        elif use_planet and not resize_planet:
            return np.concatenate(inputs[:-1], axis=0), inputs[-1]
        else:
            raise ValueError('Concatenation error given specified flags')
    else:
        min_ntimes = np.min(ntimes)
        min_ntimes_idx = np.argmin(ntimes)
    
        sampled = []
        for idx, sat in enumerate(inputs):
            if idx == min_ntimes_idx:
                sampled.append(sat)
            else:
                cur_sat, _, _ = sample_timeseries(sat, min_ntimes)
                sampled.append(cur_sat)
        if not use_planet or (use_planet and resize_planet):
            return np.concatenate(sampled, axis=0), None
        elif use_planet and not resize_planet:
            return np.concatenate(sampled[:-1], axis=0), sampled[-1]
        else:
            raise ValueError('Concatenation error given specified flags')


def remap_cloud_stack(cloud_stack):
    """ 
     Remap cloud mask values so clearest pixels have highest values
     Rank by clear, shadows, haze, clouds
     clear = 0 --> 3, clouds = 1  --> 0, shadows = 2 --> 2, haze = 3 --> 1
    """
    remapped_cloud_stack = np.zeros_like((cloud_stack))
    remapped_cloud_stack[cloud_stack == 0] = 3
    remapped_cloud_stack[cloud_stack == 2] = 2
    remapped_cloud_stack[cloud_stack == 3] = 1
    return remapped_cloud_stack


def sample_timeseries(img_stack, num_samples, dates=None, cloud_stack=None, remap_clouds=True, reverse=False, verbose=False, timestamps_first=False, least_cloudy=False, sample_w_clouds=True, all_samples=False):
    """
    Args:
      img_stack - (numpy array) [bands x rows x cols x timestamps], temporal stack of images
      num_samples - (int) number of samples to sample from the img_stack (and cloud_stack)
                     and must be <= the number of timestamps
      dates - (numpy array) vector of dates that correspond to the timestamps in the img_stack and
                     cloud_stack
      cloud_stack - (numpy array) [rows x cols x timestamps], temporal stack of cloud masks
      remap_clouds - (boolean) whether to remap cloud masks to new class values, ordered in terms of view obstruction
      reverse - (boolean) take 1 - probabilities, encourages cloudy images to be sampled
      seed - (int) a random seed for sampling
      verbose - (boolean) whether to print out information when function is executed
      timestamps_first - (boolean) if True, timestamps occupy first dimension
      least_cloudy - (bool) if true, take the least cloudy images rather than sampling with probability
      sample_w_clouds - (bool) if clouds are used as input, whether or not to use them for sampling
    Returns:
      sampled_img_stack - (numpy array) [bands x rows x cols x num_samples], temporal stack
                          of sampled images
      sampled_dates - (list) [num_samples], list of dates associated with the samples
      sampled_cloud_stack - (numpy array) [rows x cols x num_samples], temporal stack of
                            sampled cloud masks, only returned if cloud_stack was an input

    To read in img_stack from npy file for input img_stack:

       img_stack = np.load('/home/data/ghana/s2_64x64_npy/s2_ghana_004622.npy')
    
    To read in cloud_stack from npy file for input cloud_stack:

       cloud_stack = np.load('/home/data/ghana/s2_64x64_npy/s2_ghana_004622_mask.npy')
    
    To read in dates from json file for input dates:
       
       with open('/home/data/ghana/s2_64x64_npy/s2_ghana_004622.json') as f:
           dates = json.load(f)['dates']

    """
    timestamps = img_stack.shape[0] if timestamps_first else img_stack.shape[3]
   
    if timestamps < num_samples:
        if isinstance(cloud_stack, np.ndarray):
            return img_stack, dates, cloud_stack
        else:
            return img_stack, dates, None 
        
    # Given a stack of cloud masks, remap it and use to compute scores
    if isinstance(cloud_stack,np.ndarray):
        remapped_cloud_stack = remap_cloud_stack(cloud_stack)
    if isinstance(cloud_stack,np.ndarray) and sample_w_clouds:
        scores = np.mean(remapped_cloud_stack, axis=(0, 1))
    else:
        if verbose:
            print('NO INPUT CLOUD MASKS. USING RANDOM SAMPLING!')
        scores = np.ones((timestamps,))

    if reverse:
        scores = 3 - scores
    if least_cloudy:
        samples = scores.argsort()[-num_samples:]
    else:
        # Compute probabilities of scores with softmax
        probabilities = softmax(scores)

        if not all_samples:
            # Sample from timestamp indices according to probabilities
            samples = np.random.choice(timestamps, size=num_samples, replace=False, p=probabilities)
        else:
            # Take all indices as samples
            samples = list(range(timestamps))   
 
    # Sort samples to maintain sequential ordering
    samples.sort()

    # Use sampled indices to sample image and cloud stacks
    if timestamps_first:
        sampled_img_stack = img_stack[samples, :, :, :]
    else:
        sampled_img_stack = img_stack[:, :, :, samples]
    
    # Samples dates
    sampled_dates = None
    if dates is not None:
        sampled_dates = dates[samples] 

    if isinstance(cloud_stack, np.ndarray):
        if remap_clouds:
            if all_samples:
                sampled_cloud_stack = remapped_cloud_stack
            else: 
                sampled_cloud_stack = remapped_cloud_stack[:, :, samples]
        else:
            if all_samples:
                sampled_cloud_stack = cloud_stack
            else:
                sampled_cloud_stack = cloud_stack[:, :, samples]
        if all_samples:
            return img_stack, dates, sampled_cloud_stack
        else:
            return sampled_img_stack, sampled_dates, sampled_cloud_stack
    else:
        if all_samples:
            return img_stack, dates, None
        else:
            return sampled_img_stack, sampled_dates, None    

    
def vectorize(home, country, data_set, satellite, ylabel_dir, band_order= 'bytime', random_sample = True, num_timestamp = 25, reverse = False, seed = 0):
    """
    Save pixel arrays  # pixels * # features for raw
    
    Args:
      home - (str) the base directory of data
      country - (str) string for the country 'ghana', 'tanzania', 'southsudan'
      data_set - (str) balanced 'small' or unbalanced 'full' dataset
      satellite - (str) satellite to use 's1' 's2' 's1_s2'
      ylabel_dir - (str) dir to load ylabel
      band_order - (str) band order: 'byband', 'bytime'
      random_sample - (boolean) use random sample (True) or take median (False)
      num_timestamp - (num) minimum num for time stamp
      reverse - (boolean) use cloud mask reversed softmax probability or not 
      seed - (int) random sample seed

    Output: 
      npy array, [num_examples, features] saved in HOME/pixel_arrays

    """

    satellite_original = str(np.copy(satellite))

    X_total3types = {}
    y_total3types = {}
    
    bad_list = np.load(os.path.join(home, country, 'bad_timestamp_grids_list.npy')) # just for num_stamp 25
    
    ## Go through 'train' 'val' 'test'
    for data_type in ['train','val','test']:

        if satellite_original == 's1':
            num_band = 2
            satellite_list = ['s1']
        elif satellite_original == 's2':
            num_band = 10
            satellite_list = ['s2']
        elif satellite_original == 's1_s2':
            num_band = [2, 10]
            satellite_list = ['s1', 's2']

        X_total = {}

        for satellite in satellite_list:
            #X: # of pixels * # of features
            gridded_dir = os.path.join(home, country, satellite+'_64x64_npy')
            gridded_IDs = sorted(np.load(os.path.join(home, country, country+'_'+data_set+'_'+data_type)))
            gridded_fnames = [satellite+'_'+country+'_'+gridded_ID+'.npy' for gridded_ID in gridded_IDs]
            good_grid = np.where([gridded_ID not in bad_list for gridded_ID in gridded_IDs])[0]
            
            # Time json
            time_fnames = [satellite+'_'+country+'_'+gridded_ID+'.json' for gridded_ID in gridded_IDs]
            time_json = [json.loads(open(os.path.join(gridded_dir,f),'r').read())['dates'] for f in time_fnames]
            
            # keep num of timestamps >=25
            gridded_IDs = [gridded_IDs[idx] for idx in good_grid]
            gridded_fnames = [gridded_fnames[idx] for idx in good_grid]
            time_json = [time_json[idx] for idx in good_grid]
            time_fnames = [time_fnames[idx] for idx in good_grid]
            
            
            if random_sample == True and satellite == 's2':
                # cloud mask
                cloud_mask_fnames = [satellite+'_'+country+'_'+gridded_ID+'_mask.npy' for gridded_ID in gridded_IDs]
                num_band = num_band + 1

            Xtemp = np.load(os.path.join(gridded_dir,gridded_fnames[0]))

            grid_size_a = Xtemp.shape[1]
            grid_size_b = Xtemp.shape[2]

            X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
            X[:] = np.nan

            for i in range(len(gridded_fnames)):

                X_one = np.load(os.path.join(gridded_dir,gridded_fnames[i]))[0:num_band,:,:]
                Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
                Xtemp[:] = np.nan

                if random_sample == True and satellite == 's2':
                    cloud_stack = np.load(os.path.join(gridded_dir,cloud_mask_fnames[i]))
                    [sampled_img_stack, _,  sampled_cloud_stack] = sample_timeseries(X_one, num_samples = num_timestamp, cloud_stack=cloud_stack, reverse = reverse)
                    Xtemp = np.copy(np.vstack((sampled_img_stack,np.expand_dims(sampled_cloud_stack, axis=0))))
                
                elif random_sample == True and satellite == 's1':
                    [sampled_img_stack, _, _] = sample_timeseries(X_one, num_samples = num_timestamp, cloud_stack=None, reverse = reverse)
                    Xtemp = np.copy(sampled_img_stack)
                    
                else:
                    time_idx = np.array([np.int64(time.split('-')[1]) for time in time_json[i]])

                    # Take median in each bucket
                    for j in np.arange(12)+1:
                        Xtemp[:,:,:,j-1] = np.nanmedian(X_one[:,:,:,np.where(time_idx==j)][:,:,:,0,:],axis = 3)

                Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
                if band_order == 'byband':
                    Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
                elif band_order == 'bytime':
                    Xtemp = np.swapaxes(Xtemp, 0, 1)
                    Xtemp = np.swapaxes(Xtemp, 1, 2).reshape(Xtemp.shape[0],-1)

                X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

            #y: # of pixels
            y_mask = get_y_label(home, country, data_set, data_type, ylabel_dir)
            y_mask = y_mask[good_grid,:,:]
            y = y_mask.reshape(-1)   
            crop_id = crop_ind(y)

            X_noNA = fill_NA(X[crop_id,:][0,:,:])
            y = y[crop_id]

            X_total[satellite] = X_noNA

        if len(satellite_list)<2:
            X_total3types[data_type] = np.copy(X_total[satellite_original])
        else:
            X_total3types[data_type] = np.hstack((X_total['s1'], X_total['s2']))

        y_total3types[data_type] = np.copy(y)

        
        
        if random_sample == True and satellite == 's2':
            output_fname = "_".join([data_set, 'raw', satellite_original, 'cloud_mask','reverse'+str(reverse), band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'cloud_s2', 'reverse_'+str(reverse).lower(), output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, 'cloud_mask','reverse'+str(reverse), band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'cloud_s2', 'reverse_'+str(reverse).lower(), output_fname), y_total3types[data_type])
            
        elif random_sample == True and satellite == 's1':
            output_fname = "_".join([data_set, 'raw', satellite_original, 'sample', band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'sample_s1', output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, 'sample', band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'sample_s1', output_fname), y_total3types[data_type])

        else: 
            output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), y_total3types[data_type])
   
    return [X_total3types, y_total3types]  

