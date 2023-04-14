from __future__ import print_function, division
# from skimage import io, transform
import numpy as np
import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, utils
from copy import deepcopy
import random
from utils.config_files_utils import get_params_values
from scipy import ndimage


remap_label_dict = {
    'labels_20k2k': {0: 20,
                     1: 0, 2: 1, 3: 2,
                     4: 20, 5: 20,
                     6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11,
                     15: 20, 16: 20, 17: 20, 18: 20, 19: 20,
                     20: 12,
                     21: 20, 22: 20, 23: 20, 24: 20,
                     25: 13,
                     26: 20, 27: 20, 28: 20, 29: 20, 30: 20, 31: 20, 32: 20,
                     33: 14, 34: 15, 35: 16, 36: 17,
                     37: 20, 38: 20, 39: 20, 40: 20, 41: 20, 42: 20, 43: 20, 44: 20, 45: 20, 46: 20, 47: 20,
                     48: 18,
                     49: 20, 50: 20, 51: 20, 52: 20, 53: 20, 54: 20, 55: 20, 56: 20, 57: 20, 58: 20, 59: 20, 60: 20,
                     61: 20, 62: 20, 63: 20, 64: 20, 65: 20, 66: 20, 67: 20, 68: 20, 69: 20, 70: 20, 71: 20, 72: 20,
                     73: 19,
                     74: 20, 75: 20, 76: 20, 77: 20, 78: 20, 79: 20, 80: 20, 81: 20, 82: 20, 83: 20, 84: 20, 85: 20,
                     86: 20, 87: 20, 88: 20, 89: 20, 90: 20, 91: 20, 92: 20, 93: 20, 94: 20, 95: 20, 96: 20, 97: 20,
                     98: 20, 99: 20, 100: 20, 101: 20, 102: 20, 103: 20, 104: 20, 105: 20, 106: 20, 107: 20,
                     108: 20, 109: 20, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20, 116: 20, 117: 20,
                     118: 20, 119: 20, 120: 20, 121: 20, 122: 20, 123: 20, 124: 20, 125: 20, 126: 20, 127: 20,
                     128: 20, 129: 20, 130: 20, 131: 20, 132: 20, 133: 20, 134: 20, 135: 20, 136: 20, 137: 20,
                     138: 20, 139: 20, 140: 20, 141: 20, 142: 20, 143: 20, 144: 20, 145: 20, 146: 20, 147: 20,
                     148: 20, 149: 20, 150: 20, 151: 20, 152: 20, 153: 20, 154: 20, 155: 20, 156: 20, 157: 20,
                     158: 20, 159: 20, 160: 20, 161: 20, 162: 20, 163: 20, 164: 20, 165: 20, 166: 20, 167: 20
                     }
                    }

dataset_img_size_dict = {'psetae_repl_55x55_11': 48, 'psetae_repl_2016_100_3': 48, 'psetae_repl_2017_100_3': 48,
                         'psetae_repl_2018_100_3': 48}


def get_label_names(label_dict):
    names = {}
    for label in label_dict:
        names[remap_label_dict[label]] = label_dict[label]
    return names

    
def France_segmentation_transform(model_config, data_config, is_training):
    """
    """
    input_img_res = model_config['img_res']
    max_seq_len = data_config['max_seq_len']
    n_class = model_config['num_classes']
    extra_data = get_params_values(data_config, 'extra_data', [])
    doy_bins = get_params_values(data_config, 'doy_bins', None)
    # label_sc_fact = get_params_values(data_config, 'label_sc_fact', 1.)
    include_ids = get_params_values(data_config, 'include_ids', True)  # False)
    ignore_background = get_params_values(model_config, 'ignore_background', True)  # if True mask out background locs from loss, metrics
    output_magnification = int(get_params_values(model_config, 'output_magnification', 1))  # if True mask out background locs from loss, metrics
    label_magnification = int(get_params_values(data_config, 'label_magnification', output_magnification))  # if True mask out background locs from loss, metrics
    if label_magnification > 1:
        output_magnification = max(output_magnification, label_magnification)
    keep_x1_labels = get_params_values(model_config, 'keep_x1_labels', False)  # if True mask out background locs from loss, metrics
    bidir_input = model_config['architecture'] == "ConvBiRNN"
    equal_int_bound = get_params_values(data_config, 'equal_int_bound', False)

    dataset = data_config['dataset']
    dataset_img_res = 48  # dataset_img_size_dict[dataset]
    train_stage = get_params_values(model_config, 'train_stage', None)

    label_map = get_params_values(data_config, 'label_map', dataset)
    if label_map is None:
        label_map = dataset
    if label_map in ['labels_19', 'labels_44', 'labels_20k2k']:
        label_type = 'labels'
    else:
        label_type = 'groups'

    ground_truth_target = 'labels'
    ground_truth_masks = ['full_ass', 'part_ass']
    ground_truths = ['labels', 'full_ass', 'part_ass']
    if include_ids:
        ground_truths.append('ids')

    if (output_magnification is not None) and (output_magnification != 1) and (label_magnification != 1):
        print("output magnification factor: ", output_magnification)
        # output_magnification = "x%s" % 4  # output_magnification
        ground_truths = [gt + "_x%d" % output_magnification for gt in deepcopy(ground_truths)]
        ground_truth_target = 'labels' + "_x%s" % output_magnification
        ground_truth_masks = [gt + "_x%s" % output_magnification for gt in deepcopy(ground_truth_masks)]


    transform_list = []
    transform_list.append(ToTensor(label_type=label_type, ground_truths=ground_truths))                                  # data from numpy arrays to torch.float32

    transform_list.append(RemapLabel(remap_label_dict[label_map], ground_truth2remap=ground_truth_target))                      # remap labels to new values

    transform_list.append(Normalize())                                 # normalize all inputs individually

    transform_list.append(Rescale(output_size=(dataset_img_res, dataset_img_res), ground_truths=[]))               # scale x20, x60 to match x10 H, W

    if dataset_img_res != input_img_res:
        transform_list.append(
            Crop(img_size=dataset_img_res, crop_size=input_img_res, random=is_training, ground_truths=ground_truths))  # random crop

    transform_list.append(TileDates(H=input_img_res, W=input_img_res, doy_bins=doy_bins))                       # tile day and year to shape TxWxHx1
    transform_list.append(Concat(concat_keys=['x10', 'x20', 'x60', 'doy']))     # concat x10, x20, x60, day, year
    if bidir_input:
        transform_list.append(AddBackwardInputs())
    # transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=True, from_start=False))  # pad with zeros to maximum sequence length
    transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=False, from_start=True))  # pad with zeros to maximum sequence length

    if is_training:
        transform_list.append(HVFlip(hflip_prob=0.5, vflip_prob=0.5, ground_truths=ground_truths))  # horizontal, vertical flip

    transform_list.append(Add2UnkClass(unk_class=remap_label_dict[label_map][0], ground_truth_target=ground_truth_target,
                                       ground_truth_masks=ground_truth_masks))  # extract unknown label mask
    if ignore_background:
        transform_list.append(UnkMask(unk_class=remap_label_dict[label_map][0], ground_truth_target=ground_truth_target))  # extract unknown label mask

    # if include_ids:
    #     transform_list.append(UpdateIds())

    #     transform_list.append(SOLOGroundTruths(num_grid=24, label_res=dataset_img_res, unk_class=remap_label_dict[label_map][0]))
    # transform_list.append(AddBagOfLabels(n_class=n_class))

    if (output_magnification > 1) and (label_magnification != output_magnification) and (label_magnification != 1):
        transform_list.append(Rescale(output_size=(label_magnification*dataset_img_res, label_magnification*dataset_img_res),
                                      ground_truths=['labels_x4'], rescale_gt_only=True))

    if (output_magnification > 1) and (not keep_x1_labels) and (label_magnification != 1):
        transform_list.append(RenameGroundTruth(rename_dict={'labels' + "_x%d" % output_magnification: 'labels'}))

    if 'edge_labels' in extra_data:
        transform_list.append(AddEdgeLabel())

    if equal_int_bound:
        transform_list.append(EqualIntBoundPoints())

    if train_stage == 0:
    # if 'same_class_labels' in extra_data:
    #     global_attn = model_config['global_attn']
        kernel_size = model_config['cscl_win_size']
        kernel_stride = model_config['cscl_win_stride']
        kernel_dilation = model_config['cscl_win_dilation']
        add_mask = False  # data_config['add_mask']
        transform_list.append(AddCSCLLabels(
            kernel_size=kernel_size, kernel_stride=kernel_stride, kernel_dilation=kernel_dilation,
            pad_value=99, add_mask=add_mask))

    return transforms.Compose(transform_list)


class RenameGroundTruth(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, rename_dict):
        self.rename_dict = rename_dict

    def __call__(self, sample):
        for key in self.rename_dict:
            sample[self.rename_dict[key]] = sample[key].clone()
            del sample[key]
        return sample


# ['groups_x4', 'full_ass_x4', 'labels_x4', 'invalid_x4', 'ratios_x4', 'ids_x4', 'part_ass_x4']
# 1
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, label_type='groups', ground_truths=[]):
        self.label_type = label_type
        self.ground_truths = ground_truths

    def __call__(self, sample):
        tensor_sample = {}
        # inputs
        tensor_sample['x10'] = torch.stack([torch.tensor(sample[key].astype(np.float32)) for key in ['B04', 'B03', 'B02', 'B08']]).permute(1, 2, 3, 0)
        tensor_sample['x20'] = torch.stack([torch.tensor(sample[key].astype(np.float32)).type(torch.float32) for key in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).permute(1, 2, 3, 0)
        tensor_sample['x60'] = torch.stack([torch.tensor(sample[key].astype(np.float32)) for key in ['B01', 'B09', 'B10']]).permute(1, 2, 3, 0)
        tensor_sample['doy'] = torch.tensor(np.array(sample['doy']).astype(np.float32))
        # ground truths
        for gt in self.ground_truths:
            tensor_sample[gt] = torch.tensor(sample[gt].astype(np.float32)).unsqueeze(dim=0).permute(1, 2, 0)  # pixels assigned fully to at least two parcels

        return tensor_sample


# 3
class RemapLabel(object):
    """
    Remap labels from original values to new consecutive integers
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    
    def __init__(self, labels_dict, ground_truth2remap='labels'):
        assert isinstance(labels_dict, (dict,))
        self.labels_dict = labels_dict
        self.ground_truth2remap = ground_truth2remap
    
    def __call__(self, sample):
        labels = sample[self.ground_truth2remap]
        not_remapped = torch.ones(labels.shape, dtype=torch.bool)
        for label_ in self.labels_dict:
            label_idx = labels == label_
            remap_idx = label_idx & not_remapped
            labels[remap_idx] = self.labels_dict[label_]
            not_remapped[remap_idx] = False
        sample[self.ground_truth2remap] = labels
        return sample


# 4
class Normalize(object):
    """
    Normalize inputs as in https://arxiv.org/pdf/1802.02080.pdf
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        sample['x10'] = sample['x10'] * 1e-4
        sample['x20'] = sample['x20'] * 1e-4
        sample['x60'] = sample['x60'] * 1e-4
        sample['doy'] = sample['doy'] / 365.0001  # 365 + h, h = 0.0001 to avoid placing day 365 in out of bounds bin
        # sample['year'] = sample['year'] - 2016
        return sample


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, img_size, crop_size, random=False, ground_truths=[]):
        self.img_size = img_size
        self.crop_size = crop_size
        self.random = random
        if not random:
            self.top = int((img_size - crop_size) / 2)
            self.left = int((img_size - crop_size) / 2)
        self.ground_truths = ground_truths

    def __call__(self, sample):
        if self.random:
            top = torch.randint(self.img_size - self.crop_size, (1,))[0]
            left = torch.randint(self.img_size - self.crop_size, (1,))[0]
        else:  # center
            top = self.top
            left = self.left

        sample['x10'] = sample['x10'][:, top:top+self.crop_size, left:left+self.crop_size]
        sample['x20'] = sample['x20'][:, top:top+self.crop_size, left:left+self.crop_size]
        sample['x60'] = sample['x60'][:, top:top+self.crop_size, left:left+self.crop_size]

        for gt in self.ground_truths:
            sample[gt] = sample[gt][top:top+self.crop_size, left:left+self.crop_size]

        return sample


# 5
class Rescale(object):
    """
    Rescale the image in a sample to a given square side
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, output_size, ground_truths=[], rescale_gt_only=False):
        assert isinstance(output_size, (tuple,))
        self.new_h, self.new_w = output_size
        self.ground_truths = ground_truths
        self.rescale_gt_only = rescale_gt_only

    def __call__(self, sample):
        # inputs
        if not self.rescale_gt_only:
            for inputc in ['x20', 'x60']:  # 'x10',
                sample[inputc] = self.rescale_3d_map(sample[inputc], mode='bilinear')

        # ground truths
        for gt in self.ground_truths:
            sample[gt] = self.rescale_2d_map(sample[gt], mode='nearest')

        return sample

    def rescale_3d_map(self, image, mode):
        # t, h, w, c = image.shape
        img = image.permute(0, 3, 1, 2)  # put height and width in front
        img = F.upsample(img, size=(self.new_h, self.new_w), mode=mode)
        img = img.permute(0, 2, 3, 1)  # move back
        return img
    
    def rescale_2d_map(self, image, mode):
        # t, h, w, c = image.shape
        img = image.permute(2, 0, 1).unsqueeze(0)
        # print("0", img.shape)
        img = F.upsample(img, size=(self.new_h, self.new_w), mode=mode)
        # print("1", img.shape)
        img = img.squeeze(0).squeeze(0)#.permute(1, 2, 0)  # move back
        # print("2", img.shape)
        return img


# 6
class TileDates(object):
    """
    Tile a 1d array to height (H), width (W) of an image.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, H, W, doy_bins=None):
        assert isinstance(H, (int,))
        assert isinstance(W, (int,))
        self.H = H
        self.W = W
        self.doy_bins = doy_bins

    def __call__(self, sample):
        sample['doy'] = self.repeat(sample['doy'], binned=self.doy_bins is not None)
        return sample
    
    def repeat(self, tensor, binned=False):
        if binned:
            out = tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.H, self.W, 1)#.permute(0, 2, 3, 1)
        else:
            out = tensor.repeat(1, self.H, self.W, 1).permute(3, 1, 2, 0)
        return out
    
    
# 7
class Concat(object):
    """
    Concat all inputs
    items in  : x10, x20, x60, day, year, labels
    items out : inputs, labels
    """
    def __init__(self, concat_keys):
        self.concat_keys = concat_keys
        
    def __call__(self, sample):
        try:
            inputs = torch.cat([sample[key] for key in self.concat_keys], dim=-1)
            sample["inputs"] = inputs
            sample = {key: sample[key] for key in sample.keys() if key not in self.concat_keys}
        except:
            print([("conc", key, sample[key].shape) for key in sample.keys()])
        return sample


# 8
class AddBackwardInputs(object):
    """
    random horizontal, vertical flip
    items in  : inputs, labels
    items out : inputs, inputs_backward, labels
    """
    def __call__(self, sample):
        sample['inputs_backward'] = torch.flip(sample['inputs'], (0,))
        return sample
    
    
# 9
class CutOrPad(object):
    """
    Pad series with zeros (matching series elements) to a max sequence length or cut sequential parts
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels, seq_lengths

    REMOVE DEEPCOPY OR REPLACE WITH TORCH FUN
    """

    def __init__(self, max_seq_len, random_sample=False, from_start=False):
        assert isinstance(max_seq_len, (int, tuple))
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample
        self.from_start = from_start
        assert int(random_sample) * int(from_start) == 0, "choose either one of random, from start sequence cut methods but not both"

    def __call__(self, sample):
        seq_len = deepcopy(sample['inputs'].shape[0])
        sample['inputs'] = self.pad_or_cut(sample['inputs'])
        if "inputs_backward" in sample:
            sample['inputs_backward'] = self.pad_or_cut(sample['inputs_backward'])
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
        sample['seq_lengths'] = seq_len
        return sample

    def pad_or_cut(self, tensor, dtype=torch.float32):
        seq_len = tensor.shape[0]
        diff = self.max_seq_len - seq_len
        if diff > 0:
            tsize = list(tensor.shape)
            if len(tsize) == 1:
                # t is series of scalars
                pad_shape = [diff]
            else:
                pad_shape = [diff] + tsize[1:]
            tensor = torch.cat((tensor, torch.zeros(pad_shape, dtype=dtype)), dim=0)
        elif diff < 0:
            if self.random_sample:
                return tensor[self.random_subseq(seq_len)]
            elif self.from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(seq_len - self.max_seq_len, (1,))[0]
            tensor = tensor[start_idx:start_idx+self.max_seq_len]
        return tensor
    
    def random_subseq(self, seq_len):
        return torch.randperm(seq_len)[:self.max_seq_len].sort()[0]


# 10
class HVFlip(object):
    """
    random horizontal, vertical flip
    items in  : inputs, *inputs_backward, labels
    items out : inputs, *inputs_backward, labels
    """
    
    def __init__(self, hflip_prob, vflip_prob, ground_truths=[]):
        assert isinstance(hflip_prob, (float,))
        assert isinstance(vflip_prob, (float,))
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.ground_truths = ground_truths
    
    def __call__(self, sample):
        if random.random() < self.hflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (2,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (2,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (1,))
        
        if random.random() < self.vflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (1,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (1,))
            for gt in self.ground_truths:
                sample[gt] = torch.flip(sample[gt], (0,))

        return sample


# 11
class Add2UnkClass(object):
    """
    Extract mask of unk classes in labels
    items in  : inputs, *inputs_backward, labels, seq_lengths
    items out : inputs, *inputs_backward, labels, seq_lengths, unk_masks
    """

    def __init__(self, unk_class, ground_truth_target, ground_truth_masks):
        assert isinstance(unk_class, (int,))
        self.unk_class = unk_class
        self.ground_truth_target = ground_truth_target
        self.ground_truth_masks = ground_truth_masks

    def __call__(self, sample):
        labels = sample[self.ground_truth_target]

        for gtm in self.ground_truth_masks:
            labels[sample[gtm].to(torch.bool).clone()] = self.unk_class
            del sample[gtm]

        sample[self.ground_truth_target] = labels.clone()

        return sample


# 11
class UnkMask(object):
    """
    Extract mask of unk classes in labels
    items in  : inputs, *inputs_backward, labels, seq_lengths
    items out : inputs, *inputs_backward, labels, seq_lengths, unk_masks
    """

    def __init__(self, unk_class, ground_truth_target):
        assert isinstance(unk_class, (int,))
        self.unk_class = unk_class
        self.ground_truth_target = ground_truth_target

    def __call__(self, sample):

        sample['unk_masks'] = (sample[self.ground_truth_target] != self.unk_class)

        if 'labels_grid' in sample.keys():
            sample['unk_masks_grid'] = self.rescale_2d_map(sample['unk_masks'].to(torch.float32), mode='nearest').to(
                torch.bool)

        return sample

    def rescale_2d_map(self, image, mode):
        # t, h, w, c = image.shape
        img = image.unsqueeze(0).permute(0, 3, 1, 2)  # permute(2, 0, 1). put height and width in front
        # print(img.shape)
        img = F.upsample(img, size=(self.num_grid, self.num_grid), mode=mode)
        img = img.squeeze(0).permute(1, 2, 0)  # move back
        return img


# 12
class AddBagOfLabels(object):
    """
    random horizontal, vertical flip
    items in  : inputs, labels
    items out : inputs, inputs_backward, labels
    """
    
    def __init__(self, n_class):
        self.n_class = n_class
    
    def __call__(self, sample):
        labels = sample['labels']
        bol = torch.zeros(self.n_class)
        bol[labels.unique().to(torch.long)] = 1.
        sample['bag_of_labels'] = bol
        return sample


# 13
class AddEdgeLabel(object):
    """
    Remap labels from original values to new consecutive integers
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    
    def __init__(self, nb_size=3, stride=1, pad_size=1, axes=[0, 1]):
        self.nb_size = nb_size
        self.stride = stride
        self.pad_size = pad_size
        self.axes = axes
    
    def __call__(self, sample):
        labels = sample['labels'].permute(2, 0, 1)[0]
        # print(labels.shape)
        edge_labels = self.get_edge_labels(labels)
        sample['edge_labels'] = edge_labels
        return sample
    
    def get_edge_labels(self, labels):
        lto = labels.to(torch.float32)
        H = lto.shape[self.axes[0]]
        W = lto.shape[self.axes[1]]
        lto = torch.nn.functional.pad(
            lto.unsqueeze(0).unsqueeze(0), [self.pad_size, self.pad_size, self.pad_size, self.pad_size], 'reflect')[
            0, 0]
        patches = lto.unfold(self.axes[0], self.nb_size, self.stride).unfold(self.axes[1], self.nb_size, self.stride)
        patches = patches.reshape(-1, self.nb_size ** 2)
        edge_map = (patches != patches[:, 0].unsqueeze(-1).repeat(1, self.nb_size ** 2)).any(dim=1).reshape(W, H).to(
            torch.bool)
        return edge_map


# 14
class EqualIntBoundPoints(object):
    """
    Update mask such that an equal number of interior and boundary points are used in loss
    """
    def __init__(self):
        self.extract_edges = AddEdgeLabel().get_edge_labels

    def __call__(self, sample):
        unk_masks = sample['unk_masks'][:, :, 0]
        if 'edge_labels' in sample.keys():
            edge_labels = sample['edge_labels']#[0]
        else:
            edge_labels = self.extract_edges(sample['labels'].permute(2, 0, 1)[0])
        kn_bound = unk_masks & edge_labels
        Nbound = kn_bound.sum()
        kn_int = unk_masks & (~edge_labels)
        Nint = kn_int.sum()
        if Nint > Nbound:
            kn_int_ = kn_int.reshape(-1)
            kn_int_idx = torch.arange(0, kn_int_.shape[0])[kn_int_]
            kn_int_idx = kn_int_idx[torch.randperm(kn_int_idx.shape[0])][:Nbound]
            mask_int = torch.zeros(kn_int_.shape[0])
            mask_int[kn_int_idx] = 1.0
            mask_int = mask_int.reshape(kn_int.shape)
            mask_bound = kn_bound
        else:
            kn_bound_ = kn_bound.reshape(-1)
            kn_bound_idx = torch.arange(0, kn_bound_.shape[0])[kn_bound_]
            kn_bound_idx = kn_bound_idx[torch.randperm(kn_bound_idx.shape[0])][:Nint]
            mask_bound = torch.zeros(kn_bound_.shape[0])
            mask_bound[kn_bound_idx] = 1.0
            mask_bound = mask_bound.reshape(kn_bound.shape)
            mask_int = kn_int
        mask = mask_int + mask_bound
        sample['unk_masks'] = mask.to(torch.bool)

        return sample


# 14
class AddCSCLLabels(object):
    """
    Remap labels from original values to new consecutive integers
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, kernel_size=None, kernel_stride=1, kernel_dilation=1, pad_value=100, add_mask=False):
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_dilation = kernel_dilation
        self.pad_value = pad_value

        if self.kernel_stride >= self.kernel_dilation:
            assert self.kernel_stride % self.kernel_dilation == 0, \
                "CSSL stride should be integer multiple of dilation rate for special case to work"
            self.first_stride = self.kernel_dilation
            self.final_stride = int(self.kernel_stride / self.kernel_dilation)
            self.final_dilation = 1
            self.final_kernel_size = self.kernel_size
        elif self.kernel_stride < self.kernel_dilation:
            assert self.kernel_dilation % self.kernel_stride == 0, \
                "CSSL dilation should be integer multiple of stride for special case to work"
            self.first_stride = self.kernel_stride
            self.final_stride = 1
            self.final_dilation = int(self.kernel_dilation / self.kernel_stride)
            self.final_kernel_size = (self.kernel_size - 1) * self.final_dilation + 1

        self.center_idx = kernel_size // 2
        self.add_mask = add_mask
        self.dilated_kernel_size = (kernel_size - 1) * kernel_dilation + 1
        print("dilated win size: ", self.dilated_kernel_size)
        self.pad_size = self.dilated_kernel_size // 2
    
    def __call__(self, sample):
        ### ALSO unfold unk masks to mask SAL
        # https://en.wikipedia.org/wiki/Logical_matrix
        labels = sample['labels'].permute(2, 0, 1)[0][::self.first_stride, ::self.first_stride]

        labels = F.pad(labels, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=self.pad_value)
        windows = labels.unfold(0, self.dilated_kernel_size, self.kernel_stride).unfold(
            1, self.dilated_kernel_size, self.kernel_stride)[:, :, ::self.final_dilation, ::self.final_dilation]

        h1, w1, h2, w2 = windows.shape
        windows_q = windows[:, :, self.center_idx, self.center_idx].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h2 * w2)
        windows_k = windows.reshape(h1, w1, 1, h2 * w2)
        is_same = (windows_q == windows_k).to(torch.float32).reshape(h1, w1, h2, w2)
        sample['cscl_labels'] = is_same

        if self.add_mask:
            mask = torch.ones(windows.shape)
            mask[windows == self.pad_value] = 0.
            mask[windows_q[:, :, 0].reshape(h1, w1, h2, w2) == self.pad_value] = 0.
            mask[:, :, self.center_idx, self.center_idx] = 0.
            background_mask_self = sample['unk_masks'].clone().repeat(1, 1, self.kernel_size**2).reshape(h1, w1, h2, w2)
            background_mask_other = F.pad(sample['unk_masks'].clone().squeeze(-1), (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=False)#.unsqueeze(-1)
            background_mask_other = background_mask_other.unfold(0, self.kernel_size, self.kernel_stride).unfold(1, self.kernel_size, self.kernel_stride)
            sample['cscl_labels_mask'] = mask.to(torch.bool) & background_mask_self & background_mask_other

        return sample


class AddConstantYear(object):
    """
    Concat all inputs
    items in  : x10, x20, x60, day, year, labels
    items out : inputs, labels
    """

    def __init__(self, year):
        self.year = year

    def __call__(self, sample):
        inputs = sample["inputs"].clone()
        t, h, w, c = inputs.shape
        sample["inputs"] = torch.cat([inputs, self.year * torch.ones((t, h, w, 1))], dim=-1)
        return sample


class UpdateIds(object):
    """
    Remap ids instances to relative instead of global numbers
    """
    def __call__(self, sample):
        ids = sample['ids']
        uids_dict = {v: i for i, v in enumerate(ids.unique())}
        not_remapped = torch.ones(ids.shape, dtype=torch.bool)
        for id_ in uids_dict:
            id_idx = ids == id_
            remap_idx = id_idx & not_remapped
            ids[remap_idx] = uids_dict[id_]
            not_remapped[remap_idx] = False
        sample['ids'] = ids
        return sample


class SOLOGroundTruths(object):
    """
    Remap ids instances to relative instead of global numbers
    """

    def __init__(self, num_grid, label_res, unk_class):
        # assert isinstance(labels_dict, (dict,))
        self.num_grid = num_grid
        self.label_res = label_res
        self.unk_class = unk_class
        self.sigma = 0.2

    def __call__(self, sample):
        labels = sample['labels']
        ids = sample['ids']

        uids = ids.unique()[1:]  # no background
        ids_mask = torch.zeros(torch.Size([self.num_grid**2]) + ids.shape, dtype=torch.float32)  # [:2])
        cate_label = self.unk_class * torch.ones([self.num_grid, self.num_grid], dtype=torch.int64)

        for ii, id_ in enumerate(uids):

            mask = torch.zeros(ids.shape, dtype=torch.uint8)  # [:2])
            mask[ids == id_] = 1

            center_h, center_w = ndimage.measurements.center_of_mass(mask[:, :, 0].numpy())

            coord_w = int((center_w / ids.shape[1]) // (1. / self.num_grid))
            coord_h = int((center_h / ids.shape[0]) // (1. / self.num_grid))

            objs = ndimage.find_objects(mask[:, :, 0].numpy())
            top_bbox = objs[0][0].start
            bottom_bbox = objs[0][0].stop
            left_bbox = objs[0][1].start
            right_bbox = objs[0][1].stop
            center_h_bbox = (top_bbox + bottom_bbox) / 2
            center_w_bbox = (left_bbox + right_bbox) / 2
            dh = bottom_bbox - center_h_bbox
            dw = right_bbox - center_w_bbox
            # Note: currently this assumes a very simple center mass region (cm +- 1pix), test how to expand this
            # top = max(int(objs[0][0].start * self.num_grid / self.label_res), coord_h - 1)  # MAKE THIS WORK WITH sigma=0.2
            # down = min(int(objs[0][0].stop * self.num_grid / self.label_res), coord_h + 1)
            # left = max(int(objs[0][1].start * self.num_grid / self.label_res), coord_w - 1)
            # right = min(int(objs[0][1].stop * self.num_grid / self.label_res), coord_w + 1)
            top = max(int(round((center_h - self.sigma * dh) * self.num_grid / self.label_res)), coord_h - 1)
            down = min(int(round((center_h + self.sigma * dh) * self.num_grid / self.label_res)), coord_h + 1)
            left = max(int(round((center_w - self.sigma * dw) * self.num_grid / self.label_res)), coord_w - 1)
            right = min(int(round((center_w + self.sigma * dw) * self.num_grid / self.label_res)), coord_w + 1)

            cate_label[top:(down + 1), left:(right + 1)] = labels[int(round(center_h)), int(round(center_w))]

            for i in range(top, down):
                for j in range(left, right):
                    # print(i * self.num_grid + j)
                    ids_mask[i * self.num_grid + j] = mask

        sample['ids_masks'] = ids_mask
        ids_ind_masks = torch.zeros(self.num_grid**2).to(torch.bool)
        ids_ind_masks[torch.arange(self.num_grid**2)[ids_mask.sum(dim=(1, 2, 3)) > 0]] = True
        sample['ids_ind_masks'] = ids_ind_masks
        sample['labels_grid'] = cate_label.unsqueeze(-1)  #.to(torch.int64)

        return sample

