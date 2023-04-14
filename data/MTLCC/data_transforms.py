from __future__ import print_function, division
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from copy import deepcopy
import random
from utils.config_files_utils import get_params_values


original_label_dict = {0: "unknown", 1: "sugar_beet", 2: "summer_oat", 3: "meadow", 5: "rape", 8: "hop",
                       9: "winter_spelt", 12: "winter_triticale", 13: "beans", 15: "peas", 16: "potatoes",
                       17: "soybeans", 19: "asparagus", 22: "winter_wheat", 23: "winter_barley", 24: "winter_rye",
                       25: "summer_barley", 26: "maize"}
remap_label_dict = {0: 17,  1: 0,  2: 1,  3: 2,  5: 3,  8: 4,  9: 5, 12: 6, 13: 7, 15: 8,
                    16: 9, 17: 10, 19: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16}


def get_label_names():
    names = {}
    for label in original_label_dict:
        names[remap_label_dict[label]] = original_label_dict[label]
    return names

    
def MTLCC_transform(model_config, data_config, is_training):
    """
    :param npixel:
    :return:
    """
    img_res = model_config['img_res']
    max_seq_len = data_config['max_seq_len']
    extra_data = get_params_values(data_config, 'extra_data', [])
    doy_bins = get_params_values(data_config, 'doy_bins', None)
    bidir_input = data_config['bidir_input']
    equal_int_bound = get_params_values(data_config, 'equal_int_bound', False)

    transform_list = []
    transform_list.append(ToTensor())                                  # data from numpy arrays to torch.float32
    transform_list.append(SingleLabel())                               # extract single label from series
    transform_list.append(RemapLabel(remap_label_dict))                      # remap labels to new values
    transform_list.append(Normalize())                                 # normalize all inputs individually
    transform_list.append(Rescale(output_size=(img_res, img_res)))               # scale x20, x60 to match x10 H, W
    if doy_bins is not None:
        transform_list.append(OneHotDates(N=doy_bins))
    transform_list.append(TileDates(H=img_res, W=img_res, doy_bins=doy_bins))                       # tile day and year to shape TxWxHx1
    transform_list.append(Concat(concat_keys=['x10', 'x20', 'x60', 'day', 'year']))     # concat x10, x20, x60, day, year
    if bidir_input:
        transform_list.append(AddBackwardInputs())                         # add input series in reverse for bidir models
    transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=True))             # pad with zeros to maximum sequence length
    if is_training:
        transform_list.append(HVFlip(hflip_prob=0.5, vflip_prob=0.5))  # horizontal, vertical flip
    transform_list.append(UnkMask(unk_class=17))                       # extract unknown label mask
    # transform_list.append(AddBagOfLabels(n_class=n_class))
    if 'edge_labels' in extra_data:
        transform_list.append(AddEdgeLabel())
    if equal_int_bound:
        transform_list.append(EqualIntBoundPoints())
    if 'cscl_labels' in extra_data:
        cscl_win_size = model_config['cscl_win_size']
        cscl_win_stride = model_config['cscl_win_stride']
        cscl_win_dilation = model_config['cscl_win_dilation']
        transform_list.append(AddCSCLLabels(
            kernel_size=cscl_win_size, kernel_stride=cscl_win_stride, kernel_dilation=cscl_win_dilation,
            pad_value=max(list(remap_label_dict.values())) + 1))
    return transforms.Compose(transform_list)


# 1
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        if 'B01' in sample.keys():
            x10 = torch.stack([torch.tensor(sample[key].astype(np.float32)) for key in ['B04', 'B03', 'B02', 'B08']]).permute(1, 2, 3, 0)
            x20 = torch.stack([torch.tensor(sample[key].astype(np.float32)).type(torch.float32) for key in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).permute(1, 2, 3, 0)
            x60 = torch.stack([torch.tensor(sample[key].astype(np.float32)) for key in ['B01', 'B09', 'B10']]).permute(1, 2, 3, 0)
            doy = torch.tensor(np.array(sample['doy']).astype(np.float32))
            year = torch.tensor(0.).repeat(len(sample['doy'])) + 2016
            labels = torch.tensor(sample['labels'].astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=-1)
            sample = {"x10": x10, "x20": x20, "x60": x60, "day": doy, "year": year, "labels": labels}
            return sample
        # else:
        sample['x10'] = torch.tensor(sample['x10']).type(torch.float32)
        sample['x20'] = torch.tensor(sample['x20']).type(torch.float32)
        sample['x60'] = torch.tensor(sample['x60']).type(torch.float32)
        sample['day'] = torch.tensor(sample['day']).type(torch.float32)
        sample['year'] = torch.tensor(sample['year']).type(torch.float32)
        sample['labels'] = torch.unsqueeze(
            torch.from_numpy(sample['labels'].astype(np.int64)),
            dim=-1)
        return sample


# 2
class SingleLabel(object):
    """
    Extract and use only single label from series assuming labels are repeated
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __init__(self, idx=0):
        assert isinstance(idx, (int, tuple))
        self.idx = idx
        
    def __call__(self, sample):
        sample['labels'] = sample['labels'][self.idx]
        return sample


# 3
class RemapLabel(object):
    """
    Remap labels from original values to new consecutive integers
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    
    def __init__(self, labels_dict):
        assert isinstance(labels_dict, (dict,))
        self.labels_dict = labels_dict
    
    def __call__(self, sample):
        labels = sample['labels']
        not_remapped = torch.ones(labels.shape, dtype=torch.bool)
        for label_ in self.labels_dict:
            label_idx = labels == label_
            remap_idx = label_idx & not_remapped
            labels[remap_idx] = self.labels_dict[label_]
            not_remapped[remap_idx] = False
        sample['labels'] = labels
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
        sample['day'] = sample['day'] / 365.0001  # 365 + h, h = 0.0001 to avoid placing day 365 in out of bounds bin
        sample['year'] = sample['year'] - 2016
        return sample


# 5
class Rescale(object):
    """
    Rescale the image in a sample to a given square side
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple,))
        self.new_h, self.new_w = output_size

    def __call__(self, sample):
        sample['x10'] = self.rescale(sample['x10'])
        sample['x20'] = self.rescale(sample['x20'])
        sample['x60'] = self.rescale(sample['x60'])
        return sample

    def rescale(self, image):
        img = image.permute(0, 3, 1, 2)  # put height and width in front
        img = F.upsample(img, size=(self.new_h, self.new_w), mode='bilinear')
        img = img.permute(0, 2, 3, 1)  # move back
        return img


# 5.5
class OneHotDates(object):
    """
    Tile a 1d array to height (H), width (W) of an image.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, N):
        assert isinstance(N, (int,))
        self.N = N

    def __call__(self, sample):
        sample['day'] = self.doy_to_bin(sample['day'])
        return sample
    
    def doy_to_bin(self, doy):
        """
        assuming doy = day / 365, float in [0., 1.]
        """
        bin_id = (doy // (1. / (self.N - 1))).to(torch.int64)
        out = torch.zeros(bin_id.shape[0], self.N)
        out[torch.arange(0, bin_id.shape[0]), bin_id] = 1.
        return out
    
    
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
        sample['day'] = self.repeat(sample['day'], binned=self.doy_bins is not None)
        sample['year'] = self.repeat(sample['year'], binned=False)
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
        inputs = torch.cat([sample[key] for key in self.concat_keys], dim=-1)
        sample["inputs"] = inputs
        sample = {key: sample[key] for key in sample.keys() if key not in self.concat_keys}
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

    def __init__(self, max_seq_len, random_sample=False):
        assert isinstance(max_seq_len, (int, tuple))
        self.max_seq_len = max_seq_len
        self.random_sample = random_sample

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
                tensor = tensor[self.random_subseq(seq_len)]
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
    
    def __init__(self, hflip_prob, vflip_prob):
        assert isinstance(hflip_prob, (float,))
        assert isinstance(vflip_prob, (float,))
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
    
    def __call__(self, sample):
        if random.random() < self.hflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (2,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (2,))
            sample['labels'] = torch.flip(sample['labels'], (2,))
            if 'edge_labels' in sample.keys():
                sample['edge_labels'] = torch.flip(sample['edge_labels'], (2,))
        
        if random.random() < self.vflip_prob:
            sample['inputs'] = torch.flip(sample['inputs'], (1,))
            if "inputs_backward" in sample:
                sample['inputs_backward'] = torch.flip(sample['inputs_backward'], (1,))
            sample['labels'] = torch.flip(sample['labels'], (1,))
            if 'edge_labels' in sample.keys():
                sample['edge_labels'] = torch.flip(sample['edge_labels'], (1,))
        
        return sample


# 11
class UnkMask(object):
    """
    Extract mask of unk classes in labels
    items in  : inputs, *inputs_backward, labels, seq_lengths
    items out : inputs, *inputs_backward, labels, seq_lengths, unk_masks
    """

    def __init__(self, unk_class):
        assert isinstance(unk_class, (int,))
        self.unk_class = unk_class

    def __call__(self, sample):
        sample['unk_masks'] = sample['labels'] != self.unk_class
        return sample


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
    def __init__(self):  # , N=None):
        # self.N = N
        self.extract_edges = AddEdgeLabel().get_edge_labels

    def __call__(self, sample):
        # print(labels.shape)
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


class AddCSCLLabels(object):
    """
    Ground-truth generator module section  3.2
    """

    def __init__(self, kernel_size=None, kernel_stride=1, kernel_dilation=1, pad_value=100, add_mask=False):
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_dilation = kernel_dilation
        self.pad_value = pad_value

        if self.kernel_stride >= self.kernel_dilation:
            assert self.kernel_stride % self.kernel_dilation == 0, \
                "CSCL stride should be integer multiple of dilation rate for special case to work"
            self.first_stride = self.kernel_dilation
            self.final_stride = int(self.kernel_stride / self.kernel_dilation)
            self.final_dilation = 1
            self.final_kernel_size = self.kernel_size
        elif self.kernel_stride < self.kernel_dilation:
            assert self.kernel_dilation % self.kernel_stride == 0, \
                "CSCL dilation should be integer multiple of stride for special case to work"
            self.first_stride = self.kernel_stride
            self.final_stride = 1
            self.final_dilation = int(self.kernel_dilation / self.kernel_stride)
            self.final_kernel_size = (self.kernel_size - 1) * self.final_dilation + 1

        self.center_idx = kernel_size // 2
        self.add_mask = add_mask
        self.pad_size = self.final_kernel_size // 2

    def __call__(self, sample):
        labels = sample['labels'].permute(2, 0, 1)[0][::self.first_stride, ::self.first_stride]

        labels = F.pad(labels, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=self.pad_value)
        windows = labels.unfold(0, self.final_kernel_size, self.final_stride).unfold(
            1, self.final_kernel_size, self.final_stride)[:, :, ::self.final_dilation, ::self.final_dilation]

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
            background_mask_self = sample['unk_masks'].clone().\
                repeat(1, 1, self.kernel_size ** 2).reshape(h1, w1, h2, w2)
            background_mask_other = F.pad(sample['unk_masks'].clone().squeeze(-1),
                                          (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                                          value=False)
            background_mask_other = background_mask_other.\
                unfold(0, self.kernel_size, self.kernel_stride).unfold(1, self.kernel_size, self.kernel_stride)
            sample['cscl_labels_mask'] = mask.to(torch.bool) & background_mask_self & background_mask_other

        return sample


class AddCSSLLabels(object):
    """
    Remap labels from original values to new consecutive integers
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """

    def __init__(self, global_attn=False, win_size=None, win_stride=1, pad_size=1, pad_value=100):
        self.global_attn = global_attn
        self.win_size = win_size
        self.win_stride = win_stride
        self.pad_size = pad_size
        self.pad_value = pad_value
        self.center_idx = win_size // 2

    def __call__(self, sample):
        ### ALSO unfold unk masks to mask SAL
        # https://en.wikipedia.org/wiki/Logical_matrix
        labels = sample['labels'].permute(2, 0, 1)[0]
        h0, w0 = labels.shape
        # print(labels.shape)
        if self.global_attn:
            assert self.win_size == h0 == w0, "window size should equal tensor dimensions"
            labels = labels.reshape(-1, 1)
            is_same = (labels.transpose(1, 0).repeat(h0 * w0, 1) == labels.repeat(1, h0 * w0)) \
                .reshape(h0, w0, h0, w0).to(torch.float32)
        else:
            labels = F.pad(labels, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), value=self.pad_value)
            windows = labels.unfold(0, self.win_size, self.win_stride).unfold(1, self.win_size, self.win_stride)
            h1, w1, h2, w2 = windows.shape
            windows_q = windows[:, :, self.center_idx, self.center_idx].unsqueeze(-1).unsqueeze(-1)
            windows_k = windows.reshape(h1, w1, 1, h2 * w2)
            is_same = (windows_q.repeat(1, 1, 1, h2 * w2) == windows_k).to(torch.float32).reshape(h1, w1, h2, w2)

        sample['sameclass_labels'] = is_same

        return sample
