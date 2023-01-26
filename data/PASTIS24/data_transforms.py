from __future__ import print_function, division
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy
import random
from utils.config_files_utils import get_params_values
from scipy import ndimage


def PASTIS_segmentation_transform(model_config, is_training):
    """
    """
    dataset_img_res = 24
    input_img_res = model_config['img_res']
    ground_truths = ['labels']
    max_seq_len = model_config['max_seq_len']
    inputs_backward = get_params_values(model_config, 'inputs_backward', False)
    transform_list = []
    transform_list.append(ToTensor())                                  # data from numpy arrays to torch.float32
    transform_list.append(Normalize())                                 # normalize all inputs individually

    if dataset_img_res != input_img_res:
        transform_list.append(
            Crop(img_size=dataset_img_res, crop_size=input_img_res, random=is_training, ground_truths=ground_truths))  # random crop

    transform_list.append(TileDates(H=model_config['img_res'], W=model_config['img_res'], doy_bins=None))                       # tile day and year to shape TxWxHx1
    transform_list.append(CutOrPad(max_seq_len=max_seq_len, random_sample=False, from_start=True))  # pad with zeros to maximum sequence length
    transform_list.append(UnkMask(unk_class=19, ground_truth_target='labels'))  # extract unknown label mask

    if inputs_backward:
        transform_list.append(AddBackwardInputs())

    transform_list.append(ToTHWC())

    return transforms.Compose(transform_list)


class ToTHWC(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, sample):
        sample['inputs'] = sample['inputs'].permute(0, 2, 3, 1)
        return sample


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
        tensor_sample['inputs'] = torch.tensor(sample['img']).to(torch.float32)
        tensor_sample['labels'] = torch.tensor(sample['labels'][0].astype(np.float32)).to(torch.float32).unsqueeze(-1)
        tensor_sample['doy'] = torch.tensor(np.array(sample['doy'])).to(torch.float32)
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
    def __init__(self):
        self.mean_fold1 = np.array([[[[1165.9398193359375]],
                                   [[1375.6534423828125]],
                                   [[1429.2191162109375]],
                                   [[1764.798828125]],
                                   [[2719.273193359375]],
                                   [[3063.61181640625]],
                                   [[3205.90185546875]],
                                   [[3319.109619140625]],
                                   [[2422.904296875]],
                                   [[1639.370361328125]]]]).astype(np.float32)
        self.std_fold1 = np.array([[[[1942.6156005859375]],
                                  [[1881.9234619140625]],
                                  [[1959.3798828125]],
                                  [[1867.2239990234375]],
                                  [[1754.5850830078125]],
                                  [[1769.4046630859375]],
                                  [[1784.860595703125]],
                                  [[1767.7100830078125]],
                                  [[1458.963623046875]],
                                  [[1299.2833251953125]]]]).astype(np.float32)
    def __call__(self, sample):
        # print('mean: ', sample['img'].mean(dim=(0,2,3)))
        # print('std : ', sample['img'].std(dim=(0,2,3)))
        sample['inputs'] = (sample['inputs'] - self.mean_fold1) / self.std_fold1
        sample['doy'] = sample['doy'] / 365.0001
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
        sample['inputs'] = sample['inputs'][:, :, top:top + self.crop_size, left:left + self.crop_size]
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
        if not self.rescale_gt_only:
            for inputc in ['x20', 'x60']:  # 'x10',
                sample[inputc] = self.rescale_3d_map(sample[inputc], mode='bilinear')
        for gt in self.ground_truths:
            sample[gt] = self.rescale_2d_map(sample[gt], mode='nearest')

        return sample

    def rescale_3d_map(self, image, mode):
        img = image.permute(0, 3, 1, 2)  # put height and width in front
        img = F.upsample(img, size=(self.new_h, self.new_w), mode=mode)
        img = img.permute(0, 2, 3, 1)  # move back
        return img
    
    def rescale_2d_map(self, image, mode):
        img = image.permute(2, 0, 1).unsqueeze(0)
        img = F.upsample(img, size=(self.new_h, self.new_w), mode=mode)
        img = img.squeeze(0).squeeze(0)
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
        doy = self.repeat(sample['doy'], binned=self.doy_bins is not None)
        sample['inputs'] = torch.cat((sample['inputs'], doy), dim=1)
        del sample['doy']
        return sample
    
    def repeat(self, tensor, binned=False):
        if binned:
            out = tensor.unsqueeze(1).unsqueeze(1).repeat(1, self.H, self.W, 1)#.permute(0, 2, 3, 1)
        else:
            out = tensor.repeat(1, self.H, self.W, 1).permute(3, 0, 1, 2)
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
        sample['unk_masks'] = (sample[self.ground_truth_target] != self.unk_class) #& \
        if 'labels_grid' in sample.keys():
            sample['unk_masks_grid'] = self.rescale_2d_map(sample['unk_masks'].to(torch.float32), mode='nearest').to(
                torch.bool)

        return sample

    def rescale_2d_map(self, image, mode):
        img = image.unsqueeze(0).permute(0, 3, 1, 2)  # permute(2, 0, 1). put height and width in front
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
