import torch 
import torch.nn as nn

from models.CropTypeMapping.constants import *

def set_parameter_requires_grad(model, fix_feats):
    if fix_feats:
        for param in model.parameters():
            param.requires_grad = False

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

                
def get_num_bands(kwargs):
    num_bands = 0
    added_doy = 0
    added_clouds = 0
    added_indices = 0

    if kwargs.get('include_doy'):
        added_doy = 1
    if kwargs.get('include_clouds') and kwargs.get('use_s2'): 
        added_clouds = 1
    if kwargs.get('include_indices') and (kwargs.get('use_s2') or kwargs.get('use_planet')):
        added_indices = 2

    num_bands = {'s1': 0, 's2': 0, 'planet': 0 }

    if kwargs.get('use_s1'):
        num_bands['s1'] = S1_NUM_BANDS + added_doy
    if kwargs.get('use_s2'):
        num_bands['s2'] = kwargs.get('s2_num_bands') + added_doy + added_clouds + added_indices
    if kwargs.get('use_planet'):
        num_bands['planet'] = PLANET_NUM_BANDS + added_doy + added_indices
   
    num_bands['all'] = num_bands['s1'] + num_bands['s2'] + num_bands['planet'] 
    return num_bands


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling for FCN"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
