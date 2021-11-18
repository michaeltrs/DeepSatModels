import torch
import torch.nn.functional as F


def resize_match2d(target_size, source, dim=[2, 3], mode='bilinear'):
    """
    source must have shape [..., H, W]
    :param mode: 'nearest'
    """
    target_h, target_w = target_size
    source_h, source_w = source.shape[dim[0]], source.shape[dim[1]]
    if (source_h != target_h) or (source_w != target_w):
        source_type = source.dtype
        if source_type != torch.float32:
            source = source.to(torch.float32)
            return F.interpolate(source, size=(target_h, target_w), mode=mode).to(source_type)
        return F.interpolate(source, size=(target_h, target_w), mode=mode)
    return source
