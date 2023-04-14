import torch
from data.MTLCC.dataloader import get_dataloader as get_mtlcc_dataloader
from data.MTLCC.data_transforms import MTLCC_transform
from data.France.dataloader import get_dataloader as get_france_dataloader
from data.France.data_transforms import France_segmentation_transform
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from utils.config_files_utils import get_params_values, read_yaml


DATASET_INFO = read_yaml("data/datasets.yaml")


def get_dataloaders(config):


    model_config = config['MODEL']
    train_config = config['DATASETS']['train']
    train_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    dataloaders = {}
    
    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config['base_dir'] = DATASET_INFO[train_config['dataset']]['basedir']
    train_config['paths'] = DATASET_INFO[train_config['dataset']]['paths_train']
    if train_config['dataset'] == 'MTLCC':
        dataloaders['train'] = get_mtlcc_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=MTLCC_transform(model_config, train_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
    elif 'PASTIS' in train_config['dataset']:
        dataloaders['train'] = get_pastis_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
    else:
        dataloaders['train'] = get_france_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=France_segmentation_transform(model_config, train_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])

    # EVAL data --------------------------------------------------------------------------------------------------------
    eval_config['base_dir'] = DATASET_INFO[eval_config['dataset']]['basedir']
    eval_config['paths'] = DATASET_INFO[eval_config['dataset']]['paths_eval']
    if eval_config['dataset'] == 'MTLCC':
        dataloaders['eval'] = get_mtlcc_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=MTLCC_transform(model_config, eval_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'])
    elif 'PASTIS' in eval_config['dataset']:
        dataloaders['eval'] = get_pastis_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'])
    else:
        dataloaders['eval'] = get_france_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=France_segmentation_transform(model_config, eval_config, is_training=True),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'])

    return dataloaders


def get_model_data_input(config):
    
    def unidir_segmentation_inputs(sample, device):
        inputs = sample['inputs'].to(device)
        return inputs
    
    def bidir_segmentation_inputs(sample, device):
        inputs = sample['inputs'].to(device)
        inputs_backward = sample['inputs_backward'].to(device)
        seq_lengths = sample['seq_lengths'].to(device)
        return inputs, inputs_backward, seq_lengths
    
    model = config['MODEL']['architecture']

    if model in ['ConvBiRNN']:
        return bidir_segmentation_inputs
    
    if model in ['UNET3D', 'UNET3Df', 'UNET2D-CLSTM']:
        return unidir_segmentation_inputs


def get_loss_data_input(config):

    def segmentation_ground_truths(sample, device):
        labels = sample['labels'].to(device)
        if 'unk_masks' in sample.keys():
            unk_masks = sample['unk_masks'].to(device)
        else:
            unk_masks = None

        if 'edge_labels' in sample.keys():
            edge_labels = sample['edge_labels'].to(device)
            return labels, edge_labels, unk_masks
        return labels, unk_masks

    def cscl_ground_truths(sample, device, return_masks=False):
        labels = sample['cscl_labels'].to(device)
        if return_masks:
            masks = sample['cscl_labels_mask'].to(device)
            if 'edge_locs' in sample:
                wh, ww = masks.shape[-2:]
                masks = (sample['edge_locs'].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, wh, ww) * \
                         sample['cscl_labels_mask'].to(torch.float32)).to(torch.bool).to(device)
            return labels, masks
        return labels

    loss_fn = config['SOLVER']['loss_function']
    stage = get_params_values(config['MODEL'], 'train_stage', 2)

    if config['MODEL']['architecture'] in ['UNET3Df', 'UNET2D-CLSTM']:
        if stage in [0, 4]:
            if loss_fn in ["binary_cross_entropy", "binary_focal_loss", "contrastive_loss"]:
                if stage == 0:
                    return cscl_ground_truths

            if loss_fn in ["masked_binary_cross_entropy", "masked_binary_focal_loss", "masked_contrastive_loss"]:
                if stage == 0:
                    return lambda sample, device: cscl_ground_truths(sample, device, return_masks=True)

    return segmentation_ground_truths
