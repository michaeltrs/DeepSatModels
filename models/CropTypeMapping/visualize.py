"""
import random

File for visualizing model performance.

"""

import numpy as np
import os 
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import visdom
from torchvision.utils import save_image, make_grid

import metrics
import preprocess
import util
from constants import * 

class VisdomLogger:
    def __init__(self, env_name, model_name, country, splits, port=8097):
        env_name = model_name if env_name is None else env_name
        self.vis = visdom.Visdom(port=port, env=env_name)
        self.country = country
        
        self.splits = splits
        self._init_progress_data()
        self._init_epoch_data()
        
        
    def _init_progress_data(self):
        # stores information across epochs
        self.progress_data = {}
        for split in self.splits:
            self.progress_data[f'{split}_loss'] = []
            self.progress_data[f'{split}_acc'] = []
            self.progress_data[f'{split}_f1'] = []
            self.progress_data[f'{split}_classf1'] = None    
        self.progress_data['train_gradnorm'] = []
        
    def _init_epoch_data(self):
        # stores information per epoch
        self.epoch_data = {}
        for split in self.splits:
            self.epoch_data[f'{split}_loss'] = 0
            self.epoch_data[f'{split}_correct'] = 0
            self.epoch_data[f'{split}_pix'] = 0
            self.epoch_data[f'{split}_cm'] = np.zeros((NUM_CLASSES[self.country], NUM_CLASSES[self.country])).astype(int)
            
    def update_progress(self, split, metric_name, value):
        self.progress_data[f'{split}_{metric_name}'].append(value)
    
    def update_epoch_all(self, split, cm_cur, loss, total_correct, num_pixels):
        self.epoch_data[f'{split}_cm'] += cm_cur
        self.epoch_data[f'{split}_loss'] += loss.item()
        self.epoch_data[f'{split}_correct'] += total_correct
        self.epoch_data[f'{split}_pix'] += num_pixels
    
    def reset_epoch_data(self):
        self.epoch_data = {}
        for split in self.splits:
            self.epoch_data[f'{split}_loss'] = 0
            self.epoch_data[f'{split}_correct'] = 0
            self.epoch_data[f'{split}_pix'] = 0
            self.epoch_data[f'{split}_cm'] = np.zeros((NUM_CLASSES[self.country], NUM_CLASSES[self.country])).astype(int)
    
    def record_batch(self, inputs, clouds, targets, preds, confidence, 
                     num_classes, split, include_doy, use_s1, use_s2, 
                     model_name, time_slice, 
                     save=False, save_dir=None, show_visdom=True, show_matplot=False, var_length=False):
        
        label_mask = np.sum(targets.numpy(), axis=1)
        label_mask = np.expand_dims(label_mask, axis=1)
        if show_visdom:
            visdom_plot_images(self.vis, label_mask, 'Label Masks')
            #visdom_plot_images(vis, confidence, 'Confidence')

        # TODO: not sure if this is doing anything (since best is set tp zeros_like)
        # Show best inputs judging from cloud masks
        if clouds is not None and torch.sum(clouds) != 0 and len(clouds.shape) > 1: 
            best = np.argmax(np.mean(np.mean(clouds.numpy()[:, 0, :, :, :], axis=1), axis=1), axis=1)
        else:
            if var_length and 's2' in inputs:
                best = np.random.randint(0, high=inputs['s2'].shape[1], size=(inputs['s2'].shape[0],))
            elif var_length and 'planet' in inputs:
                best = np.random.randint(0, high=inputs['planet'].shape[1], size=(inputs['planet'].shape[0],))
            elif var_length and 's1' in inputs:
                best = np.random.randint(0, high=inputs['s1'].shape[1], size=(inputs['s1'].shape[0],))
            else:
                best = np.random.randint(0, high=inputs.shape[1], size=(inputs.shape[0],))
        best = np.zeros_like(best)

        # Get bands of interest (boi) to show best rgb version of s2 or vv, vh, vv version of s1
        boi = []
        add_doy = 1 if use_s2 and use_s1 and include_doy else 0
        # TODO: change these to be constants in constants.py eventually
        start_idx = 2 if use_s2 and use_s1 else 0
        end_idx = 5 if use_s2 and use_s1 else 3
        if model_name in ['fcn_crnn', 'bidir_clstm','unet3d', 'mi_clstm', 'only_clstm_mi']:
            for idx, b in enumerate(best):
                if var_length and 's2' in inputs:
                    boi.append(inputs['s2'][idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
                elif var_length and 'planet' in inputs:
                    boi.append(inputs['planet'][idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
                elif var_length and 's1' in inputs:
                    boi.append(inputs['s1'][idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
                else:
                    boi.append(inputs[idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
            boi = torch.cat(boi, dim=0)
        elif model_name in ['fcn', 'unet'] and time_slice is not None:
            boi = inputs[:, start_idx+add_doy:end_idx+add_doy, :, :]
        elif model_name in ['unet'] and time_slice is None:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1, inputs.shape[2], inputs.shape[3])  
            for idx, b in enumerate(best):
                boi.append(inputs[idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
            boi = torch.cat(boi, dim=0)
        else:
            raise ValueError(f"Model {model_name} unsupported! check --model_name args")

        # Clip and show input bands of interest
        boi = clip_boi(boi)
        if show_visdom:
            visdom_plot_images(self.vis, boi, 'Input Images') 

        # Show targets (labels)
        disp_targets = np.concatenate((np.zeros_like(label_mask), targets.numpy()), axis=1)
        disp_targets = np.argmax(disp_targets, axis=1)
        disp_targets = np.expand_dims(disp_targets, axis=1)
        disp_targets = visualize_rgb(disp_targets, num_classes)
        if show_visdom:
            visdom_plot_images(self.vis, disp_targets, 'Target Images')

        # Show predictions, masked with label mask
        disp_preds = np.argmax(preds.detach().cpu().numpy(), axis=1) + 1
        disp_preds = np.expand_dims(disp_preds, axis=1)
        disp_preds = visualize_rgb(disp_preds, num_classes)
        disp_preds_w_mask = disp_preds * label_mask

        if show_visdom:
            visdom_plot_images(self.vis, disp_preds, 'Predicted Images')
            visdom_plot_images(self.vis, disp_preds_w_mask, 'Predicted Images with Label Mask')

        # Show gradnorm per batch
        if show_visdom:
            if split == 'train':
                visdom_plot_metric('gradnorm', split, 'Grad Norm', 'Batch', 'Norm', self.progress_data, self.vis)

        # TODO: put this into a separate helper function?
        if save:
            save_dir = save_dir.replace(" ", "")
            save_dir = save_dir.replace(":", "")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(torch.from_numpy(label_mask), os.path.join(save_dir, 'label_masks.png'), nrow=NROW, normalize=True) 
            save_image(boi, os.path.join(save_dir, 'inputs.png'), nrow=NROW, normalize=True)
            save_image(torch.from_numpy(disp_targets), os.path.join(save_dir, 'targets.png'), nrow=NROW, normalize=True) 
            save_image(torch.from_numpy(disp_preds), os.path.join(save_dir, 'preds.png'), nrow=NROW, normalize=True)
            save_image(torch.from_numpy(disp_preds_w_mask), os.path.join(save_dir, 'preds_w_masks.png'), nrow=NROW, normalize=True)

        if show_matplot:
            labels_grid = make_grid(torch.from_numpy(label_mask), nrow=NROW, normalize=True, padding=8, pad_value=255) 
            inputs_grid = make_grid(boi, nrow=NROW, normalize=True, padding=8, pad_value=255)
            targets_grid = make_grid(torch.from_numpy(disp_targets), nrow=NROW, normalize=True, padding=8, pad_value=255) 
            preds_grid = make_grid(torch.from_numpy(disp_preds), nrow=NROW, normalize=True, padding=8, pad_value=255)
            predsmask_grid = make_grid(torch.from_numpy(disp_preds_w_mask), nrow=NROW, normalize=True, padding=8, pad_value=255)
            return labels_grid, inputs_grid, targets_grid, preds_grid, predsmask_grid
    
    
    def record_epoch(self, split, epoch_num, country, save=False, save_dir=None):
        """ Record values for epoch in visdom
        """
        if country in ['ghana', 'southsudan', 'tanzania', 'germany']:
            class_names = CROPS[country]
        else:
            raise ValueError(f"Country {country} not supported in visualize.py, record_epoch")

        if self.epoch_data[f'{split}_loss'] is not None: 
            loss_epoch = self.epoch_data[f'{split}_loss'] / self.epoch_data[f'{split}_pix']
        if self.epoch_data[f'{split}_correct'] is not None: 
            acc_epoch = self.epoch_data[f'{split}_correct'] / self.epoch_data[f'{split}_pix']

        # Don't append if you are saving. Information has already been appended!
        if save == False:
            self.progress_data[f'{split}_loss'].append(loss_epoch)
            self.progress_data[f'{split}_acc'].append(acc_epoch)
            self.progress_data[f'{split}_f1'].append(metrics.get_f1score(self.epoch_data[f'{split}_cm'], avg=True))

            if self.progress_data[f'{split}_classf1'] is None:
                self.progress_data[f'{split}_classf1'] = metrics.get_f1score(self.epoch_data[f'{split}_cm'], avg=False)
                self.progress_data[f'{split}_classf1'] = np.vstack(self.progress_data[f'{split}_classf1']).T
            else:
                self.progress_data[f'{split}_classf1'] = np.vstack((self.progress_data[f'{split}_classf1'], metrics.get_f1score(self.epoch_data[f'{split}_cm'], avg=False)))

        for cur_metric in ['loss', 'acc', 'f1']:
            visdom_plot_metric(cur_metric, split, f'{split} {cur_metric}', 'Epoch', cur_metric, self.progress_data, self.vis)
            if save or split in['test']:
                save_dir = save_dir.replace(" ", "")
                save_dir = save_dir.replace(":", "")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir) 
                visdom_save_metric(cur_metric, split, f'{split}{cur_metric}', 'Epoch', cur_metric, self.progress_data, save_dir)

        visdom_plot_many_metrics('classf1', split, f'{split}_per_class_f1-score', 'Epoch', 'per class f1-score', class_names, self.progress_data, self.vis)

        fig = util.plot_confusion_matrix(self.epoch_data[f'{split}_cm'], class_names,
                                         normalize=True,
                                         title='{} confusion matrix, epoch {}'.format(split, epoch_num),
                                         cmap=plt.cm.Blues)

        self.vis.matplot(fig, win=f'{split} CM')
        if save or split in ['test']:
            visdom_save_many_metrics('classf1', split, f'{split}_per_class_f1', 'Epoch', 'per class f1-score', class_names, self.progress_data, save_dir)               
            fig.savefig(os.path.join(save_dir, f'{split}_cm.png')) 
            classification_report(self.epoch_data, split, epoch_num, country, save_dir)

def clip_boi(boi):
    """ Clip bands of interest outside of 2*std per image sample
    """
    for sample in range(boi.shape[0]):
        sample_mean = torch.mean(boi[sample, :, :, :])
        sample_std = torch.std(boi[sample, :, :, :])
        min_clip = sample_mean - 2*sample_std
        max_clip = sample_mean + 2*sample_std

        boi[sample, :, :, :][boi[sample, :, :, :] < min_clip] = min_clip
        boi[sample, :, :, :][boi[sample, :, :, :] > max_clip] = max_clip
   
        boi[sample, :, :, :] = (boi[sample, :, :, :] - min_clip)/(max_clip - min_clip)
    return boi

def classification_report(all_metrics, split, epoch_num, country, save_dir):
    if country in ['ghana', 'southsudan', 'tanzania', 'germany']:
        class_names = CROPS[country]
    else:
        raise ValueError(f"Country {country} not supported in visualize.py, record_epoch")
    
    if all_metrics[f'{split}_loss'] is not None: loss_epoch = all_metrics[f'{split}_loss'] / all_metrics[f'{split}_pix']
    if all_metrics[f'{split}_correct'] is not None: acc_epoch = all_metrics[f'{split}_correct'] / all_metrics[f'{split}_pix']
    
    observed_accuracy = np.sum(all_metrics[f'{split}_cm'].diagonal()) / np.sum(all_metrics[f'{split}_cm'])
    expected_accuracy = np.sum(np.sum(all_metrics[f'{split}_cm'], axis=0) * np.sum(all_metrics[f'{split}_cm'], axis=1) / np.sum(all_metrics[f'{split}_cm'])) / np.sum(all_metrics[f'{split}_cm'])
    kappa =  (observed_accuracy - expected_accuracy)/(1 - expected_accuracy)

    fname = os.path.join(save_dir, split + '_classification_report.txt')
    with open(fname, 'a') as f:
        f.write('Country:\n ' + country + '\n\n')
        f.write('Epoch number:\n ' + str(epoch_num) + '\n\n')
        f.write('Split:\n ' + split + '\n\n')
        f.write('Epoch Loss:\n ' + str(loss_epoch) + '\n\n')
        f.write('Epoch Accuracy:\n ' + str(acc_epoch) + '\n\n')
        f.write('Observed Accuracy:\n ' + str(observed_accuracy) + '\n\n')
        f.write('Epoch f1:\n ' + str(metrics.get_f1score(all_metrics[f'{split}_cm'], avg=True)) + '\n\n') 
        f.write('Kappa coefficient:\n ' + str(kappa) + '\n\n')
        f.write('Per class accuracies:\n ' + str(all_metrics[f'{split}_cm'].diagonal()/all_metrics[f'{split}_cm'].sum(axis=1)) + '\n\n')
        f.write('Per class f1 scores:\n ' + str(metrics.get_f1score(all_metrics[f'{split}_cm'], avg=False)) + '\n\n')
        f.write('Crop Class Names:\n ' + str(class_names) + '\n\n')
        f.write('Confusion Matrix:\n ' + str(all_metrics[f'{split}_cm']) + '\n\n')


    

def setup_visdom(env_name, model_name):
    # TODO: Add args to visdom envs default name
    env_name = model_name if not env_name else env_name
    return visdom.Visdom(port=8097, env=env_name)

def visdom_save_metric(metric_name, split, title, x_label, y_label, vis_data, save_dir):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
    Y=np.array(vis_data['{}_{}'.format(split, metric_name)])
    X=np.array(range(len(vis_data['{}_{}'.format(split, metric_name)])))

    plt.figure()
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(['{}_{}'.format(split, metric_name)])
    plt.savefig(os.path.join(save_dir, title + '.png'))
    plt.close()

def visdom_save_many_metrics(metric_name, split, title, x_label, y_label, legend_lbls, vis_data, save_dir):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
 
    Y = vis_data['{}_{}'.format(split, metric_name)]
    X = np.array([range(len(vis_data['{}_{}'.format(split, metric_name)]))] * Y.shape[1]).T 

    plt.figure()
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend_lbls)
    plt.savefig(os.path.join(save_dir, title + split + '.png'))
    plt.close()

def visdom_plot_metric(metric_name, split, title, x_label, y_label, vis_data, vis):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
    vis.line(Y=np.array(vis_data['{}_{}'.format(split, metric_name)]),
             X=np.array(range(len(vis_data['{}_{}'.format(split, metric_name)]))),
             win=title,
             opts={'legend': ['{}_{}'.format(split, metric_name)],
                   'markers': False, 
                   'title': title,
                   'xlabel': x_label,
                   'ylabel': y_label})

def visdom_plot_many_metrics(metric_name, split, title, x_label, y_label, legend_lbls, vis_data, vis):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
 
    Y = vis_data['{}_{}'.format(split, metric_name)]
    X = np.array([range(len(vis_data['{}_{}'.format(split, metric_name)]))] * Y.shape[1]).T 
    vis.line(Y=Y,
             X=X,
             win=title,
             opts={'legend': legend_lbls,
                   'markers': False, 
                   'title': title,
                   'xlabel': x_label,
                   'ylabel': y_label})
    
def visdom_plot_images(vis, imgs, win):
    """
    Plot image panel in visdom
    Args: 
      imgs - (array) array of images [batch x channels x rows x cols]
      win - (str) serves as both window name and title name
    """
    vis.images(imgs, nrow=NROW, win=win, padding=8, opts={'title': win})

def visualize_rgb(argmax_array, num_classes, class_colors=None): 
    mask = []
    rgb_output = np.zeros((argmax_array.shape[0], 3, argmax_array.shape[2], argmax_array.shape[3]))

    if class_colors == None:
        rgbs = [ [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], 
                 [192, 0, 0], [192, 192, 0], [0, 192, 0], [0, 192, 192], [0, 0, 192],
                 [128, 0, 0], [128, 128, 0], [0, 128, 0], [0, 128, 128], [0, 0, 128],
                 [64, 0, 0], [64, 64, 0], [0, 64, 0], [0, 64, 64], [0, 0, 64] ]
        rgbs = rgbs[:num_classes]
 
    assert len(rgbs) == num_classes

    for cur_class in range(0, num_classes):
        tmp = np.asarray([argmax_array == cur_class+1])[0]

        mask_cat = np.concatenate((tmp, tmp, tmp), axis=1)

        class_vals = np.concatenate((np.ones_like(tmp)*rgbs[cur_class][0],
                                     np.ones_like(tmp)*rgbs[cur_class][1],
                                     np.ones_like(tmp)*rgbs[cur_class][2]), axis=1) 

        rgb_output += (mask_cat * class_vals)
        
    return rgb_output

