import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from train_and_eval.utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries


def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device):
        optimizer.zero_grad()
        print(sample['inputs'].shape)
        outputs = net(sample['inputs'].to(torch.float32).to(device))
        ground_truth = sample['labels'][:, center, center, 0].to(torch.int64).to(device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        Neval = len(evalloader)

        predicted_all = []
        labels_all = []

        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):

                if step % 500 == 0:
                    print("Eval step %d of %d" % (step, Neval))

                logits = net(sample['inputs'].to(torch.float32).to(device))

                _, predicted = torch.max(logits.data, -1)

                predicted_all.append(predicted.view(-1).cpu().numpy())
                labels_all.append(sample['labels'][:, center, center, 0].view(-1).to(torch.int64).cpu().numpy())

                # if step > 5:
                #    break

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels = np.unique(target_classes)

        print("-------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print("-------------------------------------------------------------------------------------------------------")
        return [un_labels,
                {"macro": {"Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}]


    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    if lin_cls:
        print('Train linear classifier only')
        trainable_params = get_net_trainable_params(net)[-2:]
    else:
        print('Train network end-to-end')
        trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    writer = SummaryWriter(save_path)

    BEST_IOU = 0
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):

            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step

            logits, labels, loss = train_step(net, sample, loss_fn, optimizer, device)

            if abs_step % train_metrics_steps == 0:
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=None, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f, lr: %.6f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1'], optimizer.param_groups[0]["lr"]))
                print('predicted: ', torch.max(logits.data, -1)[1])
                print('ground truths: ', labels)
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    BEST_IOU = eval_metrics[1]['macro']['IOU']

                write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval", optimizer=None)
                net.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', nargs='+', default=[0, 1], type=int,
                        help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config
    device_ids = args.device
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    center = config['MODEL']['img_res'] // 2
    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
