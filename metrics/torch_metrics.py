import torch
from metrics.numpy_metrics import get_classification_metrics
import numpy as np


def get_binary_metrics(logits, labels, return_all=False, thresh=0.5, name=""):
    logits = logits.reshape(-1, 1)
    probs = torch.nn.functional.sigmoid(logits)
    pred = (probs > thresh).to(torch.float32)
    labels = labels.reshape(-1, 1)
    bin_metrics = get_classification_metrics(
        predicted=pred.cpu().numpy(), labels=labels.cpu().numpy(), n_classes=2)
    acc, precision, recall, F1, IOU = bin_metrics['micro']
    if return_all:
        micro_metrics = {}
        for metrics_type in ['micro']:
            micro_metrics['%s%s_Accuracy' % (name, metrics_type)] = bin_metrics[metrics_type][0]
            micro_metrics['%s%s_Precision' % (name, metrics_type)] = bin_metrics[metrics_type][1]
            micro_metrics['%s%s_Recall' % (name, metrics_type)] = bin_metrics[metrics_type][2]
            micro_metrics['%s%s_F1' % (name, metrics_type)] = bin_metrics[metrics_type][3]
            micro_metrics['%s%s_IOU' % (name, metrics_type)] = bin_metrics[metrics_type][4]
        class_metrics = {}
        for metrics_type in ['class']:
            class_metrics['%s%s_Accuracy' % (name, metrics_type)] = bin_metrics[metrics_type][0]
            class_metrics['%s%s_Precision' % (name, metrics_type)] = bin_metrics[metrics_type][1]
            class_metrics['%s%s_Recall' % (name, metrics_type)] = bin_metrics[metrics_type][2]
            class_metrics['%s%s_F1' % (name, metrics_type)] = bin_metrics[metrics_type][3]
            class_metrics['%s%s_IOU' % (name, metrics_type)] = bin_metrics[metrics_type][4]
        return micro_metrics, class_metrics
    return {"%smicro_Accuracy" % name: acc, "%smicro_Precision" % name: precision, "%smicro_Recall" % name: recall,
            "%smicro_F1" % name: F1, "%smicro_IOU" % name: IOU}
    
    
def get_mean_metrics(logits, labels, n_classes, loss, epoch=0, step=0, unk_masks=None, name=""):
    """
    :param logits: (N, D, H, W)
    """
    _, predicted = torch.max(logits.data, 1)
    # unique_predictions = predicted.unique().cpu().numpy()
    predicted = predicted.reshape(-1).cpu().numpy()
    labels = labels.reshape(-1).cpu().numpy()
    if unk_masks is not None:
        unk_masks = unk_masks.reshape(-1).cpu().numpy()
    acc, precision, recall, F1, IOU = get_classification_metrics(
        predicted, labels, n_classes, unk_masks)['micro']
    loss_ = float(loss.detach().cpu().numpy())
    return {"%sAccuracy" % name: acc, "%sPrecision" % name: precision, "%sRecall" % name: recall,
            "%sF1" % name: F1, "%sIOU" % name: IOU, "%sLoss" % name: loss_}


def get_all_metrics(predicted, labels, n_classes, unk_masks=None, name=""):
    """
    :param logits: (N, D, H, W)
    """
    predicted = predicted.reshape(-1).cpu().numpy()
    labels = labels.reshape(-1).cpu().numpy()
    if unk_masks is not None:
        unk_masks = unk_masks.reshape(-1).cpu().numpy()
    cls_metrics = get_classification_metrics(predicted, labels, n_classes, unk_masks)
    micro_metrics = {}
    for metrics_type in ['micro']:
        micro_metrics['%s%s_Accuracy' % (name, metrics_type)] = cls_metrics[metrics_type][0]
        micro_metrics['%s%s_Precision' % (name, metrics_type)] = cls_metrics[metrics_type][1]
        micro_metrics['%s%s_Recall' % (name, metrics_type)] = cls_metrics[metrics_type][2]
        micro_metrics['%s%s_F1' % (name, metrics_type)] = cls_metrics[metrics_type][3]
        micro_metrics['%s%s_IOU' % (name, metrics_type)] = cls_metrics[metrics_type][4]
    class_metrics = {}
    for metrics_type in ['class']:
        class_metrics['%s%s_Accuracy' % (name, metrics_type)] = cls_metrics[metrics_type][0]
        class_metrics['%s%s_Precision' % (name, metrics_type)] = cls_metrics[metrics_type][1]
        class_metrics['%s%s_Recall' % (name, metrics_type)] = cls_metrics[metrics_type][2]
        class_metrics['%s%s_F1' % (name, metrics_type)] = cls_metrics[metrics_type][3]
        class_metrics['%s%s_IOU' % (name, metrics_type)] = cls_metrics[metrics_type][4]
    return micro_metrics, class_metrics


def accuracy(logits, labels, unk_masks):
    _, predicted = torch.max(logits, -1)
    is_correct = (predicted == labels)[unk_masks].type(torch.float32)
    acc = is_correct.mean().item()
    return acc


def place_value(number):
    return ("{:,}".format(number))


def get_counts(logits, ground_truth):
    real, real_counts = [gtc.cpu().numpy() for gtc in ground_truth.unique(return_counts=True)]
    pred, pred_counts = [prc.cpu().numpy() for prc in (logits > 0.5).to(torch.float32).unique(return_counts=True)]
    out = {}
    all_labels = np.unique(np.concatenate((real, pred)))
    for l in all_labels:
        out[l] = {}
        idx = real == l
        if idx.sum() == 1:
            out[l]['real'] = place_value(real_counts[idx][0])
        else:
            out[l]['real'] = 0
        idx = pred == l
        if idx.sum() == 1:
            out[l]['pred'] = place_value(pred_counts[idx][0])
        else:
            out[l]['pred'] = 0
    return out

