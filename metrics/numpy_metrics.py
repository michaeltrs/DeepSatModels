import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_mat(predicted, labels, n_classes):  # , unk_masks=None):
    """
                predicted
            -----------------
    labels |
    """

    cm = confusion_matrix(labels, predicted)
    # cm_side = cm.shape[0]
    rem = 0
    if cm.shape[0] < n_classes:
        batch_classes = np.unique(np.concatenate((predicted, labels))).tolist()
        for i in range(n_classes):
            # internal class missing
            if (i - rem) < len(batch_classes):
                if i < batch_classes[i-rem]:
                    cm = np.insert(cm, i, 0., axis=0)
                    cm = np.insert(cm, i, 0., axis=1)
                    rem += 1
                    i += 1
            # outer class(es) missing
            else:
                diff = n_classes - rem - len(batch_classes)
                cm_side = cm.shape[0]
                cm = np.concatenate((cm, np.zeros((diff, cm_side))), axis=0)
                cm = np.concatenate((cm, np.zeros((cm_side + diff, diff))), axis=1)
                break
    return cm


def get_prediction_splits(predicted, labels, n_classes):
    cm = confusion_mat(predicted, labels, n_classes).astype(np.float32)
    diag = np.diagonal(cm)
    rowsum = cm.sum(axis=1)
    colsum = cm.sum(axis=0)
    TP = (diag).astype(np.float32)
    FN = (rowsum - diag).astype(np.float32)
    FP = (colsum - diag).astype(np.float32)
    IOU = diag / (rowsum + colsum - diag)
    macro_IOU = diag.sum() / (rowsum.sum() + colsum.sum() - diag.sum())

    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = labels == class_
        is_correct = predicted[idx] == labels[idx]
        if is_correct.size == 0:
            is_correct = np.array(0)
        num_total.append(idx.sum())
        num_correct.append(is_correct.sum())   # previously was .mean()
    num_total = np.array(num_total).astype(np.float32)
    num_correct = np.array(num_correct)

    return TP, FP, FN, num_correct, num_total, IOU, macro_IOU


def get_splits(predicted, labels, n_classes):
    num_total = []
    num_correct = []
    for class_ in range(n_classes):
        idx = labels == class_
        num_total.append(idx.sum())
        num_correct.append((predicted[idx] == labels[idx]).mean())
    num_total = np.array(num_total)
    num_correct = np.array(num_correct)
    return num_correct, num_total


def get_metrics_from_splits(TP, FP, FN, num_correct, num_total):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if (type(precision) in [np.float32, np.float64]) and (precision + recall == 0.0):
        F1 = 0.0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    acc = num_correct / num_total
    return acc, precision, recall, F1


def nan_mean(v):
    return v[~np.isnan(v)].mean()


def get_classification_metrics(predicted, labels, n_classes, unk_masks=None):
    if unk_masks is not None:
        predicted = predicted[unk_masks]
        labels = labels[unk_masks]
    TP, FP, FN, num_correct, num_total, IOU, micro_IOU = get_prediction_splits(predicted, labels, n_classes) #  , per_class)
    micro_acc, micro_precision, micro_recall, micro_F1 = \
        get_metrics_from_splits(TP.sum(), FP.sum(), FN.sum(), num_correct.sum(), num_total.sum())
    macro_IOU = IOU[~np.isnan(IOU)].mean()
    acc, precision, recall, F1 = \
        get_metrics_from_splits(TP, FP, FN, num_correct, num_total)
    macro_acc = nan_mean(acc)
    macro_precision = nan_mean(precision)
    macro_recall = nan_mean(recall)
    macro_F1 = nan_mean(F1)
    acc = np.nan_to_num(acc, copy=True, nan=0.0)
    precision = np.nan_to_num(precision, copy=True, nan=0.0)
    recall = np.nan_to_num(recall, copy=True, nan=0.0)
    F1 = np.nan_to_num(F1, copy=True, nan=0.0)
    IOU = np.nan_to_num(IOU, copy=True, nan=0.0)
    return {'class': [acc, precision, recall, F1, IOU],
            'micro': [micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU],
            'macro': [macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU]}

    
def get_accuracy(predicted, labels, unk_mask=None, return_splits=False):
    if unk_mask is not None:
        predicted = predicted[unk_mask]
        labels = labels[unk_mask]
    is_correct = (predicted == labels).astype(float)
    num_correct = is_correct.sum()
    num_total = is_correct.shape[0]
    if return_splits:
        return num_correct, num_total
    return num_correct / num_total


def get_per_class_loss(losses, labels, unk_masks=None):
    if unk_masks is not None:
        losses = losses[unk_masks]
        labels = labels[unk_masks]
    unique_labels = np.unique(labels)
    class_loss = []
    for label in unique_labels:
        idx = labels == label
        class_loss.append(losses[idx].mean())
    return unique_labels, np.asarray(class_loss)

