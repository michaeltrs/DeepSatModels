import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from utils.config_files_utils import get_params_values
from copy import deepcopy


def get_loss(config, device, reduction='mean'):
    model_config = config['MODEL']
    loss_config = config['SOLVER']

    print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, device, reduction=reduction))
        return loss_fun

    # Contrastive Loss -----------------------------------------------------------------------
    if loss_config['loss_function'] in ['contrastive_loss', 'masked_contrastive_loss']:
        pos_weight = get_params_values(config['SOLVER'], 'pos_weight', 1.0)
        print("cscl positive weight: ", pos_weight)
        return MaskedContrastiveLoss(pos_weight=pos_weight, reduction=reduction)

    # Binary Cross-Entropy Loss -----------------------------------------------------------
    if loss_config['loss_function'] == 'binary_cross_entropy':
        if reduction is None:
            reduction = 'none'
        return nn.BCEWithLogitsLoss(reduction=reduction)

    if loss_config['loss_function'] == 'masked_binary_cross_entropy':
        pos_weight = config['SOLVER']['pos_weight']
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        return MaskedBinaryCrossEntropy(reduction=reduction, pos_weight=pos_weight)

    # Cross-Entropy Loss ------------------------------------------------------------------
    elif loss_config['loss_function'] == 'cross_entropy':
        num_classes = get_params_values(model_config, 'num_classes', None)
        weight = torch.Tensor(num_classes * [1.0]).to(device)
        if loss_config['class_weights'] not in [None, {}]:
            for key in loss_config['class_weights']:
                weight[key] = loss_config['class_weights'][key]
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    # Masked Cross-Entropy Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_cross_entropy':
        mean = reduction == 'mean'
        return MaskedCrossEntropyLoss(mean=mean)
    
    # Focal Loss --------------------------------------------------------------------------
    elif loss_config['loss_function'] in ['focal_loss', 'masked_focal_loss']:
        gamma = get_params_values(loss_config, "gamma", 1.0)
        try:
            alpha = get_params_values(loss_config, "alpha", None)
        except ValueError:
            alpha = None
        if loss_config['loss_function'] == 'focal_loss':
            return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        elif loss_config['loss_function'] == 'masked_focal_loss':
            return MaskedFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    # Masked Multiclass Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_dice_loss':
        return MaskedDiceLoss(reduction=reduction)


def per_class_loss(criterion, logits, labels, unk_masks, n_classes):
    class_loss = []
    class_counts = []
    for class_ in range(n_classes):
        idx = labels == class_
        class_loss.append(
            criterion(logits[idx.repeat(1, 1, 1, n_classes)].reshape(-1, n_classes),  # ???
                      labels[idx].reshape(-1, 1),
                      unk_masks[idx].reshape(-1, 1)).detach().cpu().numpy()
        )
        class_counts.append(unk_masks[idx].sum().cpu().numpy())
    class_loss = np.array(class_loss)
    class_counts = np.array(class_counts)
    return np.nan_to_num(class_loss, nan=0.0), class_counts


class MaskedContrastiveLoss(torch.nn.Module):
    def __init__(self, pos_weight=1, reduction="mean"):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedContrastiveLoss, self).__init__()

        self.pos_weight = pos_weight
        self.reduction = reduction
        self.h = 1e-7

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        else:
            target = ground_truth[0]
            mask = ground_truth[1].to(torch.float32)

        loss = - self.pos_weight * target * logits + (1 - target) * logits
        if mask is not None:
            loss = mask * loss

        if self.reduction == "mean":
            return loss.mean()  # loss.sum() / (mask.sum() - 1)
        return loss


class MaskedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction="mean", pos_weight=None):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedBinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
        elif len(ground_truth) == 1:
            target = ground_truth[0]
        else:
            target = ground_truth[0][ground_truth[1]]
            logits = logits[ground_truth[1]]
        return self.loss_fn(logits, target)


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean
    
    def forward(self, logits, ground_truth):
        """
            Args:
                logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                    (batch, max_len, num_classes) which contains the
                    unnormalized probability for each class.
                target: A Variable containing a LongTensor of size
                    (batch, max_len) which contains the index of the true
                    class for each corresponding step.
                length: A Variable containing a LongTensor of size (batch,)
                    which contains the length of each data in a batch.
            Returns:
                loss: An average loss value masked by the length.
            """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")
        
        if mask is not None:
            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat


class MaskedFocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(MaskedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(-1, target.unsqueeze(-1))
        logpt = logpt.reshape(-1)
        pt = logpt.exp()  # Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class MaskedDiceLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, reduction=None):
        super(MaskedDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        target_onehot = torch.eye(logits.shape[-1])[target].to(torch.float32).cuda()  # .permute(0,3,1,2).float().cuda()
        predicted_prob = F.softmax(logits, dim=-1)

        inter = (predicted_prob * target_onehot).sum(dim=-1)
        union = predicted_prob.pow(2).sum(dim=-1) + target_onehot.sum(dim=-1)

        loss = 1 - 2 * inter / union

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class FocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
        
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")
