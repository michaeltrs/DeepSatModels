"""

Loss functions and optimization definition.

"""

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import preprocess

from constants import *

def get_loss_fn(model_name):
    """
        Allows for changing the loss function depending on the model.
        Currently always returns the focal_loss.
    """
    return mask_ce_loss

def focal_loss(y_true, y_pred, reduction, country, loss_weight=False, weight_scale=1, gamma=2):
    """ Implementation of focal loss

    Args:
      y_true - (torch tensor) torch.Size([batch_size, num_classes, img_height, img_width]) 
                tensor of ground truth crop classes
      y_pred - (torch tensor) torch.Size([batch_size, num_classes, img_height, img_width])
                tensor of predicted crop classes
      reduction - (str) "sum" specified to return loss and number examples in order to accumulate 
                   over many batches. All other strings return loss / num_examples 
      loss_weight - (bool) whether or not to use weighted loss, weights defined in constants file
      weight_scale - (float, int) constant that loss weights are multiplied by
      gamma - (int, float) constant for focal loss 

    Returns:
      loss - (float) loss value calculated wrt y_true and y_pred
      num_examples - (int) returned when reduction == "sum" so that loss
                      can be calculated over many batches
    """ 
    y_true = preprocess.reshapeForLoss(y_true)
    num_examples = torch.sum(y_true, dtype=torch.float32).cuda()
    
    bs, classes, rows, cols = y_pred.shape
    
    y_pred = preprocess.reshapeForLoss(y_pred)
    y_pred, y_true = preprocess.maskForLoss(y_pred, y_true)
    y_confidence, _ = torch.sort(y_pred, dim=1, descending=True)
    y_confidence = y_confidence[:, 0] - y_confidence[:, 1]
    y_confidence = y_confidence.view([bs, rows, cols]).detach().cpu().numpy() * 255
    y_true = y_true.type(torch.LongTensor).cuda()
    
    if loss_weight:
        loss_fn = nn.NLLLoss(weight = LOSS_WEIGHT[country] ** weight_scale,reduction="none")
    else:
        loss_fn = nn.NLLLoss(reduction="none")
    
    # get the predictions for each true class
    nll_loss = loss_fn(y_pred, y_true)
    x = torch.gather(y_pred, dim=1, index=y_true.view(-1, 1))
    # tricky line, essentially gathers the predictions for the correct class and takes e^{pred} to undo 
    # log operation 
    # .view(-1) necessary to get correct shape
    focal_loss = (1 - torch.exp(x)) ** gamma
    focal_loss = focal_loss.view(-1)
    y = focal_loss * nll_loss
    loss = torch.sum(focal_loss * nll_loss)

    if num_examples == 0:
        print("WARNING: NUMBER OF EXAMPLES IS 0")

    if reduction == "sum":
        if num_examples == 0:
            return None, None, 0
        else:
            return loss, y_confidence, num_examples
    else:
        if num_examples == 0:
            return None
        else:
            return loss / num_examples, y_confidence

          
def mask_ce_loss(y_true, y_pred, reduction, country, loss_weight=False, weight_scale=1):
    """
    Args:
      y_true - (torch tensor) torch.Size([batch_size, num_classes, img_height, img_width]) 
                tensor of ground truth crop classes
      y_pred - (torch tensor) torch.Size([batch_size, num_classes, img_height, img_width])
                tensor of predicted crop classes
      reduction - (str) "sum" specified to return loss and number examples in order to accumulate 
                   over many batches. All other strings return loss / num_examples 
    
    nn.CrossEntropyLoss expects inputs: y_pred [N x classes] and y_true [N x 1]
    As input, y_pred and y_true have shapes [batch x classes x rows x cols] 

    Finally, to get y_true from [N x classes] to [N x 1], we take the argmax along
      the first dimension to get the largest class values from the one-hot encoding

    """
    y_true = preprocess.reshapeForLoss(y_true)
    num_examples = torch.sum(y_true, dtype=torch.float32).cuda()
    y_pred = preprocess.reshapeForLoss(y_pred)
    y_pred, y_true = preprocess.maskForLoss(y_pred, y_true)
   
    if loss_weight:
        loss_fn = nn.NLLLoss(weight=LOSS_WEIGHT[country] ** weight_scale, reduction="none")
    else:
        loss_fn = nn.NLLLoss(reduction="none") 

    total_loss = torch.sum(loss_fn(y_pred, y_true.cuda()))
   
    if num_examples == 0:
        print("WARNING: NUMBER OF EXAMPLES IS 0")

    if reduction == "sum":
        if num_examples == 0:
            return None, None, 0
        else:
            return total_loss, None, num_examples
    else:
        if num_examples == 0:
            return None
        else:
            return total_loss / num_examples, None

def get_optimizer(params, optimizer_name, lr, momentum, weight_decay):
    """ Define optimizer for model training
    Args:
      params - specifies parameters that are to be optimized
      optimizer_name - (str) specifies which optimizer to use
      lr - (float) initial learning rate
      momentum - (float) momentum for stochastic gradient descent
      weight_decay - (float) 

    Returns: 
      returns optimizer defined by input parameters to be used 
       in model training
    """
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        #TODO activate amsgrad=True?
        return optim.Adam(params, lr=lr, weight_decay=weight_decay) 

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")

