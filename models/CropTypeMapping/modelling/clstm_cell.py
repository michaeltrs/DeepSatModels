# script assumes it will be called from root directory, hence 'modelling.recurrent_norm' instead of just 'recurrent_norm'

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from models.CropTypeMapping.constants import *
from models.CropTypeMapping.modelling.recurrent_norm import RecurrentNorm2d
from models.CropTypeMapping.modelling.util import initialize_weights

class ConvLSTMCell(nn.Module):
    """
        ConvLSTM Cell based on Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        arXiv: https://arxiv.org/abs/1506.04214

        Implementation based on stefanopini's at https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """
    def __init__(self, input_dim, hidden_dim, num_timesteps, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
       
        self.h_conv = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
       
        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        

        self.h_norm = RecurrentNorm2d(4 * self.hidden_dim, self.num_timesteps)
        self.input_norm = RecurrentNorm2d(4 * self.hidden_dim, self.num_timesteps)
        self.cell_norm = RecurrentNorm2d(self.hidden_dim, self.num_timesteps)
        
        initialize_weights(self)

    def forward(self, input_tensor, cur_state, timestep):
        
        h_cur, c_cur = cur_state
        # BN over the outputs of these convs
        combined_conv = self.h_norm(self.h_conv(h_cur), timestep) + self.input_norm(self.input_conv(input_tensor.cuda()), timestep)
 
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        # BN over the tanh
        h_next = o * self.cell_norm(torch.tanh(c_next), timestep)
        
        
        return h_next, c_next
