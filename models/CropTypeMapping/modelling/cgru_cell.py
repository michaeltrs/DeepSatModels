# script assumes it will be called from root directory, hence 'modelling.recurrent_norm' instead of just 'recurrent_norm'

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from models.CropTypeMapping.constants import *
from models.CropTypeMapping.modelling.recurrent_norm import RecurrentNorm2d
from models.CropTypeMapping.modelling.util import initialize_weights

class ConvGRUCell(nn.Module):
    """
        
    """
    def __init__(self, input_size, input_dim, hidden_dim, num_timesteps, kernel_size, bias):
        """
        Initialize BiConvRNN cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.h_conv = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.W_h = nn.Conv2d(in_channels=self.hidden_dim,
                             out_channels=self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)
        
        self.U_h = nn.Conv2d(in_channels=self.input_dim,
                             out_channels=self.hidden_dim,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)
        
        self.h_norm = RecurrentNorm2d(2 * self.hidden_dim, self.num_timesteps)
        self.input_norm = RecurrentNorm2d(2 * self.hidden_dim, self.num_timesteps)
        
        initialize_weights(self)

    def forward(self, input_tensor, cur_state, timestep):
        # TODO: should not have to call cuda here, figure out where this belongs
        input_tensor.cuda()
        # BN over the outputs of these convs
        
        combined_conv = self.h_norm(self.h_conv(cur_state), timestep) + self.input_norm(self.input_conv(input_tensor), timestep)
        
        u_t, r_t = torch.split(combined_conv, self.hidden_dim, dim=1) 
        u_t = torch.sigmoid(u_t)
        r_t = torch.sigmoid(r_t)
        h_tilde = torch.tanh(self.W_h(r_t * cur_state) + self.U_h(input_tensor))
        h_next = (1 - u_t) * h_tilde + u_t * h_tilde
        
        return h_next
