import torch
import torch.nn as nn

from models.CropTypeMapping.modelling.recurrent_norm import RecurrentNorm2d
from models.CropTypeMapping.modelling.cgru_cell import ConvGRUCell
from models.CropTypeMapping.modelling.util import initialize_weights

class CGRU(nn.Module):

    def __init__(self, input_size, hidden_dims, kernel_sizes, gru_num_layers, batch_first=True, bias=True, return_all_layers=False):
        """
           Args:
                input_size - (tuple) should be (time_steps, channels, height, width)
                hidden_dims - (list of ints) number of filters to use per layer
                kernel_sizes - lstm kernel sizes
                gru_num_layers - (int) number of stacks of ConvLSTM units per step
        """

        super(CGRU, self).__init__()
        (self.num_timesteps, self.start_num_channels, self.height, self.width) = input_size

        self.gru_num_layers = gru_num_layers
        self.bias = bias
        
        if isinstance(kernel_sizes, list):
            if len(kernel_sizes) != gru_num_layers and len(kernel_sizes) == 1:
                self.kernel_sizes = kernel_sizes * gru_num_layers
            else:
                self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * gru_num_layers      
        
        if isinstance(hidden_dims, list):
            if len(hidden_dims) != gru_num_layers and len(hidden_dims) == 1:
                self.hidden_dims = hidden_dims * gru_num_layers
            else:
                self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dims] * gru_num_layers       
        
        self.init_hidden_state = self._init_hidden()
        
        cell_list = []
        for i in range(self.gru_num_layers):
            cur_input_dim = self.start_num_channels if i == 0 else self.hidden_dims[i-1]

            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim = cur_input_dim,
                                         hidden_dim = self.hidden_dims[i],
                                         num_timesteps = self.num_timesteps,
                                         kernel_size = self.kernel_sizes[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        initialize_weights(self)

    def forward(self, input_tensor, hidden_state=None):

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.gru_num_layers):
            # double check that this is right? i.e not resetting every time to 0?
            h = self.init_hidden_state[layer_idx]
            h = h.expand(input_tensor.size(0), h.shape[1], h.shape[2], h.shape[3]).cuda()
            output_inner_layers = []
            
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=h, timestep=t)

                output_inner_layers.append(h)

            layer_output = torch.stack(output_inner_layers, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append(h)

        # TODO: Rework this so that we concatenate all the internal outputs as features for classification
        # Just take last output for prediction
        layer_output_list = layer_output_list[-1:]
        last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self):
        init_states = []
        for i in range(self.gru_num_layers):
            init_states.append(nn.Parameter(torch.zeros(1, self.hidden_dims[i], self.width, self.height)))
        return nn.ParameterList(init_states)
