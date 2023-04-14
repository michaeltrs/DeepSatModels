# code from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device):
        """
        Initialize ConvLSTM cell.

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.device = device

        self.reset_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding)
        self.update_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding)
        self.out_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding)

    def forward(self, input_tensor, prev_state):
        stacked_inputs = torch.cat([input_tensor, prev_state], dim=1)  # concatenate along channel axis
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_tensor, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update
        return new_state

    def init_hidden(self, batch_size):
        vars = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        if self.device.type == 'cuda':
            vars = vars.cuda()
        return vars


class ConvGRU(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 shape_pattern="NTHWC", bias=True, return_all_layers=False, device=None):
        super(ConvGRU, self).__init__()

        if device is None:
            device = torch.device("cpu")

        self._check_kernel_size_consistency(kernel_size)
        self.num_layers = len(hidden_dim)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, self.num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, self.num_layers)
        if not len(kernel_size) == len(hidden_dim) == self.num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.shape_pattern = shape_pattern
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         device=device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # input_tensor = inputs["inputs"]

        # Desired shape for tensor in NTCHW
        if self.shape_pattern is "NTHWC":
            input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
        elif self.shape_pattern is "NCTHW":
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], prev_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    height = 48
    width = 48
    channels = 12
    gpu_id = 0

    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=[64],
                    kernel_size=(3, 3),
                    shape_pattern="NCTHW",
                    bias=True,
                    return_all_layers=False,
                    device=device).to(device)
    input_tensor = torch.rand((10, 12, 20, 48, 48)).to(device)
    out = model(input_tensor)
