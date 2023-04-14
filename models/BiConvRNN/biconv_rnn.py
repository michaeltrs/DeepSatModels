# code from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/sequenceencoder.py
import torch
import torch.nn
from models.BiConvRNN.conv_lstm import ConvLSTMCell, ConvLSTM
from models.BiConvRNN.conv_gru import ConvGRU
import torch.nn.functional as F
from utils.config_files_utils import get_params_values


class BiRNNSequentialEncoderClassifier(torch.nn.Module):
    """
    modified by michaeltrs
    Same model as BiRNNSequentialEncoder below. Output features are weighted sum aggregated based on masks to get
    single vector (per sample) logits for multilabel classification
    """

    def __init__(self, input_size, input_dim, conv3d_hidden_dim, rnn_hidden_dim, kernel_size,
                 shape_pattern="NCTHW", nclasses=8, bias=True, gpu_id=None, temp_model="ConvGRU"):
        super(BiRNNSequentialEncoderClassifier, self).__init__()
        self.net = BiRNNSequentialEncoder(input_size, input_dim, conv3d_hidden_dim, rnn_hidden_dim, kernel_size,
                                           shape_pattern, nclasses, bias, gpu_id, temp_model)

    def forward(self, inputs, unk_masks, seq_lengths):
        out = self.net(inputs, seq_lengths)
        n_classes = out.shape[1]
        int_masks = unk_masks.to(torch.int32)
        num_locs = int_masks.sum(dim=(1, 2, 3))
        logits = int_masks.permute(0, 3, 1, 2).repeat(1, n_classes, 1, 1) * out
        logits = logits.sum(dim=(2, 3)) / num_locs.unsqueeze(-1).repeat(1, n_classes)
        return logits


class BiRNNSequentialEncoder(torch.nn.Module):
    """
    modified by michaeltrs
    """

    def __init__(self, config, device):

        super(BiRNNSequentialEncoder, self).__init__()

        if device == -1:
            device = torch.device("cpu")
        elif type(device) is int:
            device = torch.device("cuda:%d" % device)  # if torch.cuda.is_available() else "cpu")
        elif type(device) != torch.device:
            raise ValueError('Device not defined for module BiRNNSequentialEncoder')

        self.height, self.width = 2 * [get_params_values(config, 'img_res', 24)]
        self.input_dim = get_params_values(config, 'num_channels', 3)
        self.shape_pattern = 'NTHWC'
        self.inconv_avgpool_kernel = tuple(get_params_values(config, 'inconv_avgpool_kernel', (None, None)))

        self.conv3d_hidden_dim = get_params_values(config, 'conv3d_dim', 64)
        self.inconv_xtimes = get_params_values(config, 'inconv_xtimes', 1)
        self.inconv_kernel_size = tuple(get_params_values(config, 'inconv_kernel', (1, 3, 3)))
        self.pad_size = int((self.inconv_kernel_size[-1] - 1) / 2)

        self.backbone = get_params_values(config, 'backbone', None)
        self.rnn_hidden_dim = get_params_values(config, 'rnn_hidden_dim', [256])
        self.conv_xtimes = get_params_values(config, 'conv_xtimes', 1)
        self.kernel_size = tuple(get_params_values(config, 'conv_kernel', (3, 3)))
        self.conv_kernel_dilation = get_params_values(config, 'conv_kernel_dilation', [1])

        self.outconv_xtimes = get_params_values(config, 'outconv_xtimes', 1)
        self.outconv_kernel_size = tuple(get_params_values(config, 'outconv_kernel', (3, 3)))
        self.outconv_type = get_params_values(config, 'outconv_type', "conv2d")
        self.pad_size_out = int((self.outconv_kernel_size[-1] - 1) / 2)

        self.nclasses = get_params_values(config, 'num_classes', 21)

        if self.inconv_avgpool_kernel[0] is not None:
            self.inconv_avgpool = torch.nn.AvgPool2d(self.inconv_avgpool_kernel, stride=1, padding=0)

        self.inconv = torch.nn.Conv3d(
            self.input_dim, self.inconv_xtimes * self.conv3d_hidden_dim, self.inconv_kernel_size).to(device)

        if self.backbone == "ConvLSTM":
            self.forward_model = ConvLSTM(input_size=(self.height, self.width),
                                          input_dim=self.conv3d_hidden_dim,
                                          hidden_dim=[hd for hd in self.rnn_hidden_dim],
                                          kernel_size=self.kernel_size,
                                          shape_pattern="NCTHW",
                                          bias=True,
                                          return_all_layers=False,
                                          device=device).to(device)
            self.backward_model = ConvLSTM(input_size=(self.height, self.width),
                                           input_dim=self.conv3d_hidden_dim,
                                           hidden_dim=[hd for hd in self.rnn_hidden_dim],
                                           kernel_size=self.kernel_size,
                                           shape_pattern="NCTHW",
                                           bias=True,
                                           return_all_layers=False,
                                           device=device).to(device)
        elif self.backbone == "ConvGRU":
            self.forward_model = ConvGRU(input_size=(self.height, self.width),
                                         input_dim=self.conv3d_hidden_dim,
                                         hidden_dim=[hd for hd in self.rnn_hidden_dim],
                                         kernel_size=self.kernel_size,
                                         shape_pattern="NCTHW",
                                         bias=True,
                                         return_all_layers=False,
                                         device=device).to(device)
            self.backward_model = ConvGRU(input_size=(self.height, self.width),
                                          input_dim=self.conv3d_hidden_dim,
                                          hidden_dim=[hd for hd in self.rnn_hidden_dim],
                                          kernel_size=self.kernel_size,
                                          shape_pattern="NCTHW",
                                          bias=True,
                                          return_all_layers=False,
                                          device=device).to(device)
        else:
            raise ValueError("Model name %s not understood. "
                             "Model for MTLCC_prev encoder should be either ConvLSTM or ConvGRU")

        if self.outconv_type == "conv2d":
            self.outconv = torch.nn.Conv2d(
                2 * self.rnn_hidden_dim[-1], self.outconv_xtimes * self.nclasses, self.outconv_kernel_size).to(device)
        else:
            raise NotImplemented("Only conv types: conv2d, lsa_conv2d implemented for outconv")

    def forward(self, inputs):  # , hidden=None, state=None):
        inputs_forward, inputs_backward, seq_lengths = inputs

        # Desired shape for tensor in NCTHW
        if self.shape_pattern is "NTHWC":
            inputs_forward = inputs_forward.permute(0, 4, 1, 2, 3)
            inputs_backward = inputs_backward.permute(0, 4, 1, 2, 3)
        elif self.shape_pattern is "NTCHW":
            # (b x t x c x h x w) -> (b x c x t x h x w)
            inputs_forward = inputs_forward.permute(0, 2, 1, 3, 4)
            inputs_backward = inputs_backward.permute(0, 2, 1, 3, 4)

        if self.inconv_avgpool_kernel[0] is not None:
            bs, c, t, h, w = inputs_forward.shape
            pad_size = int((self.inconv_avgpool_kernel[-1] - 1) / 2)
            inputs_forward = torch.nn.functional.pad(inputs_forward.reshape(bs, c * t, h, w),
                                                     (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                                                     'reflect').reshape(
                bs, c, t, h + 2 * pad_size, w + 2 * pad_size)
            inputs_backward = torch.nn.functional.pad(inputs_backward.reshape(bs, c * t, h, w),
                                                      (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                                                      'reflect').reshape(
                bs, c, t, h + 2 * pad_size, w + 2 * pad_size)
            bs, c, t, h, w = inputs_forward.shape
            inputs_forward = self.inconv_avgpool(inputs_forward.reshape(-1, h, w))
            inputs_backward = self.inconv_avgpool(inputs_backward.reshape(-1, h, w))
            inputs_forward = inputs_forward.reshape(bs, c, t, h - 2, w - 2)
            inputs_backward = inputs_backward.reshape(bs, c, t, h - 2, w - 2)

        if self.inconv_kernel_size[-1] == 2:
            inputs_forward = torch.nn.functional.pad(
                inputs_forward, (0, 1, 0, 1), 'constant', 0)
            inputs_backward = torch.nn.functional.pad(
                inputs_backward, (0, 1, 0, 1), 'constant', 0)
        else:
            inputs_forward = torch.nn.functional.pad(
                inputs_forward, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'constant', 0)
            inputs_backward = torch.nn.functional.pad(
                inputs_backward, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'constant', 0)

        inputs_forward = self.inconv.forward(inputs_forward)
        inputs_backward = self.inconv.forward(inputs_backward)

        if self.inconv_xtimes and (self.inconv_xtimes != 1):
            inputs_forward = reshape_and_sum(inputs_forward, self.conv3d_hidden_dim, self.inconv_xtimes, dim=1)
            inputs_backward = reshape_and_sum(inputs_backward, self.conv3d_hidden_dim, self.inconv_xtimes, dim=1)

        forward_out = self.forward_model(inputs_forward)[0]  # [0]
        backward_out = self.backward_model(inputs_backward)[0]  # [0]

        forward_out = torch.stack([forward_out[i, j, :, :, :] for i, j in enumerate(seq_lengths - 1)], dim=0)
        backward_out = torch.stack([backward_out[i, j, :, :, :] for i, j in enumerate(seq_lengths - 1)], dim=0)

        state = torch.cat((forward_out, backward_out), dim=1)  # (N x Cout x H x W)

        if self.outconv_type != "lsa_conv2d":
            if self.outconv_kernel_size[-1] == 2:
                state = torch.nn.functional.pad(state, (0, 1, 0, 1), 'constant', 0)
            else:
                state = torch.nn.functional.pad(state, (
                self.pad_size_out, self.pad_size_out, self.pad_size_out, self.pad_size_out), 'constant', 0)

        output = self.outconv.forward(state)

        if self.outconv_xtimes and (self.outconv_xtimes != 1):
            output = reshape_and_sum(output, self.nclasses, self.outconv_xtimes, dim=1)

        return F.log_softmax(output, dim=1)


class LSTMSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=13, hidden_dim=64, nclasses=8, kernel_size=(3, 3), bias=False):
        super(LSTMSequentialEncoder, self).__init__()

        self.inconv = torch.nn.Conv3d(input_dim, hidden_dim, (1, 3, 3))

        # unidir ???
        self.cell = ConvLSTMCell(input_size=(height, width),
                                 input_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 bias=bias)

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3))

    def forward(self, x, hidden=None, state=None):

        # (b x t x c x h x w) -> (b x c x t x h x w)
        x = x.permute(0, 2, 1, 3, 4)

        x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        x = self.inconv.forward(x)

        b, c, t, h, w = x.shape

        if hidden is None:
            hidden = torch.zeros((b, c, h, w))
        if state is None:
            state = torch.zeros((b, c, h, w))

        if torch.cuda.is_available():
            hidden = hidden.cuda()
            state = state.cuda()

        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :, :], (hidden, state))

        x = torch.nn.functional.pad(state, (1, 1, 1, 1), 'constant', 0)
        x = self.final.forward(x)

        return F.log_softmax(x, dim=1)


def reshape_and_sum(inputs, feat_dim, xtimes, dim=1):
    """
    inputs size is [bs, d, t, h, w]
    """
    input_size = inputs.shape
    if len(input_size) == 5:
        bs, d, t, h, w = inputs.shape
        inputs = inputs.reshape(bs, xtimes, feat_dim, t, h, w)
    elif len(input_size) == 4:
        bs, d, h, w = inputs.shape
        inputs = inputs.reshape(bs, xtimes, feat_dim, h, w)
    inputs = inputs.sum(dim=dim)
    return inputs

