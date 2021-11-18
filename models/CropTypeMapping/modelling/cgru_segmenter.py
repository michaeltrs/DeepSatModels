import torch
import torch.nn as nn
from models.CropTypeMapping.modelling.util import initialize_weights
from models.CropTypeMapping.modelling.cgru import CGRU

class CGRUSegmenter(nn.Module):
    """ cgru followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, gru_kernel_sizes, 
                 conv_kernel_size, gru_num_layers, num_classes, bidirectional, early_feats):

        super(CGRUSegmenter, self).__init__()
        self.early_feats = early_feats

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.cgru = CGRU(input_size, hidden_dims, gru_kernel_sizes, gru_num_layers)
        self.bidirectional = bidirectional
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.logsoftmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)

    def forward(self, inputs):
        layer_output_list, last_state_list = self.cgru(inputs)
        final_state = last_state_list[0]
        if self.bidirectional:
            rev_inputs = torch.tensor(inputs.cpu().detach().numpy()[::-1].copy(), dtype=torch.float32).cuda()
            rev_layer_output_list, rev_last_state_list = self.cgru(rev_inputs)
            final_state = torch.cat([final_state, rev_last_state_list[0][0]], dim=1)
        scores = self.conv(final_state)

        output = scores if self.early_feats else self.logsoftmax(scores)
        return output
