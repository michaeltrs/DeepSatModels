import torch
import torch.nn as nn
from models.CropTypeMapping.modelling.util import initialize_weights
from models.CropTypeMapping.modelling.clstm import CLSTM
from models.CropTypeMapping.modelling.attention import ApplyAtt, attn_or_avg

class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, 
                 lstm_num_layers, num_outputs, bidirectional, with_pred=False, 
                 avg_hidden_states=None, attn_type=None, d=None, r=None, dk=None, dv=None): 

        super(CLSTMSegmenter, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.with_pred = with_pred        

        if self.with_pred:
            self.avg_hidden_states = avg_hidden_states
            self.attention = ApplyAtt(attn_type, hidden_dims, d=d, r=r, dk=dk, dv=dv) 
            self.final_conv = nn.Conv2d(in_channels=hidden_dims, 
                                        out_channels=num_outputs, 
                                        kernel_size=conv_kernel_size, 
                                        padding=int((conv_kernel_size-1)/2)) 
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.clstm_rev = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers, bidirectional)
        
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        initialize_weights(self)
       
    def forward(self, inputs):

        layer_outputs, last_states = self.clstm(inputs)
    
        rev_layer_outputs = None
        if self.bidirectional:
            rev_inputs = torch.flip(inputs, dims=[1])
            rev_layer_outputs, rev_last_states = self.clstm_rev(rev_inputs)

        if self.with_pred:
            # Apply attention
            reweighted = attn_or_avg(self.attention, self.avg_hidden_states, layer_outputs, rev_layer_ouputs, self.bidirectional)

            # Apply final conv
            scores = self.final_conv(reweighted)
            output = self.logsoftmax(scores)
            return output
        else:
            return layer_outputs, rev_layer_outputs
