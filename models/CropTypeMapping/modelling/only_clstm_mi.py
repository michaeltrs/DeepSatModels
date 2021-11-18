import torch
import torch.nn as nn
from models.CropTypeMapping.modelling.util import initialize_weights
from models.CropTypeMapping.modelling.clstm import CLSTM
from models.CropTypeMapping.modelling.clstm_segmenter import CLSTMSegmenter
from models.CropTypeMapping.modelling.attention import ApplyAtt, attn_or_avg
from pprint import pprint

class ONLY_CLSTM_MI(nn.Module):
    """ ONLY_CLSTM_MI = MI_CLSTM model without UNet features
    """
    def __init__(self, 
                 num_bands,
                 crnn_input_size,
                 hidden_dims, 
                 lstm_kernel_sizes, 
                 conv_kernel_size, 
                 lstm_num_layers, 
                 avg_hidden_states, 
                 num_classes,
                 bidirectional,
                 max_timesteps,
                 satellites,
                 main_attn_type,
                 attn_dims):
        """
            input_size - (tuple) should be (time_steps, channels, height, width)
        """
        super(ONLY_CLSTM_MI, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.avg_hidden_states = avg_hidden_states
        self.bidirectional = bidirectional
        self.satellites = satellites
        self.num_bands = num_bands
        
        self.clstms = {}
        self.attention = {}
        self.finalconv = {}
 
        crnn_out_feats = crnn_input_size[1]

        for sat in satellites:
            if satellites[sat]: 
                crnn_input_size = list(crnn_input_size)
                crnn_input_size[1] = self.num_bands[sat]
                crnn_input_size = tuple(crnn_input_size)

                self.clstms[sat] = CLSTMSegmenter(input_size=crnn_input_size,
                                                  hidden_dims=hidden_dims, 
                                                  lstm_kernel_sizes=lstm_kernel_sizes, 
                                                  conv_kernel_size=conv_kernel_size, 
                                                  lstm_num_layers=lstm_num_layers, 
                                                  num_outputs=crnn_out_feats, 
                                                  bidirectional=bidirectional) 

                self.attention[sat] = ApplyAtt(main_attn_type, hidden_dims, attn_dims)

                self.finalconv[sat] = nn.Conv2d(in_channels=hidden_dims[-1], 
                                                out_channels=num_classes, 
                                                kernel_size=conv_kernel_size, 
                                                padding=int((conv_kernel_size-1)/2))

        for sat in satellites:
            if satellites[sat]:
                self.add_module(sat + "_clstm", self.clstms[sat])
                self.add_module(sat + "_finalconv", self.finalconv[sat])
                self.add_module(sat + "_attention", self.attention[sat])

        total_sats = len([sat for sat in self.satellites if self.satellites[sat]])
        self.out_linear = nn.Linear(num_classes * total_sats, num_classes)
        self.softmax = nn.Softmax2d()
        self.logsoftmax = nn.LogSoftmax(dim=1)
                
    def forward(self, inputs):
        preds = []
        for sat in self.satellites:
            if self.satellites[sat]:
                sat_data = inputs[sat]
                lengths = inputs[sat + "_lengths"]
                batch, timestamps, bands, rows, cols = sat_data.size()
                
                # Apply CRNN
                if self.clstms[sat] is not None:
                    crnn_output_fwd, crnn_output_rev = self.clstms[sat](sat_data) #, lengths)
                else:
                    crnn_output_fwd = crnn_input
                    crnn_output_rev = None

                # Apply attention
                reweighted = attn_or_avg(self.attention[sat], self.avg_hidden_states, crnn_output_fwd, crnn_output_rev, self.bidirectional, lengths)

                # Apply final conv
                scores = self.finalconv[sat](reweighted)
                sat_preds = self.logsoftmax(scores)
                preds.append(sat_preds)
        
        all_preds = torch.cat(preds, dim=1).permute(0, 2, 3, 1).contiguous()
        preds = self.out_linear(all_preds).permute(0, 3, 1, 2).contiguous()
        preds = self.logsoftmax(preds)
        return preds
