import torch
import torch.nn as nn

def attn_or_avg(attention, avg_hidden_states, layer_outputs, rev_layer_outputs, bidirectional, lengths=None):
    if (attention is None) or (attention(layer_outputs) is None):
        if not avg_hidden_states:
            # TODO: want to take the last non-zero padded output here instead!
            last_fwd_feat = layer_outputs[:, -1, :, :, :]
            last_rev_feat = rev_layer_outputs[:, -1, :, :, :] if bidirectional else None
            reweighted = torch.concat([last_fwd_feat, last_rev_feat], dim=1) if bidirectional else last_fwd_feat
            reweighted = torch.mean(reweighted, dim=1)
        else:
            if lengths is not None:
                layer_outputs = [torch.mean(layer_outputs[i, :length], dim=0) for i, length in enumerate(lengths)]
                # TODO: sequences are padded, so you need to reverse only the non-padded inputs
                if rev_layer_outputs is not None: rev_layer_outputs = [torch.mean(rev_layer_outputs[i, :length], dim=0) for i, length in enumerate(lengths)] 
                outputs = torch.stack(layer_outputs + rev_layer_outputs) if rev_layer_outputs is not None else torch.stack(layer_outputs)
                reweighted = outputs #?
            else:
                outputs = torch.cat([layer_outputs, rev_layer_outputs], dim=1) if rev_layer_outputs is not None else layer_outputs
                reweighted = torch.mean(outputs, dim=1)
    else:
        outputs = torch.cat([layer_outputs, rev_layer_outputs], dim=1) if rev_layer_outputs is not None else layer_outputs
        reweighted = attention(outputs, lengths)
        reweighted = torch.sum(reweighted, dim=1)
    return reweighted

class VectorAtt(nn.Module):

    def __init__(self, hidden_dim_size):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted hidden states.
        """
        super(VectorAtt, self).__init__()
        self.linear = nn.Linear(hidden_dim_size, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states, lengths=None):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous() # puts channels last
        weights = self.softmax(self.linear(hidden_states))
        b, t, c, h, w = weights.shape
        if lengths is not None: #TODO: gives backprop bug
            for i, length in enumerate(lengths):
                weights[i, t:] *= 0
        reweighted = weights * hidden_states
        return reweighted.permute(0, 1, 4, 2, 3).contiguous()

class TemporalAtt(nn.Module):

    def __init__(self, hidden_dim_size, d, r):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted timestamps. 

            Implementation based on the following blog post: 
            https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69
        """
        super(TemporalAtt, self).__init__()
        self.w_s1 = nn.Linear(in_features=hidden_dim_size, out_features=d, bias=False) 
        self.w_s2 = nn.Linear(in_features=d, out_features=r, bias=False) 
        nn.init.constant_(self.w_s1.weight, 1)
        nn.init.constant_(self.w_s2.weight, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        z1 = self.tanh(self.w_s1(hidden_states))
        attn_weights = self.softmax(self.w_s2(z1))
        reweighted = attn_weights * hidden_states
        reweighted = reweighted.permute(0, 1, 4, 2, 3).contiguous()
        return reweighted

class SelfAtt(nn.Module):
    def __init__(self, hidden_dim_size, dk, dv):
        """
            Self attention.
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width) 

            Implementation based on self attention in the following paper: 
            https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        """
        super(SelfAtt, self).__init__()
        self.dk = dk
        self.dv = dv

        self.w_q = nn.Linear(in_features=hidden_dim_size, out_features=dk, bias=False)
        self.w_k = nn.Linear(in_features=hidden_dim_size, out_features=dk, bias=False)
        self.w_v = nn.Linear(in_features=hidden_dim_size, out_features=dv, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        nb, nt, nr, nc, nh = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        queries = self.w_q(hidden_states)
        keys = self.w_k(hidden_states)
        values = self.w_v(hidden_states)
        
        attn = torch.mm(self.softmax(torch.mm(queries, torch.transpose(keys, 0, 1)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float).cuda())), values)      
        attn = attn.view(nb, nt, nr, nc, -1)
        attn = attn.permute(0, 1, 4, 2, 3).contiguous() 
        return attn

class ApplyAtt(nn.Module):
    def __init__(self, attn_type, hidden_dim_size, attn_dims):
        super(ApplyAtt, self).__init__()
        if attn_type == 'vector':
            self.attention = VectorAtt(hidden_dim_size)
        elif attn_type == 'temporal':
            self.attention = TemporalAtt(hidden_dim_size, attn_dims['d'], attn_dims['r'])
        elif attn_type == 'self':
            self.attention = SelfAtt(hidden_dim_size, attn_dims['dk'], attn_dims['dv'])
        elif attn_type == 'None':
            self.attention = None
        else:
            raise ValueError('Specified attention type is not compatible')

    def forward(self, hidden_states):
        attn_weighted = self.attention(hidden_states) if self.attention is not None else None
        return attn_weighted

