import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional, init

class RecurrentNorm2d(nn.Module):
    """
    Normalization Module which keeps track of separate statistics for each timestep as described in
    https://arxiv.org/pdf/1603.09025.pdf
    
    Currently only configured to use BN
    
    TODO:
    - Add support for Layer Norm
    - Add support for Group Norm

    based on the work from https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py

    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(RecurrentNorm2d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            # no bias term as described in the paper
            self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            # initialize to .1 as advocated in the paper
            self.weight.data = torch.ones(self.num_features) * .1
            
    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))



# class RecurrentNorm2d(nn.Module):
#     """
#     Normalization Module which keeps track of separate statistics for each timestep as described in
#     https://arxiv.org/pdf/1603.09025.pdf
    
#     Currently only configured to use BN
    
#     TODO:
#     - Add support for Layer Norm
#     - Add support for Group Norm
    
#     """

#     def __init__(self, num_features, max_timesteps, eps=1e-5, momentum=0.1,
#                  affine=True): 
#         super(RecurrentNorm2d, self).__init__()
#         self.num_features = num_features
#         self.max_timesteps = max_timesteps
#         self.affine = affine
#         self.eps = eps
#         self.momentum = momentum
#         self.norms = []
        
#         for i in range(self.max_timesteps):
#             # TODO: condition on the type of normalization
#             self.norms.append(nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine))
#             self.norms[i].reset_parameters()
            
#         self.norms = nn.ModuleList(self.norms)
    
#     def _check_input_dim(self, input_):
#         if input_.size(1) != self.num_features:
#             raise ValueError('got {}-feature tensor, expected {}'
#                              .format(input_.size(1), self.num_features))

#     def forward(self, input_, timestep):
#         self._check_input_dim(input_)
#         if timestep >= self.max_timesteps:
#             timestep = self.max_timesteps - 1
#         return self.norms[timestep](input_)
    