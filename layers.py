import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from parser1 import parameter_parser
from torch.nn.parameter import Parameter
from torch.nn import init

args = parameter_parser()


import numpy as np


# GraphConv layers and models
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 
 
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)


    def forward(self, input, adj):
        batch = input.shape[0]
        output = []
        for i in range(batch):
            nums = torch.mm(adj[i],adj[i])
            nums_ = torch.mm(nums,adj[i])
            if len(input.shape) == 3:
                support = torch.mm(input[i], self.weight)
                out = torch.spmm(nums_, support)
                output.append(out.data.numpy())
            else:
                support = torch.mm(input[i], self.weight)
                out = torch.spmm(nums_, support[i])
                output.append(out.data.numpy())
        output = torch.from_numpy(np.array(output))
        if self.bias is not None:
            output = output + self.bias
            return output
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'
