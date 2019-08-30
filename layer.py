# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolution(Module):
    #定义卷积操作
    def __init__(self,input,output,bias=True):
        super(GraphConvolution,self).__init__()
        self.input = input
        self.output = output
        self.weight = Parameter(torch.FloatTensor(input, output))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input,adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


