import sys
from nnutil.layer import MLP
import util.init_path as init_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class baseline_network(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 network_shape=[],
                 define_std=True,
                 device="cpu",
                 transform=None
                 ):
        super(baseline_network,self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._device = device

        self._define_std = define_std
        if define_std:
            self.log_std = Variable(torch.rand(input_size).to(device))

        self._network_shape = network_shape

        self.transform = transform

        self.__build_network()
    def __build_network(self):

        self.network = MLP(self._input_size, self._network_shape).to(self._device)

    def forward(self, x):

        if self.transfrom is not None:
            x = self.transform(x)

        output = self.network.forward(x).squeeze(-1)[:-1]

        return output
