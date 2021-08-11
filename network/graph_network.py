import sys
from nnutil.layer import MLP, GRU, LSTM
import util.init_path as init_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_max, scatter_add


# import numpy as np

class graph_network(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 network_shape=[],
                 define_std=True,
                 device="cpu",
                 args=None
                 ):
        super(graph_network, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._device = device

        self._hidden_size = args["hidden_size"]
        self._output_type = args["output_type"]  # default: unified"

        ##The same type node need to be put together

        self._node_type = args["node_type"]  # assign a list
        self._node_type_num = len(set(self._node_type))
        self._node_num = len(self._node_type)

        self._state_update_func = args["state_update_func"]  # default: mlp
        self._state_update_type = args["state_update_type"]  # default: shared
        self._edge_update_type = args["edge_update_type"]  # default: separate
        self._node_agg_fn_type = args["node_agg_fn_type"]  # default: sum

        assert self._output_type in ["unified", "shared", "separate"]
        assert self._state_update_type in ["unified", "shared", "separate"]
        assert self._state_update_func in ["mlp", "gru", "lstm"]
        assert self._edge_update_type in ["unified", "separate"]
        assert self._node_agg_fn_type in ["mean", "max", "sum"]

        self._propagation_step = args["propagation_step"]  # default: 4

        self._define_std = define_std
        if define_std:
            self.log_std = Variable(torch.rand(input_size).to(device))

        self._network_shape = network_shape

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._node_type_stat = [0] * self._node_type_num
            for i in range(self._node_num):
                self._node_type_stat[self._node_type[i]] += 1

            temp = [0] * self._node_num
            self._inv_mask = [0] * self._node_num
            for i in range(self._node_num):
                index = self._node_type[i]
                for j in range(index):
                    self._inv_mask[i] += self._node_type_stat[j]
                self._inv_mask[i] += temp[index]
                temp[index] += 1

            self._inv_mask = torch.LongTensor(self._inv_mask).to(device)

        self._node_type = torch.LongTensor(self._node_type).to(device)

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._mask = []
            for i in range(self._node_type_num):
                flag_i = self._node_type == i
                self._mask.append(flag_i)

        self.edge_attr = args["edge_attr"]

        self.transform = args["transform"]
        self.__build_network()

    def __build_network(self):

        # input network
        self.input_model = MLP(self._input_size, [self._hidden_size]).to(self._device)

        # output network
        self.output_model = []
        if self._output_type == "unified":
            self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "separate":
            for i in range(self._node_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "shared":
            for i in range(self._node_type_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        # edge network
        self.edge_model = []
        if self._edge_update_type == "unified":
            self.edge_model.append(MLP(self._hidden_size, self._network_shape, self._hidden_size).to(self._device))
        else:  # case: separate
            for i in range(self._input_size):
                self.edge_model.append(MLP(self._hidden_size, self._network_shape, self._hidden_size).to(self._device))

        # aggregation function
        if self._node_agg_fn_type == 'mean':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_mean(src, output_index, dim=0, dim_size=x_size)

        elif self._node_agg_fn_type == 'max':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_max(src, output_index, dim=0, dim_size=x_size)[
                0]

        elif self._node_agg_fn_type == 'sum':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_add(src, output_index, dim=0, dim_size=x_size)

        # state update
        self.node_model = []
        if self._state_update_func == "mlp":

            if self._state_update_type == "unified":
                self.node_model += [MLP(self._hidden_size * 2, self._network_shape).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(MLP(self._hidden_size * 2, self._network_shape).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(MLP(self._hidden_size * 2, self._network_shape).to(self._device))

        elif self._state_update_func == "gru":

            if self._state_update_type == "unified":
                self.node_model += [GRU(self._hidden_size, self._hidden_size).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(GRU(self._hidden_size, self._hidden_size).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(GRU(self._hidden_size, self._hidden_size).to(self._device))

        else:  # case: lstm

            if self._state_update_type == "unified":
                self.node_model += [LSTM(self._hidden_size, self._hidden_size).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(LSTM(self._hidden_size, self._hidden_size).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(LSTM(self._hidden_size, self._hidden_size).to(self._device))

    def forward(self, x):

        if self.transform is not None:
            x = self.transform(x)

        edge_attr = self.edge_attr
        assert x.size(0) % self._node_num == 0
        N = x.size(0) // self._node_num
        NN = x.size(0)
        EN = edge_attr.size(1)

        ## input model
        h = self.input_model.forward(x)
        in_node, out_node = edge_attr

        for k in range(self._propagation_step):

            ## message computation
            if self._edge_update_type == "unified":
                m = self.edge_model[0].forward(h)
                m = m[in_node]

            else:

                h_r = h.view(self._node_num, N, self._hidden_size)
                m = []
                for i in range(self._node_num):
                    out = self.edge_model[i].forward(h_r[i])
                    m.append(out)
                m = torch.stack(m, dim=0).view(NN, self._hidden_size)
                m = m[in_node]

            ## message aggregation
            m = self.node_agg_fn(m, out_node, NN)

            if self._state_update_func == "lstm":
                c = h

            ## state update
            if self._state_update_func == "mlp":

                if self._state_update_type == "unified":
                    h = self.node_model[0].forward(torch.cat([h, m], dim=-1))

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_num):
                        out = self.node_model[i].forward(torch.cat([h_r[i], m[i]], dim=-1))
                        out_agg.append(out)
                    h = torch.stack(out_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_type_num):
                        out = self.node_model[i].forward(torch.cat([h_r[self._mask[i]], m[self._mask[i]]], dim=-1))
                        out_agg.append(out)
                    h = torch.cat(out_agg, dim=0)
                    h = h[self._inv_mask].view(NN, -1)

            elif self._state_update_func == "gru":

                if self._state_update_type == "unified":
                    h = self.node_model[0].forward(h, m)

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_num):
                        out = self.node_model[i].forward(h_r[i], m[i])
                        out_agg.append(out)
                    h = torch.stack(out_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_type_num):
                        out = self.node_model[i].forward(h_r[self._mask[i]], m[self._mask[i]])
                        out_agg.append(out)
                    h = torch.cat(out_agg, dim=0)
                    h = h[self._inv_mask].view(NN, -1)

            elif self._state_update_func == "lstm":

                if self._state_update_type == "unified":
                    h, c = self.node_model[0].forward(h, m, c)

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    c_r = c.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_h_agg = []
                    out_c_agg = []
                    for i in range(self._node_num):
                        out_h, out_c = self.node_model[i].forward(h_r[i], m[i], c_r[i])
                        out_h_agg.append(out_h)
                        out_c_agg.append(out_c)
                    h = torch.stack(out_h_agg, dim=0).view(NN, -1)
                    c = torch.stack(out_c_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    c_r = c.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_h_agg = []
                    out_c_agg = []
                    for i in range(self._node_type_num):
                        out_h, out_c = self.node_model[i].forward(h_r[self._mask[i]], m[self._mask[i]], c_r[self._mask[i]])
                        out_h_agg.append(out_h)
                        out_c_agg.append(out_c)
                    h = torch.cat(out_h_agg, dim=0).view(NN, -1)
                    c = torch.cat(out_c_agg, dim=0).view(NN, -1)


        ## output model
        if self._output_type == "unified":
            output = self.output_model[0].forward(h)

        elif self._output_type == "separate":
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_num):
                out = self.output_model[i].forward(h[i])
                output.append(out)
            output = torch.stack(output, dim=0).view(NN)

        else:
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_type_num):
                out = self.output_model[i].forward(h[self._mask[i]])
                output.append(out)
            output = torch.cat(output, dim=0)
            output = output[self._inv_mask].view(NN)

        return output[:-1]

class graph_network_noaction(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 network_shape=[],
                 define_std=True,
                 device="cpu",
                 args=None
                 ):
        super(graph_network_noaction, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._device = device

        self._hidden_size = args["hidden_size"]
        self._output_type = args["output_type"]  # default: unified"

        ##The same type node need to be put together

        self._node_type = args["node_type"]  # assign a list
        self._node_type_num = len(set(self._node_type))
        self._node_num = len(self._node_type)

        self._state_update_func = args["state_update_func"]  # default: mlp
        self._state_update_type = args["state_update_type"]  # default: shared
        self._edge_update_type = args["edge_update_type"]  # default: separate
        self._node_agg_fn_type = args["node_agg_fn_type"]  # default: sum

        assert self._output_type in ["unified", "shared", "separate"]
        assert self._state_update_type in ["unified", "shared", "separate"]
        assert self._state_update_func in ["mlp", "gru", "lstm"]
        assert self._edge_update_type in ["unified", "separate"]
        assert self._node_agg_fn_type in ["mean", "max", "sum"]

        self._propagation_step = args["propagation_step"]  # default: 4

        self._define_std = define_std
        if define_std:
            self.log_std = Variable(torch.rand(input_size).to(device))

        self._network_shape = network_shape

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._node_type_stat = [0] * self._node_type_num
            for i in range(self._node_num):
                self._node_type_stat[self._node_type[i]] += 1

            temp = [0] * self._node_num
            self._inv_mask = [0] * self._node_num
            for i in range(self._node_num):
                index = self._node_type[i]
                for j in range(index):
                    self._inv_mask[i] += self._node_type_stat[j]
                self._inv_mask[i] += temp[index]
                temp[index] += 1

            self._inv_mask = torch.LongTensor(self._inv_mask).to(device)

        self._node_type = torch.LongTensor(self._node_type).to(device)

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._mask = []
            for i in range(self._node_type_num):
                flag_i = self._node_type == i
                self._mask.append(flag_i)

        self.edge_attr = args["edge_attr"]

        self.transform = args["transform"]
        self.__build_network()

    def __build_network(self):

        # input network
        self.input_model = MLP(self._input_size, [self._hidden_size]).to(self._device)

        # output network
        """
        self.output_model = []
        if self._output_type == "unified":
            self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "separate":
            for i in range(self._node_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "shared":
            for i in range(self._node_type_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))
        """

        # edge network
        self.edge_model = []
        if self._edge_update_type == "unified":
            self.edge_model.append(MLP(self._hidden_size, self._network_shape, self._hidden_size).to(self._device))
        else:  # case: separate
            for i in range(self._input_size):
                self.edge_model.append(MLP(self._hidden_size, self._network_shape, self._hidden_size).to(self._device))

        # aggregation function
        if self._node_agg_fn_type == 'mean':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_mean(src, output_index, dim=0, dim_size=x_size)

        elif self._node_agg_fn_type == 'max':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_max(src, output_index, dim=0, dim_size=x_size)[
                0]

        elif self._node_agg_fn_type == 'sum':
            self.node_agg_fn = lambda src, output_index, x_size: scatter_add(src, output_index, dim=0, dim_size=x_size)

        # state update
        self.node_model = []
        if self._state_update_func == "mlp":

            if self._state_update_type == "unified":
                self.node_model += [MLP(self._hidden_size * 2, self._network_shape).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(MLP(self._hidden_size * 2, self._network_shape).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(MLP(self._hidden_size * 2, self._network_shape).to(self._device))

        elif self._state_update_func == "gru":

            if self._state_update_type == "unified":
                self.node_model += [GRU(self._hidden_size, self._hidden_size).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(GRU(self._hidden_size, self._hidden_size).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(GRU(self._hidden_size, self._hidden_size).to(self._device))

        else:  # case: lstm

            if self._state_update_type == "unified":
                self.node_model += [LSTM(self._hidden_size, self._hidden_size).to(self._device)]

            elif self._state_update_type == "separate":
                for i in range(self._node_num):
                    self.node_model.append(LSTM(self._hidden_size, self._hidden_size).to(self._device))

            elif self._state_update_type == "shared":
                for i in range(self._node_type_num):
                    self.node_model.append(LSTM(self._hidden_size, self._hidden_size).to(self._device))

    def forward(self, x):

        if self.transform is not None:
            x = self.transform(x)

        edge_attr = self.edge_attr
        assert x.size(0) % self._node_num == 0
        N = x.size(0) // self._node_num
        NN = x.size(0)
        EN = edge_attr.size(1)

        ## input model
        h = self.input_model.forward(x)
        in_node, out_node = edge_attr

        for k in range(self._propagation_step):

            ## message computation
            if self._edge_update_type == "unified":
                m = self.edge_model[0].forward(h)
                m = m[in_node]

            else:

                h_r = h.view(self._node_num, N, self._hidden_size)
                m = []
                for i in range(self._node_num):
                    out = self.edge_model[i].forward(h_r[i])
                    m.append(out)
                m = torch.stack(m, dim=0).view(NN, self._hidden_size)
                m = m[in_node]

            ## message aggregation
            m = self.node_agg_fn(m, out_node, NN)

            if self._state_update_func == "lstm":
                c = h

            ## state update
            if self._state_update_func == "mlp":

                if self._state_update_type == "unified":
                    h = self.node_model[0].forward(torch.cat([h, m], dim=-1))

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_num):
                        out = self.node_model[i].forward(torch.cat([h_r[i], m[i]], dim=-1))
                        out_agg.append(out)
                    h = torch.stack(out_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_type_num):
                        out = self.node_model[i].forward(torch.cat([h_r[self._mask[i]], m[self._mask[i]]], dim=-1))
                        out_agg.append(out)
                    h = torch.cat(out_agg, dim=0)
                    h = h[self._inv_mask].view(NN, -1)

            elif self._state_update_func == "gru":

                if self._state_update_type == "unified":
                    h = self.node_model[0].forward(h, m)

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_num):
                        out = self.node_model[i].forward(h_r[i], m[i])
                        out_agg.append(out)
                    h = torch.stack(out_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_agg = []
                    for i in range(self._node_type_num):
                        out = self.node_model[i].forward(h_r[self._mask[i]], m[self._mask[i]])
                        out_agg.append(out)
                    h = torch.cat(out_agg, dim=0)
                    h = h[self._inv_mask].view(NN, -1)

            else:

                if self._state_update_type == "unified":
                    h, c = self.node_model[0].forward(h, m, c)

                elif self._state_update_type == "separate":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    c_r = c.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_h_agg = []
                    out_c_agg = []
                    for i in range(self._node_num):
                        out_h, out_c = self.node_model[i].forward(h_r[i], m[i], c_r[i])
                        out_h_agg.append(out_h)
                        out_c_agg.append(out_c)
                    h = torch.stack(out_h_agg, dim=0).view(NN, -1)
                    c = torch.stack(out_c_agg, dim=0).view(NN, -1)

                elif self._state_update_type == "shared":
                    h_r = h.view(self._node_num, N, self._hidden_size)
                    c_r = c.view(self._node_num, N, self._hidden_size)
                    m = m.view(self._node_num, N, self._hidden_size)
                    out_h_agg = []
                    out_c_agg = []
                    for i in range(self._node_type_num):
                        out_h, out_c = self.node_model[i].forward(h_r[self._mask[i]], m[self._mask[i]], c_r[self._mask[i]])
                        out_h_agg.append(out_h)
                        out_c_agg.append(out_c)
                    h = torch.cat(out_h_agg, dim=0).view(NN, -1)
                    c = torch.cat(out_c_agg, dim=0).view(NN, -1)

        return h
        ## output model
        """
        if self._output_type == "unified":
            output = self.output_model[0].forward(h)

        elif self._output_type == "separate":
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_num):
                out = self.output_model[i].forward(h[i])
                output.append(out)
            output = torch.stack(output, dim=0).view(NN)

        else:
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_type_num):
                out = self.output_model[i].forward(h[self._mask[i]])
                output.append(out)
            output = torch.cat(output, dim=0)
            output = output[self._inv_mask].view(NN)

        return output[:-1]
        """

class graph_network_action(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 network_shape=[],
                 device="cpu",
                 args=None
                 ):
        super(graph_network_action, self).__init__()

        self._output_size = output_size
        self._base_dir = init_path.get_abs_base_dir()
        self._device = device

        self._hidden_size = args["hidden_size"]
        self._output_type = args["output_type"]  # default: unified"

        ##The same type node need to be put together

        self._node_type = args["node_type"]  # assign a list
        self._node_type_num = len(set(self._node_type))
        self._node_num = len(self._node_type)

        assert self._output_type in ["unified", "shared", "separate"]



        self._network_shape = network_shape

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._node_type_stat = [0] * self._node_type_num
            for i in range(self._node_num):
                self._node_type_stat[self._node_type[i]] += 1

            temp = [0] * self._node_num
            self._inv_mask = [0] * self._node_num
            for i in range(self._node_num):
                index = self._node_type[i]
                for j in range(index):
                    self._inv_mask[i] += self._node_type_stat[j]
                self._inv_mask[i] += temp[index]
                temp[index] += 1

            self._inv_mask = torch.LongTensor(self._inv_mask).to(device)

        self._node_type = torch.LongTensor(self._node_type).to(device)

        if self._state_update_type == "shared" or self._output_type == "shared":
            self._mask = []
            for i in range(self._node_type_num):
                flag_i = self._node_type == i
                self._mask.append(flag_i)


        self.__build_network()

    def __build_network(self):

        # output network
        self.output_model = []
        if self._output_type == "unified":
            self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "separate":
            for i in range(self._node_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))

        elif self._output_type == "shared":
            for i in range(self._node_type_num):
                self.output_model.append(MLP(self._hidden_size, [], self._output_size).to(self._device))



    def forward(self, h):

        N = h.size(0) // self._node_num
        NN = h.size(0)
        ## output model

        if self._output_type == "unified":
            output = self.output_model[0].forward(h)

        elif self._output_type == "separate":
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_num):
                out = self.output_model[i].forward(h[i])
                output.append(out)
            output = torch.stack(output, dim=0).view(NN)

        else:
            h = h.view(self._node_num, N, self._hidden_size)
            output = []
            for i in range(self._node_type_num):
                out = self.output_model[i].forward(h[self._mask[i]])
                output.append(out)
            output = torch.cat(output, dim=0)
            output = output[self._inv_mask].view(NN)

        return output[:-1]
