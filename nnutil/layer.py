import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, output_dim = None, activation = "relu", dropout_p=None, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation =="tanh":
                    layers.append(nn.Tanh())

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        if output_dim is not None:
            layers.append(nn.Linear(input_dim,output_dim))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()

        self.w_ir = nn.Linear(input_dim, hidden_dim)
        self.w_iu = nn.Linear(input_dim, hidden_dim)
        self.w_in = nn.Linear(input_dim, hidden_dim)

        self.w_hr = nn.Linear(hidden_dim, hidden_dim)
        self.w_hu = nn.Linear(hidden_dim, hidden_dim)
        self.w_hn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):

        rt = torch.sigmoid(self.w_ir(x) + self.w_hr(h))
        ut = torch.sigmoid(self.w_iu(x) + self.w_hu(h))
        nt = torch.tanh(self.w_in(x) + rt * self.w_hn(h))
        ht = (1 - ut) * nt + ut * h

        return ht

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.w_ii = nn.Linear(input_dim, hidden_dim)
        self.w_if = nn.Linear(input_dim, hidden_dim)
        self.w_ig = nn.Linear(input_dim, hidden_dim)
        self.w_io = nn.Linear(input_dim, hidden_dim)

        self.w_hi = nn.Linear(hidden_dim, hidden_dim)
        self.w_hf = nn.Linear(hidden_dim, hidden_dim)
        self.w_hg = nn.Linear(hidden_dim, hidden_dim)
        self.w_ho = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h, c):

        it = torch.sigmoid(self.w_ii(x) + self.w_hi(h))
        ft = torch.sigmoid(self.w_if(x) + self.w_hf(h))
        gt = torch.tanh(self.w_ig(x) + self.w_hg(h))
        ot = torch.sigmoid(self.w_io(x) + self.w_ho(h))
        ct =  ft * c + it * gt
        ht = ot * torch.tanh(ct)
        return ht, ct