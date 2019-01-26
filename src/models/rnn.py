import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input Gate
        self.linear_i = nn.Linear(input_size + hidden_size, hidden_size)
        # Forget Gate
        self.linear_f = nn.Linear(input_size + hidden_size, hidden_size)
        # G Gate
        self.linear_g = nn.Linear(input_size + hidden_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        """
        :param input: batch_size x input_size
        :param h: (batch_size x hidden_size, batch_size x hidden_size)
        :return: hidden, cell
        """
        hidden, cell = h
        c = torch.cat([input, hidden], dim=1)
        it = F.sigmoid(self.linear_i(c))    # batch_size x hidden_size
        f_tmp = self.linear_f(c)
        ft = F.sigmoid(f_tmp)    # batch_size x hidden_size
        gt = F.tanh(self.linear_g(c))       # batch_size x hidden_size
        cell = ft * cell + it * gt
        ot = F.tanh(f_tmp)
        hidden = ot * F.tanh(cell)
        return hidden, cell