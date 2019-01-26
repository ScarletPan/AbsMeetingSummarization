import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def get_rnn(rnn_type, input_size, hidden_size, n_layers, dropout, bidirectional=False):
    if rnn_type in ['LSTM', 'GRU']:
        rnn = getattr(nn, rnn_type)(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=n_layers,
                                    batch_first=True,
                                    dropout=dropout,
                                    bidirectional=bidirectional)
    else:
        try:
            non_linearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        except KeyError:
            raise ValueError("""An invalid option for `--model` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn = nn.RNN(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=n_layers,
                     nonlinearity=non_linearity,
                     batch_first=True,
                     dropout=dropout,
                     bidirectional=bidirectional )
    return rnn


def get_hidden(rnn_type, _hidden):
    if rnn_type == "LSTM":
        return _hidden[0]
    else:
        return _hidden


def batch_padding(_input, sos_token=None, eos_token=None, variable=True):
    """
    :param: _input: list of Tensor (Variable), x[0].size = seq_len
    :param: _output: batch_size x max_len
    """
    assert sos_token is not None or eos_token is not None
    lens = [item.size(0) for item in _input]
    max_len = np.max(lens)
    if variable:
        pad_type = type(_input[0].data)
    else:
        pad_type = type(_input[0])
    # Padding before x
    if sos_token is not None:
        if variable:
            _output = [torch.cat([Variable(pad_type(
                 [sos_token] * (max_len - x.size(0)))), x])
                       if x.size(0) < max_len else x for x in _input]
        else:
            _output = [torch.cat([pad_type(
                 [sos_token] * (max_len - x.size(0))), x])
                       if x.size(0) < max_len else x for x in _input]
    # Padding after x
    else:
        if variable:
            _output = [torch.cat([x, Variable(pad_type(
                 [eos_token] * (max_len - x.size(0))))])
                       if x.size(0) < max_len else x for x in _input]
        else:
            _output = [torch.cat([x, pad_type(
                 [eos_token] * (max_len - x.size(0)))])
                       if x.size(0) < max_len else x for x in _input]
    return torch.cat([item.unsqueeze(0) for item in _output]), lens


def batch_unpadding(_input, lens, right=True):
    """
    :param: _input: m x n tensor
    :param: lens:   list with size of m
    :param: right:  eliminate padding from right if true
    """
    if right:
        return torch.cat([x[:lens[i]] for i, x in enumerate(_input) if lens[i] > 0 ])
    else:
        return torch.cat([x[-lens[i]:] for i, x in enumerate(_input) if lens[i] > 0 ])


def move_2d_tensor(_input, pad_val, stride=1, right=True, variable=False):
    """
    :param: _input:   m x n tensor
    :param: pad_val:  <int>
    :param: stride:   <int> padding num on 2nd dimension
    :param: right: padding direction, true to move right, false to move left
    :param: variable: if _input is a torch.autograd.variable
    """
    if variable:
        pad_type = type(_input[0].data)
    else:
        pad_type = type(_input[0])
    if pad_type == torch.LongTensor or pad_type == torch.cuda.LongTensor:
        np_type = np.int32
    else:
        np_type = np.float32
    m, n = _input.size()
    padding = pad_type((np.ones((m, stride), dtype=np_type) * pad_val).tolist())
    if variable:
        padding = Variable(padding)
    if right:
        return torch.cat([padding, _input], 1)[:, :n]
    else:
        return torch.cat([_input, padding, 1][:, :n])


def penalize(logits, penalized_tokens=None, penalized_val=np.log(2)):
    if len(logits.data.shape) == 2:
        logits[:, penalized_tokens] -= penalized_val
    else:
        logits[:, :, penalized_tokens] -= penalized_val


def batch_dot(a, b):
    assert a.size() == b.size()
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2))


def zeropadding_2d(x, target_size):
    a, b = x.size()
    a_t, b_t = target_size
    assert a_t >= a and b_t >= b
    _type = type(x)
    if _type == torch.autograd.Variable:
        _type = type(x.data)
    if b_t > b:
        z = _type(torch.zeros(a, b_t - b))
        t = torch.cat([x, z], 1)
    else:
        t = x
    if a_t > a:
        z = _type(torch.zeros(a_t - a, b_t))
        t = torch.cat([t, z])
    return t