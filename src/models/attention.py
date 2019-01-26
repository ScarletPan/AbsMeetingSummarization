import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=True):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.zeros((1, self.hidden_size)))

    def init_param(self):
        self.other.data.uniform_(-0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Calculate energies for each encoder output
        if self.method == "dot":
            attn_energies = torch.bmm(encoder_outputs, hidden.unsqueeze(2))
        elif self.method == "general":
            energies = self.attn(hidden).unsqueeze(2)
            attn_energies = torch.bmm(encoder_outputs, energies)
        elif self.method == "concat":
            concat = torch.cat([hidden.unsqueeze(1).repeat(1, seq_len, 1), encoder_outputs], dim=2)
            energy = F.tanh(self.attn(concat))
            other_batch = self.other.repeat(batch_size, 1).unsqueeze(2)
            attn_energies = torch.bmm(energy, other_batch)

        # Normalize energies to weights in range 0 to 1, resize to batch_size x 1 x seq_len
        return F.softmax(attn_energies, dim=1).transpose(1, 2)


class IntraEncoderAttn(nn.Module):
    def __init__(self, hidden_size, use_cuda=True):
        super(IntraEncoderAttn, self).__init__()

        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.attn_linear = nn.Linear(self.hidden_size, hidden_size)
        self.time_step = 0
        self.past_attn_energies_exps = None

    def reset_attn(self):
        self.time_step = 0
        self.past_attns = None

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)

        # Calculate energies for each encoder output
        energies = self.attn_linear(encoder_outputs)
        attn_energies = torch.bmm(energies, hidden.unsqueeze(2)).squeeze(2)

        if self.time_step == 0:
            attn_energies_exp = torch.exp(attn_energies)
            self.past_attn_energies_exps = attn_energies_exp.unsqueeze(1)   # batch_size x 1 x seq_len
        else:
            attn_energies_exp = torch.exp(attn_energies) / torch.sum(self.past_attn_energies_exps, dim=1)
            self.past_attn_energies_exps = torch.cat(
                [self.past_attn_energies_exps, attn_energies_exp.unsqueeze(1)], dim=1)
        self.time_step += 1
        # Normalize energies to weights in range 0 to 1, resize to batch_size x 1 x seq_len
        s = torch.sum(attn_energies_exp, dim=1).unsqueeze(1).repeat(1, seq_len)
        attn = attn_energies_exp / s
        return attn.unsqueeze(1)
