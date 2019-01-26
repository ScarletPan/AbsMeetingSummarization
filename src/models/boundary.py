import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.distributions import Categorical
from torch.nn.parameter import Parameter
from src.models.attention import Attn
from src.models.embedding import Embedder
from src.models.rnn import mLSTMCell
from src.models.utils import get_hidden
from src.models.beam import BeamSeqs


class BinaryGate(Function):
    """
    二值门单元
    forward中的二值门单元分为train和eval两种：
    train: 阈值为[0,1]内均匀分布的随机采样值随机的二值神经元，
    eval: 固定阈值为0.5的二值神经元
    backward中的二值门单元的导函数用identity函数
    """

    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        if inplace:
            output = input
        else:
            output = input.clone()
        ctx.thrs = random.uniform(0, 1) if training else 0.5
        output[output > ctx.thrs] = 1
        output[output <= ctx.thrs] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class BoundaryDetector(nn.Module):
    """
    Boundary Detector，边界检测模块
    """

    def __init__(self, i_features, h_features, s_features, inplace=False):
        super(BoundaryDetector, self).__init__()
        self.inplace = inplace
        self.Wsi = Parameter(torch.Tensor(s_features, i_features))
        self.Wsh = Parameter(torch.Tensor(s_features, h_features))
        self.bias = Parameter(torch.Tensor(s_features))
        self.vs = Parameter(torch.Tensor(1, s_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.Wsi.size(1))
        self.Wsi.data.uniform_(-stdv, stdv)
        self.Wsh.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.vs.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        z = F.linear(x, self.Wsi) + F.linear(h, self.Wsh) + self.bias
        z = F.sigmoid(F.linear(z, self.vs))
        return BinaryGate.apply(z, self.training, self.inplace)

    def __repr__(self):
        return self.__class__.__name__


class Encoder(nn.Module):
    """
    Boundary Aware Encoder Decoder
    """

    def __init__(self, encoder_vocab_size, embed_size, hidden_size, mid_size, extra_categorical_nums,
                 extra_embed_sizes, n_layers, dropout=0.5, use_cuda=True):
        super(Encoder, self).__init__()

        self.embedding = Embedder(encoder_vocab_size, embed_size,
                                          extra_categorical_nums, extra_embed_sizes, use_cuda)
        feat_size = self.embedding.size
        # Word Level encoder
        self.lstm1_cell = mLSTMCell(feat_size, hidden_size)
        self.lstm1_drop = nn.Dropout(p=dropout)
        # Boundary Detector
        self.bd = BoundaryDetector(feat_size, hidden_size, mid_size)
        # Boundary threshold
        # State Level encoder
        self.lstm2_cell = mLSTMCell(hidden_size, hidden_size)
        self.lstm2_drop = nn.Dropout(p=dropout)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm1_cell.reset_parameters()
        self.lstm2_cell.reset_parameters()

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        cell = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def forward(self, encoder_inputs, encoder_extra_inputs):
        b_size, seq_len = encoder_inputs.size()
        lstm1_h, lstm1_c = self.init_hidden(b_size)
        lstm2_h, lstm2_c = self.init_hidden(b_size)
        embeds = self.embedding(encoder_inputs, encoder_extra_inputs)

        lstm2_outputs = []
        states_positions = [0]
        for i in range(seq_len):
            s = self.bd(embeds[:, i, :], lstm1_h)
            lstm1_h, lstm1_c = self.lstm1_cell(embeds[:, i, :], (lstm1_h, lstm1_c))
            lstm1_h = self.lstm1_drop(lstm1_h)
            if s.data[0, 0] == 1:
                lstm2_input = lstm1_h * s
                lstm2_h, lstm2_c = self.lstm2_cell(lstm2_input, (lstm2_h, lstm2_c))
                lstm2_outputs.append(lstm2_h.unsqueeze(1))
                states_positions.append(i + 1)
            lstm1_h = lstm1_h * (1 - s)
            lstm1_c = lstm1_c * (1 - s)
        lstm2_outputs = torch.cat(lstm2_outputs, dim=1)

        return lstm2_outputs, (lstm2_h.unsqueeze(0), lstm2_c.unsqueeze(0)), states_positions


class BoundaryEncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, bd_mid_size,
                 extra_categorical_nums, extra_embed_sizes, n_layers=1, dropout=0.5, use_cuda=True):
        super(BoundaryEncoderDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = Encoder(encoder_vocab_size, embed_size, hidden_size, bd_mid_size, extra_categorical_nums,
                               extra_embed_sizes, n_layers, dropout, use_cuda)
        self.attn = Attn("dot", hidden_size, use_cuda)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embed_size)
        self.decoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers,
                               dropout=dropout, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, decoder_vocab_size)

    def load_word_vectors(self, pre_embedding):
        for i in range(pre_embedding.weight.size(0)):
            self.encoder.embedding.weight.data[i].copy_(pre_embedding.weight.data[i])
            self.decoder_embedding.weight.data[i].copy_(pre_embedding.weight.data[i])
        print("Load w2v Done")

    def reset_parameters(self):
        pass

    def forward(self, encoder_inputs, decoder_inputs, encoder_extra_inputs=None):
        self.decoder.flatten_parameters()
        encoder_outputs, last_hidden, state_pos = self.encoder(encoder_inputs, encoder_extra_inputs)
        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(decoder_embeds, last_hidden)
        decoder_outputs = self.drop(decoder_outputs)
        attn_weights = F.softmax(decoder_outputs.bmm(encoder_outputs.transpose(1, 2)), dim=2)
        # b x s2 x h * b x h x s1 => b x s2 x s1
        contexts = attn_weights.bmm(encoder_outputs)
        # b x s2 x s1 * b x s1 x h => b x s2 x h
        outlayer_inputs = torch.cat([decoder_outputs, contexts], dim=2)
        outlayer_outputs = F.tanh(self.concat(outlayer_inputs))
        decoder_outputs = self.fc(outlayer_outputs)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs, last_hidden

    def generate(self, encoder_inputs, decoder_start_input, max_len, encoder_extra_inputs=None, beam_size=1, eos_val=None):
        self.decoder.flatten_parameters()
        encoder_outputs, last_hidden, state_pos = self.encoder(encoder_inputs, encoder_extra_inputs)
        decoder_outputs = []
        if beam_size == 1:
            _input = decoder_start_input
            for i in range(max_len):
                decoder_embed = self.decoder_embedding(_input)
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                last_hidden_ = last_hidden[0]
                attn_weights = self.attn(last_hidden_[-1], encoder_outputs)
                decoder_output = decoder_output.squeeze(1)
                context = attn_weights.bmm(encoder_outputs).squeeze(1)
                outlayer_input = torch.cat([decoder_output, context], 1)
                outlayer_output = F.tanh(self.concat(outlayer_input))
                out = self.fc(outlayer_output)
                _, _input = torch.max(out, 1)
                _input = _input.unsqueeze(1)
                decoder_outputs.append(out.unsqueeze(1))
            return torch.max(F.log_softmax(torch.cat(decoder_outputs, 1), 2), dim=2)[1], last_hidden
        else:
            return self.beam_decode(last_hidden, encoder_outputs, decoder_start_input, max_len, beam_size, eos_val)

    def beam_decode(self, last_hidden, encoder_outputs, decoder_start_input, max_len, beam_size, eos_val):
        beamseqs = BeamSeqs(beam_size=beam_size)
        beamseqs.init_seqs(seqs=decoder_start_input[0], init_state=last_hidden)
        done = False
        for i in range(max_len):
            for j, (seqs, _, last_token, last_hidden) in enumerate(beamseqs.current_seqs):
                if beamseqs.check_and_add_to_terminal_seqs(j, eos_val):
                    if len(beamseqs.terminal_seqs) >= beam_size:
                        done = True
                        break
                    continue
                decoder_embed = self.decoder_embedding(last_token.unsqueeze(0))
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                last_hidden_ = get_hidden("LSTM", last_hidden)
                attn_weights = self.attn(last_hidden_[-1], encoder_outputs)
                decoder_output = decoder_output.squeeze(1)
                context = attn_weights.bmm(encoder_outputs).squeeze(1)
                outlayer_input = torch.cat([decoder_output, context], 1)
                outlayer_output = F.tanh(self.concat(outlayer_input))
                _output = F.log_softmax(self.fc(outlayer_output), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size)
                for k in range(beam_size):
                    score, token = scores.data[k], tokens[k]
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0), final_seqs[3]


class HybridBoundaryEncoder(nn.Module):
    """
    Hybrid Boundary Aware Encoder Decoder
    """

    def __init__(self, encoder_vocab_size, embed_size, hidden_size, mid_size, extra_categorical_nums,
                 extra_embed_sizes, n_layers, dropout=0.5, use_cuda=True):
        super(HybridBoundaryEncoder, self).__init__()

        self.embedding = Embedder(encoder_vocab_size, embed_size,
                                          extra_categorical_nums, extra_embed_sizes, use_cuda)
        feat_size = self.embedding.size
        self.lstm0 = nn.LSTM(input_size=feat_size, hidden_size=hidden_size,
                             num_layers=n_layers, dropout=dropout, batch_first=True)
        # Word Level encoder
        self.lstm1_cell = mLSTMCell(feat_size, hidden_size)
        self.lstm1_drop = nn.Dropout(p=dropout)
        # Boundary Detector
        self.bd = BoundaryDetector(feat_size, hidden_size, mid_size)
        # Boundary threshold
        # State Level encoder
        self.lstm2_cell = mLSTMCell(hidden_size, hidden_size)
        self.lstm2_drop = nn.Dropout(p=dropout)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm1_cell.reset_parameters()
        self.lstm2_cell.reset_parameters()

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        cell = Variable(torch.zeros(batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def forward(self, encoder_inputs, encoder_extra_inputs):
        self.lstm0.flatten_parameters()
        b_size, seq_len = encoder_inputs.size()
        lstm0_hidden = self.init_hidden(b_size)
        lstm0_hidden = (lstm0_hidden[0].unsqueeze(0), lstm0_hidden[1].unsqueeze(0))
        lstm1_h, lstm1_c = self.init_hidden(b_size)
        lstm2_h, lstm2_c = self.init_hidden(b_size)
        embeds = self.embedding(encoder_inputs, encoder_extra_inputs)

        lstm0_outputs, lstm0_hidden = self.lstm0(embeds, lstm0_hidden)
        lstm2_outputs = []
        states_positions = [0]
        for i in range(seq_len):
            s = self.bd(embeds[:, i, :], lstm1_h)
            lstm1_h, lstm1_c = self.lstm1_cell(embeds[:, i, :], (lstm1_h, lstm1_c))
            lstm1_h = self.lstm1_drop(lstm1_h)
            if i % 50 == 0 and i != 0: # s.data[0, 0] == 1:
                lstm2_input = lstm1_h * s
                lstm2_h, lstm2_c = self.lstm2_cell(lstm2_input, (lstm2_h, lstm2_c))
                lstm2_outputs.append(lstm2_h.unsqueeze(1))
                states_positions.append(i + 1)
            lstm1_h = lstm1_h * (1 - s)
            lstm1_c = lstm1_c * (1 - s)
        lstm2_outputs = torch.cat(lstm2_outputs, dim=1)

        return lstm0_outputs, lstm0_hidden, lstm2_outputs, \
               (lstm2_h.unsqueeze(0), lstm2_c.unsqueeze(0)), states_positions


class HybridBoundaryEncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, bd_mid_size,
                 extra_categorical_nums, extra_embed_sizes, n_layers=1, dropout=0.5, use_cuda=True):
        super(HybridBoundaryEncoderDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = HybridBoundaryEncoder(encoder_vocab_size, embed_size, hidden_size, bd_mid_size, extra_categorical_nums,
                               extra_embed_sizes, n_layers, dropout, use_cuda)
        self.hidden_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = Attn("dot", hidden_size, use_cuda)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embed_size)
        self.decoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers,
                               dropout=dropout, batch_first=True)
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.fc = nn.Linear(hidden_size, decoder_vocab_size)

    def load_word_vectors(self, pre_embedding):
        for i in range(pre_embedding.weight.size(0)):
            self.encoder.embedding.weight.data[i].copy_(pre_embedding.weight.data[i])
            self.decoder_embedding.weight.data[i].copy_(pre_embedding.weight.data[i])
        print("Load w2v Done")

    def reset_parameters(self):
        pass

    def init_params(self, model):
        layers = ["encoder", "hidden_linear", "decoder_embedding", "attn",
                  "decoder", "concat", "fc"]
        for layer in layers:
            tp = model.__getattr__(layer).named_parameters()
            for name, weights in self.__getattr__(layer).named_parameters():
                _, weights_ = next(tp)
                weights.data.copy_(weights_.data)

    def forward(self, encoder_inputs, decoder_inputs, encoder_extra_inputs=None):
        self.decoder.flatten_parameters()
        lstm0_outputs, lstm0_hidden, lstm2_outputs, lstm2_hidden, state_pos = self.encoder(
            encoder_inputs, encoder_extra_inputs)
        last_hidden = (F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[0], lstm2_hidden[0]], dim=2))),
                       F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[1], lstm2_hidden[1]], dim=2))))
        # last_hidden = (lstm0_hidden[0] * lstm2_hidden[0],
        #                lstm0_hidden[1] * lstm2_hidden[1])
        # last_hidden = (torch.cat([lstm0_hidden[0], lstm2_hidden[0]], dim=0),
        #                torch.cat([lstm0_hidden[1], lstm2_hidden[1]], dim=0))
        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(decoder_embeds, last_hidden)
        decoder_outputs = self.drop(decoder_outputs)
        # First Fully Connected LSTM Attention
        attn_weights_0 = F.softmax(decoder_outputs.bmm(lstm0_outputs.transpose(1, 2)), dim=2)
        contexts_0 = attn_weights_0.bmm(lstm0_outputs)
        # Second Hierachical Boundary LSTM Attention
        attn_weights_2 = F.softmax(decoder_outputs.bmm(lstm2_outputs.transpose(1, 2)), dim=2)
        contexts_2 = attn_weights_2.bmm(lstm2_outputs)
        outlayer_inputs = torch.cat([decoder_outputs, contexts_0, contexts_2], dim=2)
        outlayer_outputs = F.tanh(self.concat(outlayer_inputs))
        decoder_outputs = self.fc(outlayer_outputs)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs, last_hidden

    def generate(self, encoder_inputs, decoder_start_input, max_len, encoder_extra_inputs=None, beam_size=1, eos_val=None):
        self.decoder.flatten_parameters()
        lstm0_outputs, lstm0_hidden, lstm2_outputs, lstm2_hidden, state_pos = self.encoder(
            encoder_inputs, encoder_extra_inputs)

        last_hidden = (F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[0], lstm2_hidden[0]], dim=2))),
                       F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[1], lstm2_hidden[1]], dim=2))))
        # last_hidden = (lstm0_hidden[0] * lstm2_hidden[0],
        #                lstm0_hidden[1] * lstm2_hidden[1])
        # last_hidden = (torch.cat([lstm0_hidden[0], lstm2_hidden[0]], dim=0),
        #                torch.cat([lstm0_hidden[1], lstm2_hidden[1]], dim=0))
        decoder_outputs = []
        if beam_size == 1:
            _input = decoder_start_input
            for i in range(max_len):
                decoder_embed = self.decoder_embedding(_input)
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                last_hidden_ = last_hidden[0]
                attn_weights_0 = self.attn(last_hidden_[-1], lstm0_outputs)
                context_0 = attn_weights_0.bmm(lstm0_outputs).squeeze(1)
                attn_weights_2 = self.attn(last_hidden_[-1], lstm2_outputs)
                context_2 = attn_weights_2.bmm(lstm2_outputs).squeeze(1)
                decoder_output = decoder_output.squeeze(1)
                outlayer_input = torch.cat([decoder_output, context_0, context_2], 1)
                outlayer_output = F.tanh(self.concat(outlayer_input))
                out = self.fc(outlayer_output)
                _, _input = torch.max(out, 1)
                _input = _input.unsqueeze(1)
                decoder_outputs.append(out.unsqueeze(1))
            return torch.max(F.log_softmax(torch.cat(decoder_outputs, 1), 2), dim=2)[1], last_hidden
        else:
            return self.beam_decode(last_hidden, lstm0_outputs, lstm2_outputs, decoder_start_input, max_len, beam_size, eos_val)

    def beam_decode(self, last_hidden, lstm0_outputs, lstm2_outputs, decoder_start_input, max_len, beam_size, eos_val):
        beamseqs = BeamSeqs(beam_size=beam_size)
        beamseqs.init_seqs(seqs=decoder_start_input[0], init_state=last_hidden)
        done = False
        for i in range(max_len):
            for j, (seqs, _, last_token, last_hidden) in enumerate(beamseqs.current_seqs):
                if beamseqs.check_and_add_to_terminal_seqs(j, eos_val):
                    if len(beamseqs.terminal_seqs) >= beam_size:
                        done = True
                        break
                    continue
                decoder_embed = self.decoder_embedding(last_token.unsqueeze(0))
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                last_hidden_ = get_hidden("LSTM", last_hidden)
                attn_weights_0 = self.attn(last_hidden_[-1], lstm0_outputs)
                context_0 = attn_weights_0.bmm(lstm0_outputs).squeeze(1)
                attn_weights_2 = self.attn(last_hidden_[-1], lstm2_outputs)
                context_2 = attn_weights_2.bmm(lstm2_outputs).squeeze(1)
                decoder_output = decoder_output.squeeze(1)
                outlayer_input = torch.cat([decoder_output, context_0, context_2], 1)
                outlayer_output = F.tanh(self.concat(outlayer_input))
                _output = F.log_softmax(self.fc(outlayer_output), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size)
                for k in range(beam_size):
                    score, token = scores.data[k], tokens[k]
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0), final_seqs[3]

    def sample(self, encoder_inputs, decoder_start_input, max_len, encoder_extra_inputs=None):
        self.decoder.flatten_parameters()
        lstm0_outputs, lstm0_hidden, lstm2_outputs, lstm2_hidden, state_pos = self.encoder(
            encoder_inputs, encoder_extra_inputs)
        last_hidden = (F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[0], lstm2_hidden[0]], dim=2))),
                       F.tanh(self.hidden_linear(torch.cat([lstm0_hidden[1], lstm2_hidden[1]], dim=2))))
        _input = decoder_start_input
        seqs = []
        seq_logprobs = []
        for i in range(max_len):
            decoder_embed = self.decoder_embedding(_input)
            decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
            last_hidden_ = last_hidden[0]
            attn_weights_0 = self.attn(last_hidden_[-1], lstm0_outputs)
            context_0 = attn_weights_0.bmm(lstm0_outputs).squeeze(1)
            attn_weights_2 = self.attn(last_hidden_[-1], lstm2_outputs)
            context_2 = attn_weights_2.bmm(lstm2_outputs).squeeze(1)
            decoder_output = decoder_output.squeeze(1)
            outlayer_input = torch.cat([decoder_output, context_0, context_2], 1)
            outlayer_output = F.tanh(self.concat(outlayer_input))
            out = self.fc(outlayer_output)
            log_probs = F.log_softmax(out, 1)
            m = Categorical(torch.exp(log_probs))
            # it = torch.multinomial(torch.exp(log_probs.data), 1)
            it = m.sample()
            # seq_logprobs.append(torch.gather(log_probs, 1, _input))
            seq_logprobs.append(m.log_prob(it).unsqueeze(1))
            _input = it.unsqueeze(1)
            seqs.append(_input)
        return torch.cat(seqs, 1), torch.cat(seq_logprobs, 1), last_hidden



if __name__ == "__main__":
    encoder_inputs = Variable(torch.LongTensor([[1, 2, 3, 4], [4, 5, 6, 7]])).cuda()
    decoder_inputs = Variable(torch.LongTensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])).cuda()
    model = BoundaryEncoderDecoder(encoder_vocab_size=20, decoder_vocab_size=20, embed_size=7, hidden_size=15,
                                   bd_mid_size=13).cuda()
    outputs, last_hidden = model(encoder_inputs, decoder_inputs)
    t = outputs.mean()
    t.backward()
    pass