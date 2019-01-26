import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.attention import Attn
from src.models.utils import get_rnn, get_hidden, penalize


class BeamSeqs(object):
    def __init__(self, beam_size):
        self.current_seqs = []
        self.new_seqs = []
        self.terminal_seqs = []
        self.beam_size = beam_size

    def init_seqs(self, seqs, init_state):
        latest_token = seqs[-1]
        init_score = 0
        self.current_seqs.append((seqs, init_score, latest_token, init_state))

    def add_token_to_seq(self, i, token, new_score, last_hidden):
        seq, score, _, _ = self.current_seqs[i]
        seq = torch.cat([seq, token])
        self.new_seqs.append((seq, score + new_score, token, last_hidden))

    def update_current_seqs(self):
        self.current_seqs = self.new_seqs
        self.current_seqs = [item for item in self.current_seqs if item is not None]
        if len(self.current_seqs) > self.beam_size:
            self.current_seqs = sorted(self.current_seqs, key=lambda x: x[1], reverse=True)[:self.beam_size]
        self.new_seqs = []

    def check_and_add_to_terminal_seqs(self, j, eos_val):
        tmp = self.current_seqs[j]
        seqs = tmp[0]
        if seqs[-1].data[0] == eos_val:
            self.terminal_seqs.append(self.current_seqs[j])
            self.current_seqs[j] = None
            return True
        else:
            return False

    def return_final_seqs(self):
        if len(self.terminal_seqs) == 0:
            return max(self.current_seqs, key=lambda x: x[1])
        return max(self.terminal_seqs, key=lambda x: x[1])


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 n_layers=1, dropout=0.5, use_cuda=True):
        super(EncoderDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.encoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.decoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.fc = nn.Linear(hidden_size, decoder_vocab_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        if self.rnn_type != "LSTM":
            return hidden
        else:
            return (hidden, cell)

    def forward(self, encoder_inputs, encoder_init_hidden, decoder_inputs=None,
                start_input=None, max_len=None, penalized_tokens=None, beam_size=1):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)

        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(decoder_embeds, last_hidden)
        decoder_outputs = self.drop(decoder_outputs)
        decoder_outputs = self.fc(decoder_outputs)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs, last_hidden

    def generate(self, encoder_inputs, encoder_init_hidden,
               decoder_start_input, max_len, beam_size=1, penalized_tokens=None, eos_val=None):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        if beam_size == 1:
            decoder_outputs = []
            decoder_input = decoder_start_input
            for i in range(max_len):
                decoder_embed = self.decoder_embedding(decoder_input)
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                decoder_output = self.drop(decoder_output)
                decoder_output = self.fc(decoder_output)
                if penalized_tokens is not None:
                    penalize(decoder_output, penalized_tokens)
                _, decoder_input = torch.max(decoder_output, 2)
                decoder_outputs.append(decoder_output)
            return torch.max(F.log_softmax(torch.cat(decoder_outputs, 1), 2), 2)[1], last_hidden
        else:
            return self.beam_decode(last_hidden, decoder_start_input, max_len, beam_size, eos_val)

    def beam_decode(self, last_hidden, decoder_start_input, max_len, beam_size, eos_val):
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
                decoder_output = self.drop(decoder_output)
                _output = F.log_softmax(self.fc(decoder_output).squeeze(0), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size)
                for k in range(beam_size):
                    score, token = scores.data[k], tokens[k]
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0), final_seqs[3]


class AttentionEncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 score_method="dot", n_layers=1, dropout=0.5, use_cuda=True):
        super(AttentionEncoderDecoder, self).__init__()
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.encoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.attn = Attn(score_method, hidden_size, use_cuda)
        self.decoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, decoder_vocab_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        if self.rnn_type != "LSTM":
            return hidden
        else:
            return (hidden, cell)

    def forward(self, encoder_inputs, encoder_init_hidden, decoder_inputs=None,
                start_input=None, max_len=None, penalized_tokens=None):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        _outputs = []
        max_len = decoder_inputs.size(1)
        for i in range(max_len):
            _input = decoder_inputs[:, i].unsqueeze(1)
            decoder_embed = self.decoder_embedding(_input)
            decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
            _last_hidden = get_hidden(self.rnn_type, last_hidden)
            attn_weights = self.attn(_last_hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs).squeeze(1)
            outlayer_input = torch.cat([decoder_output.squeeze(1), context], 1)
            outlayer_output = F.tanh((self.concat(outlayer_input)))
            out = self.fc(outlayer_output)
            if penalized_tokens is not None:
                penalize(out, penalized_tokens)
            _outputs.append(out.unsqueeze(1))
        decoder_outputs = torch.cat(_outputs, 1)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs, last_hidden

    def generate(self, encoder_inputs, encoder_init_hidden,
               decoder_start_input, max_len, beam_size=1, penalized_tokens=None, eos_val=None):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)

        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        if beam_size == 1:
            _input = decoder_start_input
            decoder_outputs = []
            for i in range(max_len):
                decoder_embed = self.decoder_embedding(_input)
                decoder_output, last_hidden = self.decoder(decoder_embed, last_hidden)
                last_hidden_ = get_hidden(self.rnn_type, last_hidden)
                attn_weights = self.attn(last_hidden_[-1], encoder_outputs)
                decoder_output = decoder_output.squeeze(1)
                context = attn_weights.bmm(encoder_outputs).squeeze(1)
                outlayer_input = torch.cat([decoder_output, context], 1)
                outlayer_output = F.tanh(self.concat(outlayer_input))
                out = self.fc(outlayer_output)
                if penalized_tokens is not None:
                    penalize(out, penalized_tokens)
                _, _input = torch.max(out, 1)
                _input = _input.unsqueeze(1)
                decoder_outputs.append(out.unsqueeze(1))
            return torch.max(F.log_softmax(torch.cat(decoder_outputs, 1), 2), 2)[1], last_hidden
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
                last_hidden_ = get_hidden(self.rnn_type, last_hidden)
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


class BahdanauAttnEncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 score_method="concat", n_layers=1, dropout=0.5, use_cuda=True):
        super(BahdanauAttnEncoderDecoder, self).__init__()
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embed_size)
        self.encoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.attn = Attn(score_method, hidden_size, use_cuda)
        self.decoder = get_rnn(rnn_type, embed_size + hidden_size, hidden_size, n_layers, dropout)
        self.fc = nn.Linear(hidden_size, decoder_vocab_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        if self.rnn_type != "LSTM":
            return hidden
        else:
            return (hidden, cell)

    def forward(self, encoder_inputs, encoder_init_hidden, decoder_inputs=None,
                start_input=None, max_len=None, penalized_tokens=None):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        self.attn.init_param()
        _outputs = []
        max_len = decoder_inputs.size(1)
        for i in range(max_len):
            _input = decoder_inputs[:, i].unsqueeze(1)
            decoder_embed = self.decoder_embedding(_input)
            _last_hidden = get_hidden(self.rnn_type, last_hidden)
            attn_weights = self.attn(_last_hidden[-1], encoder_outputs)
            context = attn_weights.bmm(encoder_outputs)
            decoder_output, last_hidden = self.decoder(
                torch.cat([decoder_embed, context], dim=2), last_hidden)
            out = self.fc(decoder_output)
            if penalized_tokens is not None:
                penalize(out, penalized_tokens)
            _outputs.append(out)
        decoder_outputs = torch.cat(_outputs, 1)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs, last_hidden

    def generate(self, encoder_inputs, encoder_init_hidden,
               decoder_start_input, max_len, beam_size=1, penalized_tokens=None, eos_val=None):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        if beam_size == 1:
            _input = decoder_start_input
            decoder_outputs = []
            for i in range(max_len):
                decoder_embed = self.decoder_embedding(_input)
                last_hidden_ = get_hidden(self.rnn_type, last_hidden)
                attn_weights = self.attn(last_hidden_[-1], encoder_outputs)
                context = attn_weights.bmm(encoder_outputs)
                decoder_output, last_hidden = self.decoder(
                    torch.cat([decoder_embed, context], dim=2), last_hidden)
                out = self.fc(decoder_output)
                if penalized_tokens is not None:
                    penalize(out, penalized_tokens)
                _, _input = torch.max(out, 2)
                decoder_outputs.append(out)
            return torch.max(F.log_softmax(torch.cat(decoder_outputs, 1), 2), 2)[1], last_hidden
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
                last_hidden_ = get_hidden(self.rnn_type, last_hidden)
                attn_weights = self.attn(last_hidden_[-1], encoder_outputs)
                context = attn_weights.bmm(encoder_outputs)
                decoder_output, last_hidden = self.decoder(
                    torch.cat([decoder_embed, context], dim=2), last_hidden)
                _output = F.log_softmax(self.fc(decoder_output.squeeze(1)), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size)
                for k in range(beam_size):
                    score, token = scores.data[k], tokens[k]
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0), final_seqs[3]



if __name__ == "__main__":
    beamseqs = BeamSeqs(beam_size=2)
    s = Variable(torch.LongTensor([0]))
    beamseqs.init_seqs(s)
    a = Variable(torch.LongTensor([1]))
    b = Variable(torch.LongTensor([2]))
    c = Variable(torch.LongTensor([3]))
    d = Variable(torch.LongTensor([4]))
    beamseqs.add_token_to_seq(0, a, 0.2)
    beamseqs.add_token_to_seq(0, b, 0.3)
    beamseqs.update_current_seqs()
    beamseqs.add_token_to_seq(0, c, 0.25)
    beamseqs.add_token_to_seq(0, d, 0.21)
    beamseqs.add_token_to_seq(1, d, 0.27)
    beamseqs.add_token_to_seq(1, d, 0.29)
    beamseqs.update_current_seqs()
    pass

