import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.utils import get_rnn, get_hidden
from src.models.attention import Attn


class BasicHierarchicalEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 n_layers=1, dropout=0.5, use_cuda=True):
        super(BasicHierarchicalEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.word_encoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.word_decoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.sent_encoder = get_rnn(rnn_type, hidden_size, hidden_size, n_layers, dropout)
        self.sent_decoder = get_rnn(rnn_type, hidden_size, hidden_size, n_layers, dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.n_layers = n_layers

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

    def forward(self, word_encoder_inputs_list, word_decoder_inputs_list):
        """
        :param word_encoder_inputs_list: sent_num size of [b_size * enc_sent_len1, b_size * enc_sent_len2, ...]
               word_decoder_inputs_list: sent_num size of [b_size * dec_sent_len1, b_size * dec_sent_len2, ...]
        :return: sent_decoder_outputs_list: sent_num size of [b_size * dec_sent_len1 * vocab_size, ...]
        """
        sent_num = len(word_encoder_inputs_list)
        b_size = word_encoder_inputs_list[0].size(0)
        sent_encoder_inputs = []
        for i in range(sent_num):
            word_encoder_inputs = word_encoder_inputs_list[i] # b_size * sent_len
            word_encoder_init_hidden = self.init_hidden(batch_size=b_size)
            word_encoder_embeds = self.embedding(word_encoder_inputs)
            word_encoder_outputs, word_last_hidden = self.word_encoder(word_encoder_embeds, word_encoder_init_hidden)
            sent_encoder_inputs.append(get_hidden(self.rnn_type, word_last_hidden))
        sent_encoder_inputs = torch.cat(sent_encoder_inputs).transpose(0, 1)
        sent_encoder_init_hidden = self.init_hidden(batch_size=b_size)
        sent_encoder_outputs, sent_last_hidden = self.sent_encoder(sent_encoder_inputs, sent_encoder_init_hidden)

        sent_num = len(word_decoder_inputs_list)
        sent_decoder_outputs_list = []
        sent_decoder_input = get_hidden(self.rnn_type, sent_last_hidden).transpose(0, 1)
        for i in range(1, sent_num):
            sent_decoder_outputs, sent_last_hidden = self.sent_decoder(sent_decoder_input, sent_last_hidden)
            word_decoder_inputs = word_decoder_inputs_list[i] # b_size * sent_len
            word_decoder_embeds = self.embedding(word_decoder_inputs)
            word_decoder_outputs, word_last_hidden = self.word_decoder(word_decoder_embeds, sent_last_hidden)
            sent_decoder_input = get_hidden(self.rnn_type, word_last_hidden).transpose(0, 1)
            outputs = F.log_softmax(self.fc(word_decoder_outputs), dim=2)
            sent_decoder_outputs_list.append(outputs)
        return sent_decoder_outputs_list

    def generate(self, word_encoder_inputs_list, word_decoder_start_input,
                 max_sent_num, max_sent_len):
        """
        :param word_encoder_inputs_list: sent_num size of [b_size * enc_sent_len1, b_size * enc_sent_len2, ...]
               word_decoder_inputs_list: sent_num size of [b_size * dec_sent_len1, b_size * dec_sent_len2, ...]
        :return: sent_decoder_outputs_list: sent_num size of [b_size * dec_sent_len1 * vocab_size, ...]
        """
        self.word_encoder.flatten_parameters()
        self.word_decoder.flatten_parameters()
        self.sent_encoder.flatten_parameters()
        self.sent_decoder.flatten_parameters()
        sent_num = len(word_encoder_inputs_list)
        b_size = word_encoder_inputs_list[0].size(0)
        sent_encoder_inputs = []
        for i in range(sent_num):
            word_encoder_inputs = word_encoder_inputs_list[i]  # b_size * sent_len
            word_encoder_init_hidden = self.init_hidden(batch_size=b_size)
            word_encoder_embeds = self.embedding(word_encoder_inputs)
            word_encoder_outputs, word_last_hidden = self.word_encoder(word_encoder_embeds, word_encoder_init_hidden)
            sent_encoder_inputs.append(get_hidden(self.rnn_type, word_last_hidden))
        sent_encoder_inputs = torch.cat(sent_encoder_inputs).transpose(0, 1)
        sent_encoder_init_hidden = self.init_hidden(batch_size=b_size)
        sent_encoder_outputs, sent_last_hidden = self.sent_encoder(sent_encoder_inputs, sent_encoder_init_hidden)

        sent_decoder_input = get_hidden(self.rnn_type, sent_last_hidden).transpose(0, 1)
        sent_output_list = []
        for i in range(max_sent_num):
            sent_decoder_outputs, sent_last_hidden = self.sent_decoder(sent_decoder_input, sent_last_hidden)
            _word_input = word_decoder_start_input
            word_last_hidden = sent_last_hidden
            word_output_list = []
            for j in range(max_sent_len):
                word_decoder_embeds = self.embedding(_word_input)
                word_decoder_outputs, word_last_hidden = self.word_decoder(word_decoder_embeds, word_last_hidden)
                word_outputs = self.fc(word_decoder_outputs)
                _, word_out = word_outputs.max(2)
                word_output_list.append(word_out.unsqueeze(1))
            word_output_list = torch.cat(word_output_list, dim=1)
            sent_output_list.append(word_output_list)
            sent_decoder_input = get_hidden(self.rnn_type, word_last_hidden).transpose(0, 1)
        return sent_output_list


class AttentionHierarchicalEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 n_layers=1, dropout=0.5, use_cuda=True):
        super(AttentionHierarchicalEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.word_encoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.word_decoder = get_rnn(rnn_type, embed_size, hidden_size, n_layers, dropout)
        self.sent_encoder = get_rnn(rnn_type, hidden_size, hidden_size, n_layers, dropout)
        self.sent_decoder = get_rnn(rnn_type, hidden_size * 2, hidden_size, n_layers, dropout)
        self.attn = Attn(method="dot", hidden_size=hidden_size, use_cuda=use_cuda)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.n_layers = n_layers

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

    def forward(self, word_encoder_inputs_list, word_decoder_inputs_list):
        """
        :param word_encoder_inputs_list: sent_num size of [b_size * enc_sent_len1, b_size * enc_sent_len2, ...]
               word_decoder_inputs_list: sent_num size of [b_size * dec_sent_len1, b_size * dec_sent_len2, ...]
        :return: sent_decoder_outputs_list: sent_num size of [b_size * dec_sent_len1 * vocab_size, ...]
        """
        sent_num = len(word_encoder_inputs_list)
        b_size = word_encoder_inputs_list[0].size(0)
        sent_encoder_inputs = []
        for i in range(sent_num):
            word_encoder_inputs = word_encoder_inputs_list[i] # b_size * sent_len
            word_encoder_init_hidden = self.init_hidden(batch_size=b_size)
            word_encoder_embeds = self.embedding(word_encoder_inputs)
            word_encoder_outputs, word_last_hidden = self.word_encoder(word_encoder_embeds, word_encoder_init_hidden)
            sent_encoder_inputs.append(get_hidden(self.rnn_type, word_last_hidden))
        sent_encoder_inputs = torch.cat(sent_encoder_inputs).transpose(0, 1)
        sent_encoder_init_hidden = self.init_hidden(batch_size=b_size)
        sent_encoder_outputs, sent_last_hidden = self.sent_encoder(sent_encoder_inputs, sent_encoder_init_hidden)

        sent_num = len(word_decoder_inputs_list)
        sent_decoder_outputs_list = []
        sent_decoder_input = get_hidden(self.rnn_type, sent_last_hidden).transpose(0, 1)
        for i in range(1, sent_num):
            _sent_last_hidden = get_hidden(self.rnn_type, sent_last_hidden)
            attn_weights = self.attn(_sent_last_hidden[-1], sent_encoder_outputs)
            context = attn_weights.bmm(sent_encoder_outputs)
            sent_decoder_outputs, sent_last_hidden = self.sent_decoder(
                torch.cat([sent_decoder_input, context], dim=2), sent_last_hidden)
            word_decoder_inputs = word_decoder_inputs_list[i] # b_size * sent_len
            word_decoder_embeds = self.embedding(word_decoder_inputs)
            word_decoder_outputs, word_last_hidden = self.word_decoder(word_decoder_embeds, sent_last_hidden)
            sent_decoder_input = get_hidden(self.rnn_type, word_last_hidden).transpose(0, 1)
            outputs = F.log_softmax(self.fc(word_decoder_outputs), dim=2)
            import os, pickle
            from src.preprocess.utils import inds_to_tokens_2d
            with open(os.path.join("/home/panhaojie/AbsDialogueSum/data", "ami-vocab-hierachy.pkl"), "rb") as f:
                vocab = pickle.load(f)
            _, t = torch.max(outputs, 2)
            pred = t.data.tolist()
            pred_tokens = inds_to_tokens_2d(
                pred, vocab["output_vocab"].to_word, eliminate_tokens=["<pad>", "<sos>"], end_tokens="<eos>")
            decoder_tokens = inds_to_tokens_2d(word_decoder_inputs.data.tolist(), vocab["output_vocab"].to_word,
                              eliminate_tokens=["<pad>", "<sos>"], end_tokens="<eos>")
            sent_decoder_outputs_list.append(outputs)
        return sent_decoder_outputs_list

    def generate(self, word_encoder_inputs_list, word_decoder_start_input,
                 max_sent_num, max_sent_len):
        """
        :param word_encoder_inputs_list: sent_num size of [b_size * enc_sent_len1, b_size * enc_sent_len2, ...]
               word_decoder_start_input: b_size * 1
        :return: sent_decoder_outputs_list: sent_num size of [b_size * dec_sent_len1 * vocab_size, ...]
        """
        self.word_encoder.flatten_parameters()
        self.word_decoder.flatten_parameters()
        self.sent_encoder.flatten_parameters()
        self.sent_decoder.flatten_parameters()
        sent_num = len(word_encoder_inputs_list)
        b_size = word_encoder_inputs_list[0].size(0)
        sent_encoder_inputs = []
        for i in range(sent_num):
            word_encoder_inputs = word_encoder_inputs_list[i]  # b_size * sent_len
            word_encoder_init_hidden = self.init_hidden(batch_size=b_size)
            word_encoder_embeds = self.embedding(word_encoder_inputs)
            word_encoder_outputs, word_last_hidden = self.word_encoder(word_encoder_embeds, word_encoder_init_hidden)
            sent_encoder_inputs.append(get_hidden(self.rnn_type, word_last_hidden))
        sent_encoder_inputs = torch.cat(sent_encoder_inputs).transpose(0, 1)
        sent_encoder_init_hidden = self.init_hidden(batch_size=b_size)
        sent_encoder_outputs, sent_last_hidden = self.sent_encoder(sent_encoder_inputs, sent_encoder_init_hidden)

        sent_decoder_input = get_hidden(self.rnn_type, sent_last_hidden).transpose(0, 1)
        sent_output_list = []
        for i in range(max_sent_num):
            _sent_last_hidden = get_hidden(self.rnn_type, sent_last_hidden)
            attn_weights = self.attn(_sent_last_hidden[-1], sent_encoder_outputs)
            context = attn_weights.bmm(sent_encoder_outputs)
            sent_decoder_outputs, sent_last_hidden = self.sent_decoder(
                torch.cat([sent_decoder_input, context], dim=2), sent_last_hidden)
            _word_input = word_decoder_start_input
            word_last_hidden = sent_last_hidden
            word_output_list = []
            for j in range(max_sent_len):
                word_decoder_embeds = self.embedding(_word_input)
                word_decoder_outputs, word_last_hidden = self.word_decoder(word_decoder_embeds, word_last_hidden)
                word_outputs = self.fc(word_decoder_outputs)
                _, word_out = word_outputs.max(2)
                word_output_list.append(word_out.unsqueeze(1))
            word_output_list = torch.cat(word_output_list, dim=1)
            sent_output_list.append(word_output_list)
            sent_decoder_input = get_hidden(self.rnn_type, word_last_hidden).transpose(0, 1)
        return sent_output_list


if __name__ == "__main__":
    # 10: <sod>, 11: <eod>
    sent0 = Variable(torch.LongTensor([[10], [10]])).cuda()
    sent1 = Variable(torch.LongTensor([[1, 2, 3, 0], [2, 3, 4, 0]])).cuda()
    sent2 = Variable(torch.LongTensor([[1, 2, 3, 4, 0], [2, 3, 4, 5, 0]])).cuda()
    sent3 = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 0]])).cuda()
    sent4 = Variable(torch.LongTensor([[11], [11]])).cuda()
    word_encoder_inputs_list = [sent1, sent2, sent4]
    word_decoder_inputs_list = [sent0, sent1, sent2, sent3, sent4]
    word_decoder_targets_list = [sent1, sent2, sent3, sent4]
    word_decoder_start_input = sent0
    model = BasicHierarchicalEncoderDecoder(vocab_size=20, embed_size=7, hidden_size=25).cuda()
    # outputs = model(word_encoder_inputs_list, word_decoder_inputs_list)
    generated_outputs = model.generate(word_encoder_inputs_list, word_decoder_start_input,
                                       max_sent_num=6, max_sent_len=7)
    print()
    pass
