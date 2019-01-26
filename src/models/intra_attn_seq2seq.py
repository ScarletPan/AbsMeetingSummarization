import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.attention import IntraEncoderAttn, Attn


class IntraEncoderDecoder(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_size, hidden_size, rnn_type="LSTM",
                 n_layers=1, dropout=0.5, use_cuda=True):
        super(IntraEncoderDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(encoder_vocab_size, embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers,
                               dropout=dropout, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(input_size=embed_size + hidden_size * 4, hidden_size=hidden_size * 2,
                               num_layers=n_layers, dropout=dropout, batch_first=True)
        self.intra_encoder_attn = IntraEncoderAttn(hidden_size=2 * hidden_size, use_cuda=use_cuda)
        self.intra_decoder_attn = Attn(method="general", hidden_size=2 * hidden_size, use_cuda=use_cuda)
        self.fc_vocab = nn.Linear(hidden_size * 6, decoder_vocab_size)
        self.fc_gen = nn.Linear(hidden_size * 6, 1)
        self.sigmoid = nn.Sigmoid()

        self.vocab_size = decoder_vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * 2 , batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def forward(self, encoder_inputs, encoder_init_hidden, decoder_inputs,
                ext_oov_sizes, enc_idxes_on_ext_voc):
        batch_size = encoder_inputs.size(0)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        self.intra_encoder_attn.reset_attn()
        encoder_embeds = self.embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        last_hidden = (last_hidden[0].transpose(0, 1).contiguous().view(self.n_layers, batch_size, self.hidden_size * 2),
                    last_hidden[1].transpose(0, 1).contiguous().view(self.n_layers, batch_size, self.hidden_size * 2))
        # output_dists = [[] for _ in range(batch_size)]
        output_dists = []
        max_len = decoder_inputs.size(1)
        decoder_outputs = None
        for i in range(max_len):
            _input = decoder_inputs[:, i].unsqueeze(1)
            decoder_embed = self.embedding(_input)
            _last_hidden = last_hidden[0].squeeze(0)
            intra_enc_attn_weights = self.intra_encoder_attn(_last_hidden, encoder_outputs)
            context_et = intra_enc_attn_weights.bmm(encoder_outputs)
            if i == 0:
                context_dt = Variable(torch.zeros(context_et.size()))
                if self.use_cuda:
                    context_dt = context_dt.cuda()
            else:
                intra_dec_attn_weights = self.intra_decoder_attn(_last_hidden, decoder_outputs)
                context_dt = intra_dec_attn_weights.bmm(decoder_outputs)
            decoder_output, last_hidden = self.decoder(
                torch.cat([decoder_embed, context_et, context_dt], dim=2), last_hidden)
            if decoder_outputs is None:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat([decoder_outputs, decoder_output], dim=1)
            concat_output = torch.cat([decoder_output, context_et, context_dt], dim=2)
            out = self.fc_vocab(concat_output)
            output_dists.append(F.log_softmax(out, dim=2))
        output_dists = torch.cat(output_dists, dim=1)
        return output_dists, last_hidden

    def generate(self, encoder_inputs, encoder_init_hidden, decoder_start_input, max_len,
                 ext_oov_sizes, enc_idxes_on_ext_voc, beam_size=1):
        batch_size = encoder_inputs.size(0)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        self.intra_encoder_attn.reset_attn()
        encoder_embeds = self.embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(encoder_embeds, encoder_init_hidden)
        last_hidden = (
        last_hidden[0].transpose(0, 1).contiguous().view(self.n_layers, batch_size, self.hidden_size * 2),
        last_hidden[1].transpose(0, 1).contiguous().view(self.n_layers, batch_size, self.hidden_size * 2))
        output_tokens = []
        decoder_outputs = None
        _input = decoder_start_input
        for i in range(max_len):
            decoder_embed = self.embedding(_input)
            _last_hidden = last_hidden[0].squeeze(0)
            intra_enc_attn_weights = self.intra_encoder_attn(_last_hidden, encoder_outputs)
            context_et = intra_enc_attn_weights.bmm(encoder_outputs)
            if i == 0:
                context_dt = Variable(torch.zeros(context_et.size()))
                if self.use_cuda:
                    context_dt = context_dt.cuda()
            else:
                intra_dec_attn_weights = self.intra_decoder_attn(_last_hidden, decoder_outputs)
                context_dt = intra_dec_attn_weights.bmm(decoder_outputs)
            decoder_output, last_hidden = self.decoder(
                torch.cat([decoder_embed, context_et, context_dt], dim=2), last_hidden)
            if decoder_outputs is None:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat([decoder_outputs, decoder_output], dim=1)
            concat_output = torch.cat([decoder_output, context_et, context_dt], dim=2)
            out = self.fc_vocab(concat_output)
            output_tokens.append(torch.max(out, dim=2)[1])
        output_tokens = torch.cat(output_tokens, 1)
        return output_tokens, last_hidden


if __name__ == "__main__":
    model = IntraEncoderDecoder(encoder_vocab_size=20, decoder_vocab_size=8, embed_size=17, hidden_size=25,
                                       n_layers=1).cuda()
    print(model)

    encoder_init_hidden = model.init_hidden(batch_size=2)
    # 0: <unk>
    # real encoder inputs: [[1, 8, 3, 4, 9, 6], [8, 3, 4, 8, 6, 7]]
    encoder_inputs = Variable(torch.LongTensor([[1, 0, 3, 4, 0, 6], [0, 0, 4, 0, 6, 0]])).cuda()
    decoder_inputs = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])).cuda()
    decoder_start_inputs = Variable(torch.LongTensor([[1], [2]])).cuda()
    ext_oov_sizes = [0, 3]
    enc_idxes_on_ext_voc = Variable(torch.LongTensor([[1, 6, 3, 4, 7, 6],
                                                      [8, 9, 4, 10, 6, 9]])).cuda()
    encdec_output, _ = model(
        encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden, decoder_inputs=decoder_inputs,
        ext_oov_sizes=ext_oov_sizes, enc_idxes_on_ext_voc=enc_idxes_on_ext_voc)
    tokens, _ = model.generate(encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
                               decoder_start_input=decoder_start_inputs, max_len=5, ext_oov_sizes=ext_oov_sizes,
                               enc_idxes_on_ext_voc=enc_idxes_on_ext_voc)
    print()

#    vocab_dist = F.softmax(out, dim=2).squeeze(1)  # batch_size x ext_vocab_size
#    attn_dist = intra_enc_attn_weights.squeeze(1)
# p_gens = self.sigmoid(
#                self.fc_gen(torch.cat([decoder_output, context_et, context_dt], dim=2))).squeeze(2)
#     for b in range(batch_size):
#         if ext_oov_sizes[b] == 0:
#             vocab_dist_extended = vocab_dist[b]
#         else:
#             extra_zeros = Variable(torch.zeros(ext_oov_sizes[b]))
#             if self.use_cuda: extra_zeros = extra_zeros.cuda()
#             vocab_dist_extended = torch.cat([vocab_dist[b], extra_zeros])  # ext_vocab_size
#
#         attn_dist_project = Variable(torch.zeros(self.vocab_size + ext_oov_sizes[b]))
#         if self.use_cuda: attn_dist_project = attn_dist_project.cuda()
#         attn_dist_project.scatter_add_(0, enc_idxes_on_ext_voc[b], attn_dist[b])  # ext_vocab_size
#         final_dist = p_gens[b] * vocab_dist_extended + (1 - p_gens[b]) * attn_dist_project
#         output_dists[b].append(final_dist.unsqueeze(0))
#
# for b in range(batch_size):
#     output_dists[b] = torch.log(torch.cat(output_dists[b]))



# p_gens = self.sigmoid(
#                 self.fc_gen(torch.cat([decoder_output, context_et, context_dt], dim=2))).squeeze(2)
#             out = self.fc_vocab(torch.cat([decoder_output, context_et, context_dt], dim=2))
#             vocab_dist = F.softmax(out, dim=2).squeeze(1)  # batch_size x ext_vocab_size
#             attn_dist = intra_enc_attn_weights.squeeze(1)
#             output_token_batch = []
#             for b in range(batch_size):
#                 if ext_oov_sizes[b] == 0:
#                     vocab_dist_extended = vocab_dist[b]
#                 else:
#                     extra_zeros = Variable(torch.zeros(ext_oov_sizes[b]))
#                     if self.use_cuda: extra_zeros = extra_zeros.cuda()
#                     vocab_dist_extended = torch.cat([vocab_dist[b], extra_zeros])  # ext_vocab_size
#
#                 attn_dist_project = Variable(torch.zeros(self.vocab_size + ext_oov_sizes[b]))
#                 if self.use_cuda: attn_dist_project = attn_dist_project.cuda()
#                 attn_dist_project.scatter_add_(0, enc_idxes_on_ext_voc[b], attn_dist[b])  # ext_vocab_size
#                 final_dist = p_gens[b] * vocab_dist_extended + (1 - p_gens[b]) * attn_dist_project
#                 _, output_token = torch.max(final_dist, dim=0)
#                 output_token_batch.append(output_token)
#             output_token_batch = torch.cat(output_token_batch).unsqueeze(1)
#             _input = output_token_batch
#             output_tokens.append(output_token_batch)