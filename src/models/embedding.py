import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, embed_size, extra_categorical_nums=None, extra_embed_sizes=None, use_cuda=True):
        super(Embedder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        if extra_categorical_nums:
            self.extra_embeddings = [
                nn.Embedding(extra_categorical_nums[i], extra_embed_sizes[i]) for i in range(len(extra_embed_sizes))]
            if use_cuda:
                self.extra_embeddings = [item.cuda() for item in self.extra_embeddings]
            self.size = embed_size + sum(extra_embed_sizes)
        else:
            self.size = embed_size

    def forward(self, x, extra_inputs=None):
        """

        :param x: batch_size x seq_len
        :param extra_inputs:  [batch_size x seq_len, batch_size x seq_len ... ]
        :return: batch_size x seq_len x (embed_size1 + embed_size2 + ...)
        """
        word_embeds = self.word_embedding(x)
        if extra_inputs:
            extra_embeds = [self.extra_embeddings[i](extra_inputs[i]) for i in range(len(extra_inputs))]
            final_embeds = torch.cat([word_embeds] + extra_embeds, dim=2)
            return final_embeds
        else:
            return word_embeds
