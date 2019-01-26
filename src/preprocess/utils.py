import numpy as np
from src.basic.constant import SOS, EOS


def inds_to_tokens_1d(inds, ind_to_word, eliminate_tokens, end_tokens):
    tokens = []
    for i, idx in enumerate(inds):
        token = ind_to_word(idx)
        if token == end_tokens:
            break
        else:
            if token not in eliminate_tokens:
                tokens.append(token)
            if i == len(inds) - 1:
                tokens.append(end_tokens)

    return [tokens]


def inds_to_tokens_1d_2voc(inds, vocab, ext_vocab, eliminate_tokens, end_tokens):
    tokens = []
    for i, idx in enumerate(inds):
        if idx < vocab.size:
            token = vocab.to_word(idx)
        else:
            token = ext_vocab.to_word(idx - vocab.size)
        if token == end_tokens:
            break
        else:
            if token not in eliminate_tokens:
                tokens.append(token)
            if i == len(inds) - 1:
                tokens.append(end_tokens)

    return tokens


def inds_to_tokens_2d(inds_list, ind_to_word, eliminate_tokens, end_tokens):
    tokens_list = []
    for inds in inds_list:
        tmp = []
        for j, idx in enumerate(inds):
            token = ind_to_word(idx)
            if token == end_tokens:
                tmp.append(token)
                break
            else:
                if token not in eliminate_tokens:
                    tmp.append(token)
                if j == len(inds) - 1:
                    tmp.append(end_tokens)

        tokens_list.append(tmp)
    return tokens_list


def tokens_to_inds_2d(tokens_list, word_to_ind, pad_tokens):
    max_len = max([len(item) for item in tokens_list])
    inds_list = []
    for tokens in tokens_list:
        inds_list.append(padding_list(
            [word_to_ind(token) for token in tokens], max_len,
            padding_val=word_to_ind(pad_tokens)))
    return inds_list


def token_list_to_idx_list(token_list, vocab):
    return [vocab.to_idx(token) for token in token_list]


def padding_list(x, max_item_num, padding_val):
    if len(x) < max_item_num:
        return x + [padding_val for _ in range(max_item_num - len(x))]
    return x


def idx_to_one_hot(item, item_to_idx):
    res = [0 for _ in range(len(item_to_idx))]
    res[item_to_idx[item]] = 1
    return res