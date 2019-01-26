from src.basic.constant import UNK, PAD, SOS, EOS


class Vocabulary(object):
    def __init__(self, no_extra_token=False):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0
        if not no_extra_token:
            self.add_word(UNK)
            self.add_word(PAD)
            self.add_word(SOS)
            self.add_word(EOS)

    def has(self, word):
        return word in self.word2idx

    def add_word(self, word):
        if not self.has(word):
            ind = self.size
            self.word2idx[word] = ind
            self.idx2word[ind] = word
            self.size += 1

    def to_idx(self, word):
        if self.has(word):
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']

    def to_word(self, ind):
        if ind >= self.size:
            return 0
        return self.idx2word[ind]

    def append_vocab(self, vocab):
        for word in vocab.word2idx:
            self.add_word(word)


if __name__ == "__main__":
    import argparse
    import os
    import pickle
    from src.preprocess.preprocess import AMIDataset
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/home/panhaojie/AbsDialogueSum/data", help='data directory')
    args = parser.parse_args()

    word_freq = defaultdict(int)
    vocab = Vocabulary()
    dataset = AMIDataset(args.data_path)
    train = dataset.train
    for meeting in train:
        doc_seq = meeting.article
        sum_seq = meeting.abstract
        for word in doc_seq:
            word_freq[word] += 1
            vocab.add_word(word)
    print()
    # with open(os.path.join(args.data_path, "vocab.pkl"), "wb") as f:
    #     pickle.dump({"vocab": vocab,
    #                  "word_freq": word_freq}, f)