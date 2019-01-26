from collections import defaultdict
import os
import pickle
import sys
from src.preprocess.preprocess import AMIDataset
from src.preprocess.vocab import Vocabulary


if __name__ == "__main__":
    input_vocab = Vocabulary()
    output_vocab = Vocabulary()
    word_freq = defaultdict(int)
    data_root = sys.argv[1]
    dataset = AMIDataset(data_root)
    with open(os.path.join(data_root, "ami-dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)
    for example in dataset.train:
        for word in example.article + example.abstract:
            word_freq[word] += 1
    word_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for (word, freq) in word_sorted:
        input_vocab.add_word(word)
    # for (word, freq) in word_sorted:
    #     output_vocab.add_word(word)
    output_vocab = input_vocab.copy()
    with open(os.path.join(data_root, "/ami-vocab.pkl"), "wb") as f:
        pickle.dump({"input_vocab": input_vocab,
                     "output_vocab": output_vocab,
                    "word_freq": word_freq}, f)
    input_vocab.add_word("<sod>")
    input_vocab.add_word("<eod>")
    output_vocab.add_word("<sod>")
    output_vocab.add_word("<eod>")
    with open(os.path.join(data_root, "ami-vocab-hierachy.pkl"), "wb") as f:
        pickle.dump({"input_vocab": input_vocab,
                     "output_vocab": output_vocab,
                    "word_freq": word_freq}, f)
