from collections import defaultdict
import pickle
from src.preprocess.preprocess import CNNDataset
from src.preprocess.vocab import Vocabulary


if __name__ == "__main__":
    input_vocab = Vocabulary()
    output_vocab = Vocabulary()
    input_vocab_size = 30000
    output_vocab_size = 10000
    word_freq = defaultdict(int)
    dataset = CNNDataset("/home/panhaojie/AbsDialogueSum/data/cnn-data/chunked")
    for example in dataset.train:
        for word in example.article + example.abstract:
            word_freq[word] += 1
    word_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for (word, freq) in word_sorted[:input_vocab_size]:
        input_vocab.add_word(word)
    for (word, freq) in word_sorted[:output_vocab_size]:
        output_vocab.add_word(word)
    with open("/home/panhaojie/AbsDialogueSum/data/cnn-vocab.pkl", "wb") as f:
        pickle.dump({"input_vocab": input_vocab,
                     "output_vocab": output_vocab,
                    "word_freq": word_freq}, f)