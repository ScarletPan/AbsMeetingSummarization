import argparse
import os
import pickle
import numpy as np
from src.preprocess.preprocess import AMIDataset
from src.preprocess.vocab import Vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-max-len', type=int, default='30000', help='Max steps of encoder')
    parser.add_argument('--decoder-max-len', type=int, default='1000', help='Max steps of decoder')
    args = parser.parse_args()
    data_root = os.path.join("/home/panhaojie/AbsDialogueSum/", "data")
    with open(os.path.join(data_root, "ami-vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)["input_vocab"]
    print("Vocabulary Size: ", vocab.size)
    with open(os.path.join(data_root, "ami-dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)
    train_generator = dataset.generate(batch_size=2, input_vocab=vocab, output_vocab=vocab, args=args, _type="all")
    num_per_epoch = next(train_generator)
    encoder_lens = []
    decoder_lens = []
    for i in range(num_per_epoch):
        batch_seq = next(train_generator)
        encoder_lens.extend(batch_seq["encoder_lens"])
        decoder_lens.extend(batch_seq["decoder_lens"])
    print("Train set Num", len(dataset.train))
    print("Valid set Num", len(dataset.valid))
    print("Test set Num", len(dataset.test))
    print("Decoder lengh, MAX={}, MEAN={}, MIN={}".format(
        np.max(decoder_lens), np.mean(decoder_lens), np.min(decoder_lens)))
    print("Encoder lengh, MAX={}, MEAN={}, MIN={}".format(
        np.max(encoder_lens), np.mean(encoder_lens), np.min(encoder_lens)))