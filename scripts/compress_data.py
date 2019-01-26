import os
import pickle
from src.preprocess.vocab import Vocabulary
from src.preprocess.preprocess import AMIDataset


def chunck_data(dataset, _type):
    doc = open(os.path.join(data_root, "chunked", "%s.doc" % _type), "w")
    abs = open(os.path.join(data_root, "chunked", "%s.abs" % _type), "w")
    for meeting in dataset.__getattribute__(_type):
        doc.write(" ".join(meeting.article) + "\n")
        for sentence in meeting.summary_sents:
            abs.write("<s>" + " ".join(sentence) + "</s>")
        abs.write("\n")
    doc.close()
    abs.close()


if __name__ == "__main__":
    data_root = "/home/panhaojie/AbsDialogueSum/data"
    with open(os.path.join(data_root, "ami-vocab.pkl"), "rb") as f:
        vocab_file = pickle.load(f)
        vocab = vocab_file["input_vocab"]
        word_freq = vocab_file["word_freq"]
    with open(os.path.join(data_root, "chunked", "vocab"), "w") as f:
        for word, freq in word_freq.items():
            f.write(word + " " + str(freq) + "\n")
    if not os.path.exists(os.path.join(data_root, "ami-dataset.pkl")):
        print("Create Dataset")
        dataset = AMIDataset(data_root)
        with open(os.path.join(data_root, "ami-dataset.pkl"), "wb") as f:
            pickle.dump(dataset, f)
    else:
        print("Load Dataset")
        with open(os.path.join(data_root, "ami-dataset.pkl"), "rb") as f:
            dataset = pickle.load(f)

    if not os.path.isdir(os.path.join(data_root, "chunked")):
        os.mkdir(os.path.join(data_root, "chunked"))
    chunck_data(dataset, "train")
    chunck_data(dataset, "valid")
    chunck_data(dataset, "test")


