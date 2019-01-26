import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from src.preprocess.vocab import Vocabulary

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf-8")
    model = {}
    for i, line in enumerate(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        if i % 100 == 0:
            sys.stdout.write("\r%d" % i)
            sys.stdout.flush()
    print("Done.",len(model)," words loaded!")
    return model

if __name__ == "__main__":
    embed_dim = 100
    data_root = "/home/panhaojie/AbsDialogueSum/data"
    with open(os.path.join(data_root, "ami-vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)["input_vocab"]
    w2v_model = loadGloveModel(os.path.join(data_root, "glove.6B/glove.6B.{}d.txt".format(embed_dim)))
    embedding = nn.Embedding(vocab.size, embed_dim)
    for idx, word in vocab.idx2word.items():
        if word in w2v_model:
            embedding.weight.data[idx].copy_(torch.from_numpy(w2v_model[word].astype(np.float32)))
        if idx % 10 == 0:
            sys.stdout.write("\r%d/%d" % (idx, vocab.size))
            sys.stdout.flush()
    torch.save({"embed": embedding}, os.path.join(data_root, "glove.6B.{}d.pt".format(embed_dim)))