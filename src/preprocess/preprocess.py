import copy
import json
import sys
import os
import random
import struct
import numpy as np
from src.basic.constant import SOS, EOS, SOD, EOD, PAD, SENTENCE_START, SENTENCE_END, NER_TO_IDX, POS_TO_IDX
from src.preprocess.utils import padding_list, idx_to_one_hot
from src.preprocess.vocab import Vocabulary
from tensorflow.core.example import example_pb2


def create_one_batch(examples, input_vocab, output_vocab, args):
    batch_oov_vocabs = [Vocabulary(no_extra_token=True) for _ in range(len(examples))]
    articles = [example.article if len(example.article) < args.encoder_max_len
                else example.article[:args.encoder_max_len] for example in examples]
    abstracts = [example.abstract if len(example.abstract) < args.decoder_max_len
                 else example.abstract[:args.decoder_max_len] for example in examples]
    encoder_tokens = []
    encoder_inputs = []
    encoder_lens = [len(article) for article in articles]
    decoder_tokens = []
    decoder_inputs = []
    decoder_targets = []
    decoder_lens = [len(abstract) for abstract in abstracts]
    enc_idxes_on_ext_voc_list = []
    max_encoder_len = max(encoder_lens)
    max_decoder_len = max(decoder_lens)
    for i in range(len(examples)):
        encoder_tokens.append(articles[i] + [EOS])
        encoder_inputs.append(padding_list(
            [input_vocab.to_idx(token) for token in encoder_tokens[i]],
            max_encoder_len + 1, input_vocab.to_idx(PAD)))
        enc_idxes_on_ext_voc = [input_vocab.to_idx(PAD) for _ in range(max_encoder_len + 1)]
        for j, token in enumerate(encoder_tokens[-1]):
            if output_vocab.has(token):
                enc_idxes_on_ext_voc[j] = output_vocab.to_idx(token)
            else:
                batch_oov_vocabs[i].add_word(token)
                enc_idxes_on_ext_voc[j] = output_vocab.size + batch_oov_vocabs[i].to_idx(token)
        enc_idxes_on_ext_voc_list.append(enc_idxes_on_ext_voc)
        decoder_tokens.append(abstracts[i] + [EOS])
        decoder_inputs.append(padding_list(
            [input_vocab.to_idx(token) for token in [SOS] + decoder_tokens[i][:-1]],
            max_decoder_len + 1, input_vocab.to_idx(PAD)))
        decoder_targets.append(padding_list(
            [output_vocab.to_idx(token) for token in decoder_tokens[i]], max_decoder_len + 1,
            output_vocab.to_idx(PAD)))

    decoder_ext_targets = [item[:] for item in decoder_targets]
    for i, token_list in enumerate(decoder_tokens):
        for j, token in enumerate(token_list):
            if not output_vocab.has(token) and batch_oov_vocabs[i].has(token):
                decoder_ext_targets[i][j] = output_vocab.size + batch_oov_vocabs[i].to_idx(token)
    batch = {
        "encoder_tokens": encoder_tokens,
        "decoder_tokens": decoder_tokens,
        "encoder_lens": encoder_lens,
        "decoder_lens": decoder_lens,
        "encoder_inputs": encoder_inputs,
        "decoder_inputs": decoder_inputs,
        "decoder_targets": decoder_targets,
        "decoder_ext_targets": decoder_ext_targets,
        "enc_idxes_on_ext_voc": enc_idxes_on_ext_voc_list,
        "batch_oov_vocabs": batch_oov_vocabs
    }
    return batch


def create_batches(batch_size, examples, input_vocab, output_vocab, early_stop, args):
    total_steps = len(examples) // batch_size
    batches = []
    for i in range(total_steps):
        st = i * batch_size
        end = (i + 1) * batch_size
        batches.append(create_one_batch(examples[st:end], input_vocab, output_vocab, args))
        if i % 10:
            sys.stdout.write("\r{}/{} batch created".format(i, total_steps))
            sys.stdout.flush()
        if early_stop is not None and i > early_stop:
            break
    print()
    return batches


class AMIMeeting(object):
    def __init__(self, data_path, meeting_id):
        self.data_path = data_path
        self.meeting_id = meeting_id
        self.agents = []
        self.sentence = []
        self.segments = []
        self.st_times = []
        self.end_times = []
        self.article_pos_tags = []
        self.article_ner_tags = []
        self.article_tf_tags = []
        self.article_idf_tags = []
        self.summary_sents = []
        self.summary_pos_tags = []
        self.summary_ner_tags = []
        self.size = 0
        self.create_meeting()
        self.article = [x.lower() for sentence in self.sentence for x in sentence]
        self.abstract = [x.lower() for sentence in self.summary_sents for x in sentence]
        self.article_pos_tag_seq = [x for tags in self.article_pos_tags for x in tags]
        self.article_ner_tag_seq = [x for tags in self.article_ner_tags for x in tags]
        self.article_tf_tag_seq = [x for tags in self.article_tf_tags for x in tags]
        self.article_idf_tag_seq = [x for tags in self.article_idf_tags for x in tags]
        self.summary_pos_tag_seq = [x for tags in self.summary_pos_tags for x in tags]
        self.summary_ner_tag_seq = [x for tags in self.summary_ner_tags for x in tags]
        # self.article = [x.lower() for sentence in self.sentence for x in sentence if x not in
        #                 ["uh", "um", "mm-hmm", "mm", "oh", "hmm"]]
        # self.abstract = [x.lower() for sentence in self.summary_sents for x in sentence if x not in
        #                 ["uh", "um", "mm-hmm", "mm", "oh", "hmm"]]

    def create_meeting(self):
        with open(os.path.join(self.data_path, "docs", self.meeting_id +".json")) as f:
            raw = json.load(f)
        last_agent = ""
        for item in raw:
            agent = item["agent"]
            self.segments.append([t.strip().lower() for t in item["seg_tokens"]])
            if agent == last_agent:
                self.sentence[-1] += item["seg_tokens"]
                self.article_pos_tags[-1] += item["postag"]
                self.article_ner_tags[-1] += item["nertag"]
                self.article_tf_tags[-1] += item["TF"]
                self.article_idf_tags[-1] += item["IDF"]
                self.end_times[-1] = item["end_time"]
            else:
                self.agents.append(item["agent"])
                self.sentence.append(item["seg_tokens"])
                self.article_pos_tags.append(item["postag"])
                self.article_ner_tags.append(item["nertag"])
                self.article_tf_tags.append(item["TF"])
                self.article_idf_tags.append(item["IDF"])
                self.st_times.append(item["start_time"])
                self.end_times.append(item["end_time"])
            last_agent = agent
        with open(os.path.join(self.data_path, "doc_abs", self.meeting_id +".json")) as f:
            raw = json.load(f)
        for item in raw:
            self.summary_sents.append(item["summary_sents"])
            self.summary_pos_tags.append(item["postag"])
            self.summary_ner_tags.append(item["nertag"])
        self.size = len(self.agents)

    def __str__(self):
        s = self.meeting_id + "\n"
        s += "=" * 10 + " SUMMARY " + "=" * 10 + "\n"
        for sentence in self.summary_sents:
            s += " ".join(sentence) + "\n"
        s += "\n"
        s += "=" * 10 + " DIALOGUE " + "=" * 10 + "\n"
        for i in range(self.size):
            s += "{:7.2f} - {:7.2f} [{}] : {}\n".format(
                self.st_times[i], self.end_times[i], self.agents[i], " ".join(self.sentence[i]))
        s += "\n"

        return s


class AMIDataset(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.train = []
        self.valid = []
        self.test = []
        self.create_dataset()
        pass

    def create_dataset(self):
        doc_filenames = os.listdir(os.path.join(self.data_root, "docs"))
        total_sample_num = len(doc_filenames)
        idxes = list(range(len(doc_filenames)))
        # random.shuffle(idxes)
        train_idxes = idxes[:total_sample_num // 7 * 5]
        valid_idxes = idxes[total_sample_num // 7 * 5:total_sample_num // 7 * 6]
        test_idxes = idxes[total_sample_num // 7 * 6:]
        self.train = self.create_meetings([doc_filenames[i] for i in train_idxes])
        self.valid = self.create_meetings([doc_filenames[i] for i in valid_idxes])
        self.test = self.create_meetings([doc_filenames[i] for i in test_idxes])

    def create_meetings(self, filenames):
        res = []
        for filename in filenames:
            res.append(AMIMeeting(self.data_root, filename.split(".")[0]))
        return res

    @staticmethod
    def create_one_sentence_batch(examples, input_vocab, output_vocab, args):
        pad, sos, eos, sod, eod = input_vocab.to_idx(PAD), input_vocab.to_idx(SOS), \
                                input_vocab.to_idx(EOS), input_vocab.to_idx(SOD), input_vocab.to_idx(EOD)
        encoder_sent_nums = [len(example.segments) for example in examples]
        encoder_max_sent_num = max(encoder_sent_nums)
        encoder_sent_lens = np.array([padding_list([len(x) + 1 for x in example.segments], encoder_max_sent_num, 0)
                             for example in examples], dtype=np.int32)
        encoder_max_sent_lens = encoder_sent_lens.max(0)
        word_encoder_inputs_list = [np.ones((len(examples), sent_len), dtype=np.int32) * pad
                                    for sent_len in encoder_max_sent_lens] + \
                                  [np.ones((len(examples), 1), dtype=np.int32) * eod]

        decoder_sent_nums = [len(example.summary_sents) for example in examples]
        decoder_max_sent_num = max(decoder_sent_nums)
        decoder_sent_lens = np.array([padding_list([len(x) + 1 for x in example.summary_sents], decoder_max_sent_num, 0)
                            for example in examples], dtype=np.int32)
        decoder_max_sent_lens = decoder_sent_lens.max(0)
        word_decoder_inputs_list =[np.ones((len(examples), 1), dtype=np.int32) * sod] + \
                                  [np.ones((len(examples), sent_len), dtype=np.int32) * pad for
                                   sent_len in decoder_max_sent_lens] + \
                                  [np.ones((len(examples), 1), dtype=np.int32) * eod]
        word_decoder_targets_list = [np.ones((len(examples), sent_len), dtype=np.int32) * pad for
                                     sent_len in decoder_max_sent_lens] + \
                                  [np.ones((len(examples), 1), dtype=np.int32) * eod]
        for b, example in enumerate(examples):
            for i, sent in enumerate(example.segments):
                word_idxes = [input_vocab.to_idx(w) for w in sent]
                word_encoder_inputs_list[i][b][:len(word_idxes)] = word_idxes
                word_encoder_inputs_list[i][b][len(word_idxes)] = eos
            for i, sent in enumerate(example.summary_sents):
                word_idxes = [input_vocab.to_idx(w) for w in sent]
                word_decoder_inputs_list[i + 1][b][0] = sos
                word_decoder_inputs_list[i + 1][b][1:len(word_idxes) + 1] = word_idxes
                word_decoder_targets_list[i][b][:len(word_idxes)] = word_idxes
                word_decoder_targets_list[i][b][len(word_idxes)] = eos
        decoder_tokens = []
        for example in examples:
            summary_sents = example.summary_sents
            sent_list = [sent + [EOS] for sent in summary_sents]
            sent_list = [x for tokens in sent_list for x in tokens]
            decoder_tokens.append(sent_list)
        return {
            "sent_encoder_tokens": [example.segments for example in examples],
            "decoder_tokens": decoder_tokens,
            "word_encoder_inputs_list": word_encoder_inputs_list,
            "word_decoder_inputs_list": word_decoder_inputs_list,
            "word_decoder_targets_list": word_decoder_targets_list,
            "encoder_sent_lens": encoder_sent_lens.T.tolist(),
            "decoder_sent_lens": decoder_sent_lens.T.tolist(),
            "encoder_sent_nums": encoder_sent_nums,
            "decoder_sent_nums": decoder_sent_nums,
        }

    def create_sentence_batches(self, batch_size, examples, input_vocab, output_vocab, early_stop, args):
        total_steps = len(examples) // batch_size
        batches = []
        for i in range(total_steps):
            st = i * batch_size
            end = (i + 1) * batch_size
            batches.append(self.create_one_sentence_batch(examples[st:end], input_vocab, output_vocab, args))
            if i % 10:
                sys.stdout.write("\r{}/{} batch created".format(i, total_steps))
                sys.stdout.flush()
            if early_stop is not None and i > early_stop:
                break
        print()
        return batches

    @staticmethod
    def create_one_batch(examples, input_vocab, output_vocab, args):
        batch_oov_vocabs = [Vocabulary(no_extra_token=True) for _ in range(len(examples))]
        articles = [example.article if len(example.article) < args.encoder_max_len
                    else example.article[:args.encoder_max_len] for example in examples]
        abstracts = [example.abstract if len(example.abstract) < args.decoder_max_len
                     else example.abstract[:args.decoder_max_len] for example in examples]
        article_pos_tags = [example.article_pos_tag_seq if len(example.article_pos_tag_seq) < args.encoder_max_len
                    else example.article_pos_tag_seq[:args.encoder_max_len] for example in examples]
        article_ner_tags = [example.article_ner_tag_seq if len(example.article_ner_tag_seq) < args.encoder_max_len
                    else example.article_ner_tag_seq[:args.encoder_max_len] for example in examples]
        article_tf_tags = [example.article_tf_tag_seq if len(example.article_tf_tag_seq) < args.encoder_max_len
                    else example.article_tf_tag_seq[:args.encoder_max_len] for example in examples]
        article_idf_tags = [example.article_idf_tag_seq if len(example.article_idf_tag_seq) < args.encoder_max_len
                           else example.article_idf_tag_seq[:args.encoder_max_len] for example in examples]
        summary_pos_tags = [example.summary_pos_tag_seq if len(example.summary_pos_tag_seq) < args.decoder_max_len
                    else example.summary_pos_tag_seq[:args.decoder_max_len] for example in examples]
        summary_ner_tags = [example.summary_ner_tag_seq if len(example.summary_ner_tag_seq) < args.decoder_max_len
                    else example.summary_ner_tag_seq[:args.decoder_max_len] for example in examples]
        encoder_tokens = []
        encoder_pos_tags = []
        encoder_ner_tags = []
        encoder_tf_tags = []
        encoder_idf_tags = []
        encoder_inputs = []
        encoder_lens = [len(article) for article in articles]
        decoder_tokens = []
        decoder_pos_tags = []
        decoder_ner_tags = []
        decoder_inputs = []
        decoder_targets = []
        decoder_lens = [len(abstract) for abstract in abstracts]
        enc_idxes_on_ext_voc_list = []
        max_encoder_len = max(encoder_lens)
        max_decoder_len = max(decoder_lens)
        for i in range(len(examples)):
            encoder_tokens.append(articles[i] + [EOS])
            encoder_pos_tags.append(padding_list(
                [POS_TO_IDX[item] for item in article_pos_tags[i]] +
                [POS_TO_IDX["."]], max_encoder_len + 1, padding_val=POS_TO_IDX["."]))
            encoder_ner_tags.append(padding_list(
                [NER_TO_IDX[item] for item in article_ner_tags[i]] +
                [NER_TO_IDX["O"]], max_encoder_len + 1, padding_val=NER_TO_IDX["O"]))
            encoder_tf_tags.append(padding_list(
                [item for item in article_tf_tags[i]] + [0], max_encoder_len + 1, padding_val=0))
            encoder_idf_tags.append(padding_list(
                [item for item in article_idf_tags[i]] + [0], max_encoder_len + 1, padding_val=0))
            encoder_inputs.append(padding_list(
                [input_vocab.to_idx(token) for token in encoder_tokens[i]],
                max_encoder_len + 1, input_vocab.to_idx(PAD)))
            enc_idxes_on_ext_voc = [input_vocab.to_idx(PAD) for _ in range(max_encoder_len + 1)]
            for j, token in enumerate(encoder_tokens[-1]):
                if output_vocab.has(token):
                    enc_idxes_on_ext_voc[j] = output_vocab.to_idx(token)
                else:
                    batch_oov_vocabs[i].add_word(token)
                    enc_idxes_on_ext_voc[j] = output_vocab.size + batch_oov_vocabs[i].to_idx(token)
            enc_idxes_on_ext_voc_list.append(enc_idxes_on_ext_voc)
            decoder_tokens.append(abstracts[i] + [EOS])
            decoder_pos_tags.append(padding_list(
                [POS_TO_IDX[item] for item in summary_pos_tags[i]] +
                [POS_TO_IDX["."]], max_decoder_len + 1, padding_val=POS_TO_IDX["."]))
            decoder_ner_tags.append(padding_list(
                [NER_TO_IDX[item] for item in summary_ner_tags[i]] +
                [NER_TO_IDX["O"]], max_decoder_len + 1, padding_val=NER_TO_IDX["O"]))
            decoder_inputs.append(padding_list(
                [input_vocab.to_idx(token) for token in [SOS] + decoder_tokens[i][:-1]],
                max_decoder_len + 1, input_vocab.to_idx(PAD)))
            decoder_targets.append(padding_list(
                [output_vocab.to_idx(token) for token in decoder_tokens[i]], max_decoder_len + 1,
                output_vocab.to_idx(PAD)))

        decoder_ext_targets = [item[:] for item in decoder_targets]
        for i, token_list in enumerate(decoder_tokens):
            for j, token in enumerate(token_list):
                if not output_vocab.has(token) and batch_oov_vocabs[i].has(token):
                    decoder_ext_targets[i][j] = output_vocab.size + batch_oov_vocabs[i].to_idx(token)
        batch = {
            "encoder_tokens": encoder_tokens,
            "encoder_pos_tags": encoder_pos_tags,
            "encoder_ner_tags": encoder_ner_tags,
            "encoder_tf_tags": encoder_tf_tags,
            "encoder_idf_tags": encoder_idf_tags,
            "decoder_tokens": [example.abstract + ["<eos>"] for example in examples],
            "decoder_pos_tags": decoder_pos_tags,
            "decoder_ner_tags": decoder_ner_tags,
            "encoder_lens": encoder_lens,
            "decoder_lens": decoder_lens,
            "encoder_inputs": encoder_inputs,
            "decoder_inputs": decoder_inputs,
            "decoder_targets": decoder_targets,
            "decoder_ext_targets": decoder_ext_targets,
            "enc_idxes_on_ext_voc": enc_idxes_on_ext_voc_list,
            "batch_oov_vocabs": batch_oov_vocabs
        }
        return batch

    def create_batches(self, batch_size, examples, input_vocab, output_vocab, early_stop, args):
        total_steps = len(examples) // batch_size
        batches = []
        for i in range(total_steps):
            st = i * batch_size
            end = (i + 1) * batch_size
            batches.append(self.create_one_batch(examples[st:end], input_vocab, output_vocab, args))
            if i % 10:
                sys.stdout.write("\r{}/{} batch created".format(i, total_steps))
                sys.stdout.flush()
            if early_stop is not None and i > early_stop:
                break
        print()
        return batches

    def generate(self, batch_size, input_vocab, output_vocab, args, _type="train",
                 batch_type="word", shuffle=False, early_stop=None):
        if _type == "all":
            examples = self.train + self.valid + self.test
        else:
            examples = self.__getattribute__(_type)
        self.early_stop = early_stop
        if shuffle:
            idxes = list(range(len(examples)))
            random.shuffle(idxes)
            examples = [examples[i] for i in idxes]
        if batch_type == "word":
            batches = self.create_batches(batch_size, examples, input_vocab, output_vocab, early_stop, args)
        elif batch_type == "sentence":
            batches = self.create_sentence_batches(batch_size, examples, input_vocab, output_vocab, early_stop, args)
        # if _type == "valid":
        #     batches = batches[:10]
        yield len(batches)
        idxes = list(range(len(batches)))
        while True:
            for idx in idxes:
                yield batches[idx]


class CNNArticleSummary(object):
    def __init__(self, article_str, abstract_str):
        self.article = article_str.strip().split(" ")
        self.abstract = abstract_str.replace("<s>", "").replace("</s>", "").strip().split(" ")
        self.abstract_sentences = self.abstract2sents(abstract_str)

    @staticmethod
    def abstract2sents(abstract):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        cur = 0
        sents = []
        while True:
            try:
                start_p = abstract.index(SENTENCE_START, cur)
                end_p = abstract.index(SENTENCE_END, start_p + 1)
                cur = end_p + len(SENTENCE_END)
                sents.append(abstract[start_p + len(SENTENCE_START):end_p])
            except ValueError as e:  # no more sentences
                return sents

    def __str__(self):
        s = "=" * 10 + " SUMMARY " + "=" * 10 + "\n"
        s += " ".join(self.abstract) + "\n"
        s += "\n"
        s += "=" * 10 + " Article " + "=" * 10 + "\n"
        s +=  " ".join(self.article)
        s += "\n"
        return s


class CNNDataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train = self.create_dataset("train")
        self.valid = self.create_dataset("val")
        self.test = self.create_dataset("test")

    def create_dataset(self, _type):
        dataset = []
        with open(os.path.join(self.data_path, "%s_000.bin" % _type), "rb") as reader:
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break  # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                e = example_pb2.Example.FromString(example_str)
                article = e.features.feature['article'].bytes_list.value[0]
                abstract = e.features.feature['abstract'].bytes_list.value[0]
                dataset.append(CNNArticleSummary(article.decode("utf-8"), abstract.decode("utf-8")))
        return dataset

    def generate(self, batch_size, input_vocab, output_vocab, args, _type="train", shuffle=False, early_stop=None):
        examples = self.__getattribute__(_type)
        if shuffle:
            idxes = list(range(len(examples)))
            random.shuffle(idxes)
            examples = [examples[i] for i in idxes]
        batches = create_batches(batch_size, examples, input_vocab, output_vocab, early_stop, args)
        if _type == "valid":
            batches = batches[:10]
        yield len(batches)
        idxes = list(range(len(batches)))
        while True:
            for idx in idxes:
                yield batches[idx]


if __name__ == "__main__":
    import pickle
    dataset = AMIDataset("/home/panhaojie/AbsDialogueSum/data")
    # print(dataset.train[0])
    # dataset = CNNDataset("/home/panhaojie/AbsDialogueSum/data/cnn-data/chunked")
    with open("/home/panhaojie/AbsDialogueSum/data/ami-vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    train_iter = dataset.generate(batch_size=10, input_vocab=vocab["input_vocab"],
                                  output_vocab=vocab["output_vocab"], _type="test", batch_type="word", args=None)
    num = next(train_iter)
    batch = next(train_iter)
    pass