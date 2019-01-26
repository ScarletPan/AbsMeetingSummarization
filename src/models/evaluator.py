from collections import Counter
from random import randint
import subprocess
from gensim.summarization import summarize
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.basic.constant import PAD, SOS, SOD, EOS
from src.preprocess.utils import inds_to_tokens_2d, inds_to_tokens_1d_2voc
from src.basic.test_rouge import test_rouge


class Evaluator(object):
    def __init__(self, vocab, model, logger, args):
        self.vocab = vocab
        self.model = model
        self.logger = logger
        self.args = args
        self.test_loss = nn.NLLLoss(size_average=False)
        self.res = {}

    def _run_encdec_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_start_input = decoder_start_input.cuda()
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.test_batch_size)
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            decoder_start_input=decoder_start_input, max_len=self.args.decoder_max_len, beam_size=self.args.beam_size,
            eos_val=self.vocab["output_vocab"].to_idx(EOS))
        pred = decoder_preds.data.tolist()
        pred_tokens = inds_to_tokens_2d(
            pred, self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        return {
            "pred_tokens": pred_tokens
        }

    def _run_intra_attn_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        enc_idxes_on_ext_voc = Variable(torch.LongTensor(batch_data["enc_idxes_on_ext_voc"]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_start_input = decoder_start_input.cuda()
            enc_idxes_on_ext_voc = enc_idxes_on_ext_voc.cuda()

        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.test_batch_size)
        ext_oov_sizes = [vocab.size for vocab in batch_data["batch_oov_vocabs"]]
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            decoder_start_input=decoder_start_input, max_len=self.args.decoder_max_len, beam_size=1,
            ext_oov_sizes=ext_oov_sizes, enc_idxes_on_ext_voc=enc_idxes_on_ext_voc)
        pred = decoder_preds.data.tolist()
        pred_tokens = inds_to_tokens_2d(
            pred, self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        # pred_tokens = []
        # ext_vocabs = batch_data["batch_oov_vocabs"]
        # for b in range(encoder_inputs.size(0)):
        #     ext_vocab = ext_vocabs[b]
        #     pred_token = inds_to_tokens_1d_2voc(decoder_preds[b].data, self.vocab["output_vocab"], ext_vocab,
        #                                         eliminate_tokens=["<pad>", "<sos>"], end_tokens="<eos>")
        #     pred_tokens.append(pred_token)
        return {
            "pred_tokens": pred_tokens
        }

    def _run_hierarchical_batch(self, batch_data):
        word_encoder_inputs_list = [Variable(torch.LongTensor(x)) for x in batch_data["word_encoder_inputs_list"]]
        b_size = word_encoder_inputs_list[0].size(0)
        sent_decoder_start_input = Variable(torch.LongTensor(b_size * [[self.vocab["input_vocab"].to_idx(SOD)]]))
        word_decoder_start_input = Variable(torch.LongTensor(b_size * [[self.vocab["input_vocab"].to_idx(SOS)]]))
        if self.args.use_cuda:
            word_encoder_inputs_list = [x.cuda() for x in word_encoder_inputs_list]
            sent_decoder_start_input = sent_decoder_start_input.cuda()
            word_decoder_start_input = word_decoder_start_input.cuda()
        pred_list = self.model.generate(word_encoder_inputs_list,
                                        word_decoder_start_input, max_sent_num=self.args.max_sent_num,
                                        max_sent_len=self.args.max_sent_len)
        pred_tokens_list = [[] for _ in range(b_size)]
        for pred in pred_list:
            pred = pred.squeeze(2).data.tolist()
            pred_tokens_list = [pred_tokens_list[i] + inds_to_tokens_2d(
                [pred[i]], self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
                for i in range(b_size)]
        for i in range(len(pred_tokens_list)):
            pred_tokens_list[i] = [x for pred_tokens in pred_tokens_list[i] for x in pred_tokens]
        result_dict = {
            "pred_tokens": pred_tokens_list
        }
        return result_dict

    def _run_boundary_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        if self.args.feature_rich:
            encoder_extra_inputs = [Variable(torch.LongTensor(batch_data["encoder_pos_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_ner_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_tf_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_idf_tags"]))]
        else:
            encoder_extra_inputs = None
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            encoder_extra_inputs = [item.cuda() for item in encoder_extra_inputs] if encoder_extra_inputs else None
            decoder_start_input = decoder_start_input.cuda()
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=encoder_inputs, encoder_extra_inputs=encoder_extra_inputs,
            decoder_start_input=decoder_start_input,
            max_len=self.args.decoder_max_len, beam_size=self.args.beam_size,
            eos_val=self.vocab["output_vocab"].to_idx(EOS))
        pred = decoder_preds.data.tolist()
        pred_tokens = inds_to_tokens_2d(
            pred, self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        return {
            "pred_tokens": pred_tokens
        }

    def _run_feature_rich_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        encoder_extra_inputs = [Variable(torch.LongTensor(batch_data["encoder_pos_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_ner_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_tf_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_idf_tags"]))]
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            encoder_extra_inputs = [item.cuda() for item in encoder_extra_inputs]
            decoder_start_input = decoder_start_input.cuda()
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.test_batch_size)
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            encoder_extra_inputs=encoder_extra_inputs,decoder_start_input=decoder_start_input,
            max_len=self.args.decoder_max_len, beam_size=self.args.beam_size,
            eos_val=self.vocab["output_vocab"].to_idx(EOS))
        pred = decoder_preds.data.tolist()
        pred_tokens = inds_to_tokens_2d(
            pred, self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        return {
            "pred_tokens": pred_tokens
        }

    def _run_text_rank_batch(self, batch_data):
        pred_tokens = []
        for encoder_tokens in batch_data["encoder_tokens"]:
            article = " ".join(encoder_tokens)
            pred_tokens.append(summarize(article, word_count=self.args.decoder_max_len).split(" "))
        return {
            "pred_tokens": pred_tokens
        }

    def run_batch(self, batch_data):
        if self.args.model in ["encdec", "attn-encdec", "b-attn-encdec"]:
            return self._run_encdec_batch(batch_data)
        elif self.args.model in ["intra-attn"]:
            return self._run_intra_attn_batch(batch_data)
        elif self.args.model in ["hierarchical-encdec", "attn-hierarchical-encdec"]:
            return self._run_hierarchical_batch(batch_data)
        elif self.args.model == "textRank":
            return self._run_text_rank_batch(batch_data)
        elif self.args.model in ["boundary-encdec", "hybrid-boundary-encdec"]:
            return self._run_boundary_batch(batch_data)
        elif self.args.model in ["fr-encdec"]:
            return self._run_feature_rich_batch(batch_data)

    def test(self, num_in_total, test_iter):
        if self.args.model not in ["textRank"]:
            self.model.eval()
        f_cand = open("cache/%s/candidate-test.txt" % (self.args.model + "-" + self.args.dataset
                      + "-" + self.args.path_prefix), "w", encoding="utf-8")
        f_ref = open("cache/%s/reference-test.txt" % (self.args.model + "-" + self.args.dataset
                     + "-" + self.args.path_prefix), "w", encoding="utf-8")
        rand_print_idx = randint(0, num_in_total - 1)
        summaries = []
        for i in range(num_in_total):
            batch_data = next(test_iter)
            result_dict = self.run_batch(batch_data)
            pred_tokens = result_dict["pred_tokens"]
            f_ref.write(" ".join(batch_data["decoder_tokens"][0]) + "\n")
            f_cand.write(" ".join(pred_tokens[0]) + "\n")
            if i == rand_print_idx:
                tmp_cand_str = " ".join(pred_tokens[0])
                tmp_ref_str = " ".join(batch_data["decoder_tokens"][0]) + "\n"
            summaries.append(" ".join(pred_tokens[0]))

        f_cand.close()
        f_ref.close()

        self.test_bleu()
        self.test_rouge()
        # self.logger.info("\n" + "="*10 + " RANDOM SAMPLE " + "="*10)
        # self.logger.info("REFERENCE: " + tmp_ref_str)
        # self.logger.info("GENERATED: " + tmp_cand_str)
        # self.logger.info("UNIQUE NUM: {}/{}".format(len(Counter(summaries)), len(summaries)))
        if self.args.model not in ["textRank"]:
            self.model.train()
        return self.res

    def test_bleu(self):
        res = subprocess.check_output("perl multi-bleu.perl "
                                      "cache/%s/reference-test.txt < cache/%s/candidate-test.txt"
                                      % ((self.args.model + "-" + self.args.dataset  + "-" + self.args.path_prefix),
                                         (self.args.model + "-" + self.args.dataset + "-" + self.args.path_prefix)),
                                      shell=True).decode("utf-8")
        self.logger.info(">> " + res.strip())
        self.res["bleu_output"] = res.strip()

    def test_rouge(self):
        results_dict = test_rouge(
            cand_file="cache/%s/candidate-test.txt" % (self.args.model + "-" + self.args.dataset
                                                       + "-" + self.args.path_prefix),
            ref_file="cache/%s/reference-test.txt" % (self.args.model + "-" + self.args.dataset
                                                      + "-" + self.args.path_prefix),
        prefix=self.args.path_prefix)
        log_str = ">> ROUGE(1/2/SU4): {:.2f}/{:.2f}/{:.2f}".format(results_dict["rouge_1_f_score"] * 100,
                                                                    results_dict["rouge_2_f_score"] * 100 ,
                                                                    results_dict["rouge_su*_f_score"] * 100)
        self.logger.info(log_str)

        self.res["rouge_output"] = log_str
        self.res["rouge_su4"] = results_dict["rouge_su*_f_score"] * 100
