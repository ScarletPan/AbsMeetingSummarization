import argparse
from collections import Counter
import logging
import os
import pickle
import subprocess
import sys
import time
from gensim.summarization import summarize
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.basic.constant import EOS, PAD, SOS
from src.basic.test_rouge import test_rouge
from src.preprocess.utils import inds_to_tokens_2d
from src.models.boundary import BoundaryEncoderDecoder, HybridBoundaryEncoderDecoder
from src.models.encdec import AttentionEncoderDecoder, FeatureRicherEncoderDecoder


class Evaluator(object):
    def __init__(self, vocab, model, logger, args, model_name):
        self.vocab = vocab
        self.model = model
        self.model_name = model_name
        self.logger = logger
        self.args = args
        self.test_loss = nn.NLLLoss(size_average=False)

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
        f_cand = open("%s/candidate/%s.txt" % (self.args.ckp_path, self.model_name)
                      , "w", encoding="utf-8")
        summaries = []
        for i in range(num_in_total):
            batch_data = next(test_iter)
            result_dict = self.run_batch(batch_data)
            pred_tokens = result_dict["pred_tokens"]
            f_cand.write(" ".join(pred_tokens[0]) + "\n")
            summaries.append(" ".join(pred_tokens[0]))
            sys.stdout.write("\r{}/{} decoded".format(i, num_in_total))
            sys.stdout.flush()

        print()
        f_cand.close()

        if self.args.report_score:
            self.test_bleu()
            # self.test_rouge()
            self.logger.info("UNIQUE NUM: {}/{}".format(len(Counter(summaries)), len(summaries)))
        if self.args.model not in ["textRank"]:
            self.model.train()

    def test_bleu(self):
        res = subprocess.check_output("perl multi-bleu.perl "
                                      "%s/reference.txt < %s/candidate/%s.txt"
                                      % (self.args.ckp_path, self.args.ckp_path, self.args.model),
                                      shell=True).decode("utf-8")
        self.logger.info(">> " + res.strip())

    def test_rouge(self):
        results_dict = test_rouge(
            cand_file="%s/candidate/%s.txt" % (self.args.ckp_path, self.model_name),
            ref_file="%s/reference.txt" % self.args.ckp_path)
        log_str = ">> ROUGE(1/2/SU4): {:.2f}/{:.2f}/{:.2f}".format(results_dict["rouge_1_f_score"] * 100,
                                                                    results_dict["rouge_2_f_score"] * 100 ,
                                                                    results_dict["rouge_su*_f_score"] * 100)
        self.logger.info(log_str)


def config_logs(args):
    if not os.path.isdir(os.path.join(args.ckp_path, "logs")):
        os.makedirs(os.path.join(args.ckp_path, "logs"))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(args.ckp_path, "logs",
                                              "{}.outputs.log".format(
                                                  time.strftime('%Y-%m-%d-%H-%M-%S',
                                                                time.localtime()))),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger("main")
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default="data",
                        help='checkpoint directory')
    parser.add_argument('--ckp-path', type=str,
                        default="cpt",
                        help='checkpoint directory')
    parser.add_argument('--use-cuda', default=True, action='store_true',
                        help='Use GPU if it is true')
    parser.add_argument('--test-batch-size', type=int, default='1', help='Batch size for testing')
    parser.add_argument('--beam-size', type=int, default='1', help='beam search size')
    parser.add_argument('--encoder-max-len', type=int, default='6000', help='Max steps of encoder')
    parser.add_argument('--decoder-max-len', type=int, default='160', help='Max steps of decoder')
    parser.add_argument('--model', type=str, default='encdec', help='Model Type')
    parser.add_argument('--feature_rich', default=False, action='store_true',
                        help='Use NER/POS/TF/IDF TAG if needed')
    parser.add_argument('--report_score', default=False, action='store_true',
                        help='Report Score if it is true')
    # args = parser.parse_args([
    #     # "--ckp-path", "checkpoints/cpt-2",
    #     "--ckp-path", "checkpoints/param_tuning",
    #     "--report_score"
    # ])
    args = parser.parse_args()
    logger = config_logs(args)
    data_root = args.data_root
    model_root = os.path.join(args.ckp_path, "models")
    if not os.path.exists(os.path.join(args.ckp_path, "candidate")):
        os.makedirs(os.path.join(args.ckp_path, "candidate"))

    models = [model for model in os.listdir(model_root)]
    with open(os.path.join(data_root, "ami-vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    print("Load Dataset ami")
    with open(os.path.join(data_root, "ami-dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)

    if not os.path.exists(os.path.join(args.ckp_path, "reference.txt")):
        f_ref = open(os.path.join(args.ckp_path, "reference.txt"), "w", encoding="utf-8")
        test_iter = dataset.generate(batch_size=1, input_vocab=vocab["input_vocab"],
                                     output_vocab=vocab["output_vocab"], args=args, _type="test")
        num_in_total = next(test_iter)
        for i in range(num_in_total):
            batch_data = next(test_iter)
            f_ref.write(" ".join(batch_data["decoder_tokens"][0]) + "\n")
        f_ref.close()
    for model_name in models:
        ckp = torch.load(os.path.join(model_root, model_name))
        model_name = model_name.replace(".pt", "")
        print("\n")
        print("=" * 10, model_name, "=" * 10)
        model = ckp["model"]
        logger.info(model)
        if isinstance(model, AttentionEncoderDecoder):
            args.model = "attn-encdec"
        elif isinstance(model, FeatureRicherEncoderDecoder):
            args.model = "fr-encdec"
        elif isinstance(model, BoundaryEncoderDecoder):
            args.model = "boundary-encdec"
            args.feature_rich = True
        elif isinstance(model, HybridBoundaryEncoderDecoder):
            args.model = "hybrid-boundary-encdec"
            args.feature_rich = True
        else:
            continue
        evaluator = Evaluator(vocab, model, logger, args, model_name)
        test_iter = dataset.generate(batch_size=1, input_vocab=vocab["input_vocab"],
                                         output_vocab=vocab["output_vocab"], args=args, _type="test")
        num_in_total = next(test_iter)
        evaluator.test(num_in_total=num_in_total, test_iter=test_iter)