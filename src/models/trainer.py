import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.basic.constant import SOS, EOS, PAD
from src.basic.test_rouge import test_sentences_rouge
from src.models.utils import batch_unpadding
from src.preprocess.utils import inds_to_tokens_2d, tokens_to_inds_2d


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, logger, loss=0, n_words=0, n_correct=0, bleu=None, rouge=None):
        self.logger = logger
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.bleu = bleu
        self.rouge = rouge
        self.cnt = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        if stat.bleu:
            self.bleu += stat.bleu
        if stat.rouge:
            if self.rouge is None:
                self.rouge = stat.rouge
            else:
                self.rouge += stat.rouge
        self.cnt += 1

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return np.exp(min(self.loss / self.n_words, 100))

    def get_rouge(self):
        return self.rouge / self.cnt

    def get_loss(self):
        return self.loss / self.n_words

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        self.logger.info(("Epoch %2d, %5d/%5d; acc: %6.2f; loss: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.loss / self.n_words,
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    def __init__(self, model, vocab, train_iter, valid_iter,
                 train_loss, valid_loss, optim, logger, args, pre_model=None):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.vocab = vocab
        self.train_iter = train_iter
        self.train_num_per_epoch = next(train_iter)
        self.valid_iter = valid_iter
        self.valid_num_per_epoch = next(valid_iter)
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.logger = logger
        self.args = args
        self.pre_model=pre_model

        # Set model in training mode.
        self.model.train()

    def _run_encdec_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_inputs = Variable(torch.LongTensor(batch_data["decoder_inputs"]))
        decoder_targets = Variable(torch.LongTensor(batch_data["decoder_targets"]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_targets = decoder_targets.cuda()
        decoder_lens = batch_data["decoder_lens"]
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.batch_size)
        decoder_probs, decoder_last_hidden = self.model(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            decoder_inputs=decoder_inputs)
        decoder_probs_pack = batch_unpadding(decoder_probs, decoder_lens)
        decoder_targets_pack = batch_unpadding(decoder_targets, decoder_lens)
        loss = self.train_loss(decoder_probs_pack, decoder_targets_pack)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().data[0]
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words
        }
        return result_dict

    def _run_reinforce_encdec_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        decoder_tokens = batch_data["decoder_tokens"]
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_start_input = decoder_start_input.cuda()
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.batch_size)
        # Compute the sample outputs
        seqs, seq_logprobs, _ = self.model.sample(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            decoder_start_input=decoder_start_input, max_len=self.args.decoder_max_len)
        pred_sample_tokens = inds_to_tokens_2d(
            seqs.data.tolist(), self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        sample_loss_list = []
        for i in range(seq_logprobs.size(0)):
            sample_loss = torch.sum(seq_logprobs[i])
            sample_loss_list.append(sample_loss)
        sample_rewards = test_sentences_rouge(
            candidates=[" ".join(tokens) for tokens in pred_sample_tokens],
            references=[" ".join(tokens) for tokens in decoder_tokens],
            score_type="s*")
        sample_rewards = [t * 100 for t in sample_rewards]
        # Compute baseline(greedy search) outputs
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            decoder_start_input=decoder_start_input, max_len=self.args.decoder_max_len, beam_size=self.args.beam_size,
            eos_val=self.vocab["output_vocab"].to_idx(EOS))
        pred_decode_tokens = inds_to_tokens_2d(
            decoder_preds.data.tolist(), self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        baseline_rewards = test_sentences_rouge(
            candidates=[" ".join(tokens) for tokens in pred_decode_tokens],
            references=[" ".join(tokens) for tokens in decoder_tokens],
            score_type="s*")
        baseline_rewards = [t * 100 for t in baseline_rewards]

        # loss_list = [(-sample_rewards[i]) * sample_loss[i] for i in range(len(sample_loss))]
        loss_list = [(baseline_rewards[i] - sample_rewards[i]) * sample_loss[i] for i in range(len(sample_loss))]
        rl_loss = torch.sum(torch.cat(loss_list))
        # Compute ML loss
        # tmp_res_dict = self._run_encdec_batch(batch_data)
        # ml_loss = tmp_res_dict["loss"]
        #
        # # Hybrid Loss
        # loss = rl_loss * 0.1 + ml_loss * 0.9
        loss = rl_loss
        num_correct = 0
        for i, tokens in enumerate(decoder_tokens):
            for j, token in enumerate(tokens):
                if len(pred_decode_tokens[i]) > j and pred_decode_tokens[i][j] == token:
                    num_correct += 1
        num_words = sum([len(item) for item in pred_decode_tokens])
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
            "sample_loss": torch.sum(sample_loss).data[0],
            "sample_reward": np.mean(sample_rewards),
            "baseline_reward": np.mean(baseline_rewards),
            "rouge": np.mean(baseline_rewards)
        }
        return result_dict

    def _run_intra_attn_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_inputs = Variable(torch.LongTensor(batch_data["decoder_inputs"]))
        decoder_targets = Variable(torch.LongTensor(batch_data["decoder_targets"]))
        enc_idxes_on_ext_voc = Variable(torch.LongTensor(batch_data["enc_idxes_on_ext_voc"]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()
            decoder_targets = decoder_targets.cuda()
            enc_idxes_on_ext_voc = enc_idxes_on_ext_voc.cuda()
        decoder_lens = batch_data["decoder_lens"]
        ext_oov_sizes = [vocab.size for vocab in batch_data["batch_oov_vocabs"]]
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.batch_size)
        decoder_probs, decoder_last_hidden = self.model(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden, decoder_inputs=decoder_inputs,
            ext_oov_sizes=ext_oov_sizes, enc_idxes_on_ext_voc=enc_idxes_on_ext_voc)
        # decoder_probs_list, decoder_last_hidden = self.model(
        #     encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden, decoder_inputs=decoder_inputs,
        #     ext_oov_sizes=ext_oov_sizes, enc_idxes_on_ext_voc=enc_idxes_on_ext_voc)
        # loss = 0
        # num_correct = 0
        # num_words = 0
        # for b in range(encoder_inputs.size(0)):
        #     loss += self.train_loss(decoder_probs_list[b][:decoder_lens[b]], decoder_targets[b][:decoder_lens[b]])
        #     _, pred = decoder_probs_list[b][:decoder_lens[b]].max(1)
        #     num_correct += pred.eq(decoder_targets[b][:decoder_lens[b]]).data[0]
        #     num_words += pred.size(0)
        decoder_probs_pack = batch_unpadding(decoder_probs, decoder_lens)
        decoder_targets_pack = batch_unpadding(decoder_targets, decoder_lens)
        loss = self.train_loss(decoder_probs_pack, decoder_targets_pack)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().data[0]
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words
        }
        return result_dict

    def _run_hierarchical_batch(self, batch_data):
        word_encoder_inputs_list = [Variable(torch.LongTensor(x)) for x in batch_data["word_encoder_inputs_list"]]
        word_decoder_inputs_list = [Variable(torch.LongTensor(x)) for x in batch_data["word_decoder_inputs_list"]]
        word_decoder_targets_list = [Variable(torch.LongTensor(x)) for x in batch_data["word_decoder_targets_list"]]
        decoder_sent_lens = batch_data["decoder_sent_lens"]
        if self.args.use_cuda:
            word_encoder_inputs_list = [x.cuda() for x in word_encoder_inputs_list]
            word_decoder_inputs_list = [x.cuda() for x in word_decoder_inputs_list]
            word_decoder_targets_list = [x.cuda() for x in word_decoder_targets_list]
        word_decoder_output_probs = self.model(word_encoder_inputs_list, word_decoder_inputs_list)
        loss = 0
        num_correct = 0
        num_words = 0
        for i, decoder_targets in enumerate(word_decoder_targets_list):
            decoder_probs = word_decoder_output_probs[i]
            if i == len(word_decoder_targets_list) - 1:
                decoder_lens = [1 for _ in range(decoder_targets.size(0))]
            else:
                decoder_lens = decoder_sent_lens[i]
            decoder_probs_pack = batch_unpadding(decoder_probs, decoder_lens)
            decoder_targets_pack = batch_unpadding(decoder_targets, decoder_lens)
            loss += self.train_loss(decoder_probs_pack, decoder_targets_pack)
            _, pred = decoder_probs_pack.max(1)
            num_correct += pred.eq(decoder_targets_pack).sum().data[0]
            num_words += pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words
        }
        return result_dict

    def _run_boundary_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        if self.args.feature_rich:
            encoder_extra_inputs = [Variable(torch.LongTensor(batch_data["encoder_pos_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_ner_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_tf_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_idf_tags"]))]
        else:
            encoder_extra_inputs = None
        decoder_inputs = Variable(torch.LongTensor(batch_data["decoder_inputs"]))
        decoder_targets = Variable(torch.LongTensor(batch_data["decoder_targets"]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            encoder_extra_inputs = [item.cuda() for item in encoder_extra_inputs] if encoder_extra_inputs else None
            decoder_inputs = decoder_inputs.cuda()
            decoder_targets = decoder_targets.cuda()
        decoder_lens = batch_data["decoder_lens"]
        decoder_probs, decoder_last_hidden = self.model(
            encoder_inputs=encoder_inputs,
            encoder_extra_inputs=encoder_extra_inputs,
            decoder_inputs=decoder_inputs)
        decoder_probs_pack = batch_unpadding(decoder_probs, decoder_lens)
        decoder_targets_pack = batch_unpadding(decoder_targets, decoder_lens)
        loss = self.train_loss(decoder_probs_pack, decoder_targets_pack)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().data[0]
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words
        }
        return result_dict

    def _run_reinforce_boundary_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        decoder_start_input = Variable(torch.LongTensor([[item[0]] for item in batch_data["decoder_inputs"]]))
        if self.args.feature_rich:
            encoder_extra_inputs = [Variable(torch.LongTensor(batch_data["encoder_pos_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_ner_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_tf_tags"])),
                                    Variable(torch.LongTensor(batch_data["encoder_idf_tags"]))]
        else:
            encoder_extra_inputs = None
        decoder_tokens = batch_data["decoder_tokens"]
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            encoder_extra_inputs = [item.cuda() for item in encoder_extra_inputs] if encoder_extra_inputs else None
            decoder_start_input = decoder_start_input.cuda()
        # Compute baseline(greedy search) outputs
        self.model.eval()
        decoder_preds, decoder_last_hidden = self.model.generate(
            encoder_inputs=Variable(encoder_inputs.data, volatile=True),
            encoder_extra_inputs=[Variable(item.data, volatile=True) for item in encoder_extra_inputs],
            decoder_start_input=Variable(decoder_start_input.data, volatile=True),
            max_len=self.args.decoder_max_len, beam_size=self.args.beam_size,
            eos_val=self.vocab["output_vocab"].to_idx(EOS))
        pred_decode_tokens = inds_to_tokens_2d(
            decoder_preds.data.tolist(), self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS],
            end_tokens=EOS)
        baseline_rewards = test_sentences_rouge(
            candidates=[" ".join(tokens) for tokens in pred_decode_tokens],
            references=[" ".join(tokens) for tokens in decoder_tokens],
            score_type="su*",
            prefix=self.args.path_prefix)
        baseline_rewards = [Variable(
            torch.FloatTensor([t * 100]), requires_grad=False) for t in baseline_rewards]
        self.model.train()
        # Compute the sample outputs
        seqs, seq_logprobs, _ = self.model.sample(
            encoder_inputs=encoder_inputs, decoder_start_input=decoder_start_input,
            max_len=self.args.decoder_max_len, encoder_extra_inputs=encoder_extra_inputs)
        pred_sample_tokens = inds_to_tokens_2d(
            seqs.data.tolist(), self.vocab["output_vocab"].to_word, eliminate_tokens=[PAD, SOS], end_tokens=EOS)
        sample_loss_list = []
        for i in range(seq_logprobs.size(0)):
            sample_loss = torch.sum(seq_logprobs[i])
            sample_loss_list.append(sample_loss)
        sample_rewards = test_sentences_rouge(
            candidates=[" ".join(tokens) for tokens in pred_sample_tokens],
            references=[" ".join(tokens) for tokens in decoder_tokens],
            score_type="su*",
            prefix=self.args.path_prefix)
        sample_rewards = [Variable(
            torch.FloatTensor([t * 100]), requires_grad=False) for t in sample_rewards]

        if self.args.use_cuda:
            baseline_rewards = [t.cuda() for t in baseline_rewards]
            sample_rewards = [t.cuda() for t in sample_rewards]
        # loss_list = [(-sample_rewards[i]) * sample_loss[i] for i in range(len(sample_loss))]
        loss_list = [(baseline_rewards[i] - sample_rewards[i]) * sample_loss_list[i]
                     for i in range(len(sample_loss_list))]
        rl_loss = torch.sum(torch.cat(loss_list))
        # Compute ML loss
        tmp_res_dict = self._run_boundary_batch(batch_data)
        ml_loss = tmp_res_dict["loss"]

        # # Hybrid Loss
        loss = rl_loss * self.args.alpha + ml_loss * (1 - self.args.alpha)
        # loss = rl_loss
        num_correct = 0
        for i, tokens in enumerate(decoder_tokens):
            for j, token in enumerate(tokens):
                if len(pred_decode_tokens[i]) > j and pred_decode_tokens[i][j] == token:
                    num_correct += 1
        num_words = sum([len(item) for item in pred_sample_tokens])
        # loss = loss / num_words
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
            "sample_loss": torch.sum(sample_loss).data[0],
            "sample_reward": np.mean([t.data[0] for t in sample_rewards]),
            "baseline_reward": np.mean([t.data[0] for t in baseline_rewards]),
            "rouge": np.mean([t.data[0] for t in baseline_rewards])
        }
        return result_dict

    def _run_feature_rich_batch(self, batch_data):
        encoder_inputs = Variable(torch.LongTensor(batch_data["encoder_inputs"]))
        encoder_extra_inputs = [Variable(torch.LongTensor(batch_data["encoder_pos_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_ner_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_tf_tags"])),
                                Variable(torch.LongTensor(batch_data["encoder_idf_tags"]))]
        decoder_inputs = Variable(torch.LongTensor(batch_data["decoder_inputs"]))
        decoder_targets = Variable(torch.LongTensor(batch_data["decoder_targets"]))
        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            encoder_extra_inputs = [item.cuda() for item in encoder_extra_inputs]
            decoder_inputs = decoder_inputs.cuda()
            decoder_targets = decoder_targets.cuda()
        decoder_lens = batch_data["decoder_lens"]
        encoder_init_hidden = self.model.init_hidden(batch_size=self.args.batch_size)
        decoder_probs, decoder_last_hidden = self.model(
            encoder_inputs=encoder_inputs, encoder_init_hidden=encoder_init_hidden,
            encoder_extra_inputs=encoder_extra_inputs, decoder_inputs=decoder_inputs)
        decoder_probs_pack = batch_unpadding(decoder_probs, decoder_lens)
        decoder_targets_pack = batch_unpadding(decoder_targets, decoder_lens)
        loss = self.train_loss(decoder_probs_pack, decoder_targets_pack)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().data[0]
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words
        }
        return result_dict

    def run_batch(self, batch_data):
        if not self.args.reinforce:
            if self.args.model in ["encdec", "attn-encdec", "b-attn-encdec"]:
                return self._run_encdec_batch(batch_data)
            elif self.args.model == "intra-attn":
                return self._run_intra_attn_batch(batch_data)
            elif self.args.model in ["hierarchical-encdec", "attn-hierarchical-encdec"]:
                return self._run_hierarchical_batch(batch_data)
            elif self.args.model in ["boundary-encdec", "hybrid-boundary-encdec"]:
                return self._run_boundary_batch(batch_data)
            elif self.args.model in ["fr-encdec"]:
                return self._run_feature_rich_batch(batch_data)
        else:
            if self.args.model in ["attn-encdec"]:
                return self._run_reinforce_encdec_batch(batch_data)
            elif self.args.model in ["hybrid-boundary-encdec"]:

                return self._run_reinforce_boundary_batch(batch_data)

    def train(self, epoch, print_every=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(logger=self.logger)
        report_stats = Statistics(logger=self.logger)

        for j in range(self.train_num_per_epoch):
            batch_data = next(self.train_iter)
            self.model.zero_grad()
            result_dict = self.run_batch(batch_data=batch_data)
            loss = result_dict["loss"]
            loss.backward()
            self.optim.step()

            batch_stats = Statistics(logger=self.logger,
                                     loss=loss.data[0],
                                     n_words=result_dict["num_words"],
                                     n_correct=result_dict["num_correct"])
            if self.args.reinforce:
                batch_stats.rouge = result_dict["rouge"]
            if self.args.model in ["hierarchical-encdec", "attn-hierarchical-encdec"]:
                encoder_lens = np.sum(batch_data["encoder_sent_lens"])
            else:
                encoder_lens = batch_data["encoder_lens"]
            report_stats.n_src_words += np.sum(encoder_lens)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            if print_every and j % print_every == 0:
                if not self.args.reinforce:
                    report_stats.output(epoch, j, self.train_num_per_epoch, total_stats.start_time)
                else:
                    t = report_stats.elapsed_time()
                    self.logger.info(("Epoch %2d, %3d/%3d; loss: %6.2f; rs: %6.4f; rb: %6.4f; " +
                                      "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                                     (epoch, j, self.train_num_per_epoch,
                                      report_stats.loss / report_stats.n_words,
                                      result_dict["sample_reward"],
                                      result_dict["baseline_reward"],
                                      report_stats.n_src_words / (t + 1e-5),
                                      report_stats.n_words / (t + 1e-5),
                                      time.time() - total_stats.start_time))
                    sys.stdout.flush()

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics(logger=self.logger)
        for j in range(self.valid_num_per_epoch):
            batch_data = next(self.valid_iter)
            result_dict = self.run_batch(batch_data=batch_data)
            batch_stats = Statistics(logger=self.logger,
                                     loss=result_dict["loss"].data[0],
                                     n_words=result_dict["num_words"],
                                     n_correct=result_dict["num_correct"])
            if self.args.reinforce:
                batch_stats.rouge = result_dict["rouge"]
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(ppl, epoch)

    def save_checkpoint(self, args, model, epoch, valid_stats):
        checkpoint = {
            'model': model,
            'args': args,
            'epoch': epoch,
            'optim': self.optim
        }
        if not os.path.isdir("models/{}".format(args.model + "-" + args.dataset + "-" + args.path_prefix)):
            os.mkdir("models/{}".format(args.model + "-" + args.dataset + "-" + args.path_prefix))
        torch.save(checkpoint,
                   'models/%s/%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (args.model + "-" + args.dataset + "-" + args.path_prefix, args.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
        return checkpoint