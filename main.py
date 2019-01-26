import argparse
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.basic.add_args import add_basic_arguments, add_boundary_arguments, add_optimizer_arguments
from src.basic.constant import POS_TO_IDX, NER_TO_IDX
from src.basic.utils import make_cache_files, config_logs
from src.models.boundary import BoundaryEncoderDecoder, HybridBoundaryEncoderDecoder
from src.models.encdec import EncoderDecoder, AttentionEncoderDecoder, BahdanauAttnEncoderDecoder, \
    FeatureRicherEncoderDecoder
from src.models.hierarchical import BasicHierarchicalEncoderDecoder, AttentionHierarchicalEncoderDecoder
from src.models.intra_attn_seq2seq import IntraEncoderDecoder
from src.models.evaluator import Evaluator
from src.models.trainer import Trainer
from src.preprocess.vocab import Vocabulary
from src.preprocess.preprocess import AMIDataset, CNNDataset
from src.models.optimizer import Optim


def train(dataset, model, vocab, args, logger, pre_model=None):
    logger.info(model)
    input_vocab = vocab["input_vocab"]
    output_vocab = vocab["input_vocab"]
    if args.model in ["hierarchical-encdec", "attn-hierarchical-encdec"]:
        train_iter = dataset.generate(batch_size=args.batch_size, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, batch_type="sentence", _type="train", early_stop=None)
        valid_iter = dataset.generate(batch_size=args.batch_size, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, batch_type="sentence", _type="valid", early_stop=None)
        test_iter = dataset.generate(batch_size=args.batch_size, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, batch_type="sentence", _type="valid", early_stop=None)
    else:
        train_iter = dataset.generate(batch_size=args.batch_size, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, _type="train", early_stop=None)
        valid_iter = dataset.generate(batch_size=args.batch_size, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, _type="test", early_stop=None)
        test_iter = dataset.generate(batch_size=1, input_vocab=input_vocab, args=args,
                                      output_vocab=output_vocab, _type="test", early_stop=None)
    num_in_total_test = next(test_iter)
    evaluator = Evaluator(vocab, model, logger, args)
    optimizer = Optim(
                    args.optim, args.lr, args.max_grad_norm,
                    lr_decay=args.learning_rate_decay,
                    start_decay_at=args.start_decay_at,
                    beta1=args.adam_beta1,
                    beta2=args.adam_beta2,
                    adagrad_accum=args.adagrad_accumulator_init,
                    opt=args
                )
    optimizer.set_parameters(model.parameters())
    trainer = Trainer(model=model, vocab=vocab, train_iter=train_iter, valid_iter=valid_iter,
                      train_loss=nn.NLLLoss(size_average=False), valid_loss=nn.NLLLoss(size_average=False),
                      optim=optimizer, logger=logger, args=args, pre_model=pre_model)
    best_ppl = np.inf
    best_bleu_output = ""
    best_rouge_output = ""
    best_rouge = 0
    for epoch in range(args.epoch_num):
        logger.info('\n')
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, print_every=args.print_every)
        logger.info('Train loss: %g' % train_stats.get_loss())
        logger.info('Train perplexity: %g' % train_stats.ppl())
        logger.info('Train accuracy: %g' % train_stats.accuracy())
        # 2. Validate on the validation set.
        # valid_stats = trainer.validate()
        # logger.info('Validation loss: %g' % valid_stats.get_loss())
        # logger.info('Validation perplexity: %g' % valid_stats.ppl())
        # logger.info('Validation accuracy: %g' % valid_stats.accuracy())
        evaluator.model = trainer.model
        res = evaluator.test(num_in_total=num_in_total_test, test_iter=test_iter)
        # 3. dump the checkpoints
        ckp = trainer.save_checkpoint(args, model, epoch, train_stats)
        # if args.reinforce:
        r = res["rouge_su4"]
        if r > best_rouge:
            logger.info("It's best model")
            best_bleu_output = res["bleu_output"]
            best_rouge_output = res["rouge_output"]
            best_ckp = ckp
            best_rouge = r
            torch.save(best_ckp, "models/{}/best-model.pt".format(
                args.model + "-" + args.dataset + "-" + args.path_prefix))
        # else:
        #     if valid_stats.ppl() < best_ppl:
        #         best_ckp = ckp
        #         best_ppl = valid_stats.ppl()
        #         torch.save(best_ckp, "models/{}/best-model.pt".format(
        #             args.model + "-" + args.dataset + "-" + args.path_prefix))

        # 4. Update the learning rate
        trainer.epoch_step(r, epoch)
        print("\nBest Bleu: ", best_bleu_output)
        print("Best Rouge: ", best_rouge_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_basic_arguments(parser)
    add_boundary_arguments(parser)
    add_optimizer_arguments(parser)
    # args = parser.parse_args(["--model", "attn-encdec",  # "b-attn-encdec" "attn-encdec" "encdec"
    #                           "--dataset", "ami",
    #                           "--load_model", "models/attn-encdec-ami/best-model.pt",
    #                           "--epoch-num", "10",
    #                           "--batch-size", "5",
    #                           "--hidden-size", "512",
    #                           "--word-embed-size", "256",
    #                           "--dropout", "0.3",
    #                           "--beam-size", "1",
    #                           "--print-every", "5",
    #                           "--max-sent-num", "8",
    #                           "--max-sent-len", "20",
    #                           "--use-cuda",
    #                           # "--test"
    #                           ])

    # Flatten Model Args
    # args = parser.parse_args(["--data-path", "/home/panhaojie/AbsDialogueSum/checkpoints/cpt-1/data",
    #                           "--model", "hybrid-boundary-encdec",
    #                           "--dataset", "ami",
    #                           # "--load_model", "models/hybrid-boundary-encdec-ami-h-512-e-196/model_acc_32.40_ppl_86.28_e11.pt",
    #                           # "--load_model", "checkpoints/cpt-1/models/hybrid-fr.pt",
    #                           # "--load_model", "models/hybrid-boundary-encdec-ami-reinforce/best-model.pt",
    #                           # "--load_model", "checkpoints/alpha/ml-rl-9988.pt",
    #                           "--load_model", "models/hybrid-boundary-encdec-ami-FIX-50/model_acc_47.65_ppl_12.11_e6.pt",
    #                           # "--load_w2v", "glove.6B.100d.pt",
    #                           "--epoch-num", "15",
    #                           "--batch-size", "1",
    #                           "--hidden-size", "512",
    #                           "--word-embed-size", "256",
    #                           "--dropout", "0.5",
    #                           "--beam-size", "1",
    #                           "--print-every", "20",
    #                           "--encoder-max-len", "6000",
    #                           "--decoder-max-len", "160",
    #                           "--use-cuda",
    #                           "--path-prefix", "FIX-50",
    #                           "--feature_rich",
    #                           "--n_layers", "1",
    #                           # "--reinforce",
    #                           "-max_grad_norm", "5",
    #                           "-lr", "1",
    #                           '-learning_rate_decay', "0.5",
    #                           "--alpha", "0.9985",
    #                           # "--test"
    #                           ])
    args = parser.parse_args()
    data_root = args.data_path

    if args.model in ["hierarchical-encdec", "attn-hierarchical-encdec"]:
        with open(os.path.join(data_root, "%s-vocab-hierachy.pkl" % args.dataset), "rb") as f:
            vocab = pickle.load(f)
    else:
        with open(os.path.join(data_root, "%s-vocab.pkl" % args.dataset), "rb") as f:
            vocab = pickle.load(f)
    if not os.path.exists(os.path.join(data_root, "%s-dataset.pkl" % args.dataset)):
        print("Create Dataset %s" % args.dataset)
        if args.dataset == "ami":
            dataset = AMIDataset(data_root)
        elif args.dataset == "cnn":
            dataset = CNNDataset(os.path.join(data_root, "cnn-data"))
        with open(os.path.join(data_root, "%s-dataset.pkl" % args.dataset), "wb") as f:
            pickle.dump(dataset, f)
    else:
        print("Load Dataset %s" % args.dataset)
        with open(os.path.join(data_root, "%s-dataset.pkl" % args.dataset), "rb") as f:
            dataset = pickle.load(f)

    make_cache_files(dataset, args)
    logger = config_logs(args)
    logger.info(args)
    if args.test:
        logger.info(args.load_model)
        logger.info("=" * 10 + " Test " + "=" * 10)
        if args.model not in ["textRank"]:
            ckp = torch.load(args.load_model)
            model = ckp["model"]
        else:
            model = None
        logger.info(model)
        logger.info("BEAM SIZE: {}".format(args.beam_size))
        evaluator = Evaluator(vocab, model, logger, args)
        if args.model == "hierarchical-encdec":
            test_iter = dataset.generate(batch_size=1, input_vocab=vocab["input_vocab"], args=args,
                                          output_vocab=vocab["output_vocab"], batch_type="sentence", _type="test",
                                          early_stop=None)
        else:
            test_iter = dataset.generate(batch_size=1, input_vocab=vocab["input_vocab"],
                                         output_vocab=vocab["output_vocab"], args=args, _type="test")
        num_in_total = next(test_iter)
        evaluator.test(num_in_total=num_in_total, test_iter=test_iter)
    else:
        pretrained_model = None
        tmp_vocab = vocab["output_vocab"] if args.model == "intra-attn" else vocab["input_vocab"]
        if args.model == "encdec":

            logger.info("Basic Encoder Decoder")
            model = EncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                   decoder_vocab_size=tmp_vocab.size,
                                   embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                   rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
            model.init_params()
            if args.load_w2v is not None:
                pre_embedding = torch.load(os.path.join(data_root, args.load_w2v))["embed"]
                model.load_word_vectors(pre_embedding)
        elif args.model == "attn-encdec":
            logger.info("Luong Attention Based Encoder Decoder")
            model = AttentionEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                            decoder_vocab_size=tmp_vocab.size,
                                            embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                            n_layers=args.n_layers,
                                            rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
            if args.reinforce:
                ckp = torch.load(args.load_model)
                pretrained_model = ckp["model"]
                model.init_params(pretrained_model)

        elif args.model == "b-attn-encdec":
            logger.info("Bahdanau Attention Based Encoder Decoder")
            model = BahdanauAttnEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                            decoder_vocab_size=tmp_vocab.size,
                                    embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                    rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
        elif args.model == "intra-attn":
            logger.info("Intra Attention Based Encoder Decoder")
            model = IntraEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                        decoder_vocab_size=vocab["input_vocab"].size,
                                        embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                        rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
        elif args.model == "hierarchical-encdec":
            logger.info("Basic Hierarchical Encoder Decoder")
            model = BasicHierarchicalEncoderDecoder(vocab_size=vocab["input_vocab"].size,
                                                    embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                                    rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
        elif args.model == "attn-hierarchical-encdec":
            logger.info("Attention Hierarchical Encoder Decoder")
            model = AttentionHierarchicalEncoderDecoder(vocab_size=vocab["input_vocab"].size,
                                                        embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                                        rnn_type=args.rnn_type, dropout=args.dropout,
                                                        use_cuda=args.use_cuda)
        elif args.model == "boundary-encdec":
            logger.info("Boundary Based Encoder Decoder")
            if args.feature_rich:
                extra_categorical_nums = [len(POS_TO_IDX), len(NER_TO_IDX), 11, 10]
                extra_embed_sizes = [25, 5, 10, 10]
            else:
                extra_categorical_nums = None
                extra_embed_sizes = None
            model = BoundaryEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                           decoder_vocab_size=vocab["input_vocab"].size,
                                           embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                           extra_categorical_nums=extra_categorical_nums,
                                           extra_embed_sizes=extra_embed_sizes,
                                           bd_mid_size=args.bd_mid_size, n_layers=1, dropout=0.5,
                                           use_cuda=args.use_cuda)
            model.reset_parameters()
        elif args.model == "hybrid-boundary-encdec":
            logger.info("Hybrid Boundary Based Encoder Decoder")
            if args.feature_rich:
                extra_categorical_nums = [len(POS_TO_IDX), len(NER_TO_IDX), 11, 10]
                extra_embed_sizes = [25, 5, 10, 10]
            else:
                extra_categorical_nums = None
                extra_embed_sizes = None
            model = HybridBoundaryEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                           decoder_vocab_size=vocab["input_vocab"].size,
                                           embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                           extra_categorical_nums=extra_categorical_nums,
                                           extra_embed_sizes=extra_embed_sizes,
                                           bd_mid_size=args.bd_mid_size,
                                           use_cuda=args.use_cuda)
            if args.reinforce:
                ckp = torch.load(args.load_model)
                pretrained_model = ckp["model"]
                model.init_params(pretrained_model)
            if args.load_model:
                ckp = torch.load(args.load_model)
                pretrained_model = ckp["model"]
                model.init_params(pretrained_model)
        elif args.model == "fr-encdec":
            logger.info("Feature Rich Encoder Decoder")
            model = FeatureRicherEncoderDecoder(encoder_vocab_size=vocab["input_vocab"].size,
                                                decoder_vocab_size=tmp_vocab.size,
                                                embed_size=args.word_embed_size, hidden_size=args.hidden_size,
                                                extra_categorical_nums=[len(POS_TO_IDX), len(NER_TO_IDX), 11, 10],
                                                extra_embed_sizes=[25, 5, 10, 10], n_layers=args.n_layers,
                                                rnn_type=args.rnn_type, dropout=args.dropout, use_cuda=args.use_cuda)
        if args.use_cuda:
            model = model.cuda()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Parm Counter:", count_parameters(model))
        train(dataset, model, vocab, args, logger, pretrained_model)
