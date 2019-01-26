import argparse


def add_basic_arguments(parser):
    parser.add_argument('--data-path', type=str, default="/home/panhaojie/AbsDialogueSum/data", help='data directory')
    parser.add_argument('--batch-size', type=int, default='32', help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default='1', help='Batch size for testing')
    parser.add_argument('--epoch-num', type=int, default='30', help='Total Epoch num')
    parser.add_argument('--model_path', type=str, default=None, help='Trained model path')
    parser.add_argument('--model', type=str, default='encdec', help='Model Type')
    parser.add_argument('--encoder-max-len', type=int, default='400', help='Max steps of encoder')
    parser.add_argument('--decoder-max-len', type=int, default='100', help='Max steps of decoder')
    parser.add_argument('--max-sent-num', type=int, default='8',
                        help='Max steps of Hierarchical sentence decoder (sentence level)')
    parser.add_argument('--max-sent-len', type=int, default='20',
                        help='Max steps of Hierarchical sentence decoder (word level)')
    parser.add_argument('--rnn-type', type=str, default='LSTM', help='RNN Cell type')
    parser.add_argument('--beam-size', type=int, default='2', help='beam search size')
    parser.add_argument('--hidden-size', type=int, default='100', help='RNN hidden size')
    parser.add_argument('--word-embed-size', type=int, default='50', help='Word Embedding size')
    parser.add_argument('--n_layers', type=int, default='1', help='RNN number of Layer')
    parser.add_argument('--dropout', type=float, default='0', help='Dropout')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--use-cuda', default=False, action='store_true',
                        help='Use GPU if it is true')
    parser.add_argument('--print-every', type=int, default='10', help='print loss every when training')
    parser.add_argument('--test', default=False, action='store_true',
                        help='test model if it is true')
    parser.add_argument('--dataset', default='cnn',
                        help="""Dataset Type """)
    parser.add_argument('--load_model', default=None,
                        help="""Model filename (the model will be saved as
                                <save_model>_epochN_PPL.pt where PPL is the
                                validation perplexity""")
    parser.add_argument('--load_w2v', default=None, type=str,
                        help="""W2V Model""")
    parser.add_argument('--path-prefix', default=None, type=str,
                        help="""Prefix for file path""")
    parser.add_argument('--save_model', default='model',
                        help="""Model filename (the model will be saved as
                                <save_model>_epochN_PPL.pt where PPL is the
                                validation perplexity""")
    parser.add_argument('--feature_rich', default=False, action='store_true',
                        help='Use NER/POS/TF/IDF TAG if needed')
    parser.add_argument('--reinforce', default=False, action='store_true',
                        help='Use reinforce learning loss if needed')
    parser.add_argument('--sample_times', type=int, default='1', help='Sample times during training')
    parser.add_argument('--alpha', type=float, default=1,
                        help='mixed ml & rl')


def add_boundary_arguments(parser):
    parser.add_argument('--bd-mid-size', type=int, default='128', help='Mid Dim of Boundary Detector')


def add_optimizer_arguments(parser):
    parser.add_argument('-optim', default='sgd',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-adagrad_accumulator_init', type=float, default=0,
                        help="""Initializes the accumulator values in adagrad.
                        Mirrors the initial_accumulator_value option
                        in the tensorflow adagrad (use 0.1 for their default).
                        """)
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-adam_beta1', type=float, default=0.9,
                        help="""The beta1 parameter used by Adam.
                        Almost without exception a value of 0.9 is used in
                        the literature, seemingly giving good results,
                        so we would discourage changing this value from
                        the default without due consideration.""")
    parser.add_argument('-adam_beta2', type=float, default=0.999,
                        help="""The beta2 parameter used by Adam.
                        Typically a value of 0.999 is recommended, as this is
                        the value suggested by the original paper describing
                        Adam, and is also the value adopted in other frameworks
                        such as Tensorflow and Kerras, i.e. see:
                        https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                        https://keras.io/optimizers/ .
                        Whereas recently the paper "Attention is All You Need"
                        suggested a value of 0.98 for beta2, this parameter may
                        not work well for normal models / default
                        baselines.""")
    parser.add_argument('-lr', type=float, default=1.0,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-exp_host', type=str, default="",
                        help="Send logs to this crayon server.")
    parser.add_argument('-exp', type=str, default="",
                        help="Name of the experiment for logging.")


