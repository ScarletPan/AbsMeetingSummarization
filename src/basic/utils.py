import logging
import os
import time


def make_cache_files(dataset, args):
    if not os.path.isdir("cache"):
        os.mkdir("cache")
    if not os.path.isdir(os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix)):
        os.mkdir(os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix))
        os.mkdir(os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix, "documents"))
        os.mkdir(os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix, "logs"))
        for i, example in enumerate(dataset.test):
            with open(os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix, "documents", "doc.%s.txt" % i),
                      "w", encoding="utf-8") as f:
                f.write(" ".join(example.__str__()) + "\n")


def config_logs(args):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join("cache", args.model + "-" + args.dataset + "-" + args.path_prefix, "logs",
                                              "{}.outputs.log".format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))),
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
