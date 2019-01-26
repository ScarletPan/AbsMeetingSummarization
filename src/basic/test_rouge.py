import argparse
import os
import time
import pyrouge
import shutil


def test_rouge(cand_file, ref_file, prefix=""):
    f_cand = open(cand_file, encoding="utf-8")
    f_ref = open(ref_file, encoding="utf-8")
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time) + prefix
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
        candidates = [line.strip() for line in f_cand]
        references = [line.strip() for line in f_ref]
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        f_cand.close()
        f_ref.close()
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = 'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def test_sentences_rouge(candidates, references, score_type="l", prefix=""):
    assert len(candidates) == len(references)
    cnt = len(candidates)
    result_list = []
    for i in range(cnt):
        if len(references[i]) < 1:
            continue
        try:
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            tmp_dir = ".rouge-tmp-{}".format(current_time) + prefix
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
            with open(tmp_dir + "/candidate/cand.0.txt", "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.0.txt", "w",
                      encoding="utf-8") as f:
                f.write(references[i])
            r = pyrouge.Rouge155()
            r.model_dir = tmp_dir + "/reference/"
            r.system_dir = tmp_dir + "/candidate/"
            r.model_filename_pattern = 'ref.#ID#.txt'
            r.system_filename_pattern = 'cand.(\d+).txt'
            rouge_results = r.convert_and_evaluate()
            results_dict = r.output_to_dict(rouge_results)
            result_list.append(results_dict["rouge_%s_f_score" % score_type])
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
    return result_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', type=str, default="candidate.txt", help='candidate file')
    # parser.add_argument('-r', type=str, default="reference.txt", help='reference file')
    # args = parser.parse_args()
    # results_dict = test_rouge(args.c, args.r)
    # log_str = ""
    # log_str += ">> ROUGE(1/2/SU4): {:.2f}/{:.2f}/{:.2f}".format(results_dict["rouge_1_f_score"] * 100,
    #                                                             results_dict["rouge_2_f_score"] * 100,
    #                                                             results_dict["rouge_su*_f_score"] * 100)
    # print(log_str)
    from src.preprocess.vocab import Vocabulary
    references = ["this is just a test",
                  "hello my new world , I love you pretty much",
                  "glad to meet you in this house"]
    candidates = ["this is a test",
                  "hello my great world , I love you so much",
                  "glad to see you"]
    vocab = Vocabulary()
    for token in " ".join(references).split(" ") + " ".join(candidates).split(" "):
        vocab.add_word(token)
    references_digit = [" ".join([str(vocab.to_idx(token)) for token in sent.split()]) for sent in references]
    candidates_digit = [" ".join([str(vocab.to_idx(token)) for token in sent.split()]) for sent in candidates]
    res = test_sentences_rouge(candidates, references, score_type="l")
    print(res, sum(res) / 3)
    res_digit = test_sentences_rouge(references_digit, candidates_digit, score_type="l")
    print(res_digit, sum(res_digit) / 3)
    print()