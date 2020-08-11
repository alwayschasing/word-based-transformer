#!/usr/bin/env python
# -*- encoding:utf-8 -*
import jieba
import pickle
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

def load_origin_vocab(vocab_file):
    pass

def gen_vocab(file_list, ori_vocab, vocab_dict_file):
    jieba.load_userdict(ori_vocab)
    vocab_dict = dict()
    for file_name in tqdm(file_list):
        logging.info("process %s" % file_name)
        with open(file_name, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                words = jieba.cut(line)
                for w in words:
                    if w in vocab_dict:
                        vocab_dict[w] += 1
                    else:
                        vocab_dict[w] = 1

    logging.info("vocab_dict len:%d" % len(vocab_dict))
    wfp = open(vocab_dict_file, "wb")
    pickle.dump(vocab_dict, wfp)
    wfp.close()


if __name__ == "__main__":
    file_list = []
    ori_vocab = ""
    vocab_dict_file = ""
    gen_vocab(file_list, ori_vocab, vocab_dict_file)
