#!/usr/bin/env python
# -*- encoding:utf-8 -*
import jieba
import pickle
import os
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

def load_origin_vocab(vocab_file):
    pass

def gen_vocab(file_list, added_vocab_list, vocab_dict_file):
    jieba.enable_parallel(30)
    for vocab_file in added_vocab_list:
        jieba.load_userdict(vocab_file)
    vocab_dict = dict()
    idx = 0
    for file_name in file_list:
        logging.info("process %s" % file_name)
        with open(file_name, "r", encoding="utf-8") as fp:
            content = fp.read()
            words = jieba.cut(content)
            for w in words:
                if w in vocab_dict:
                    vocab_dict[w] += 1
                else:
                    vocab_dict[w] = 1
            #lines = fp.readlines()
            #logging.info("readlines %d"%(len(lines)))
            #for line in tqdm(lines):
            #    line = line.strip()
            #    words = jieba.cut(line)
            #    for w in words:
            #        if w in vocab_dict:
            #            vocab_dict[w] += 1
            #        else:
            #            vocab_dict[w] = 1
            #    idx += 1
    logging.info("vocab_dict len:%d" % len(vocab_dict))
    wfp = open(vocab_dict_file, "wb")
    pickle.dump(vocab_dict, wfp)
    wfp.close()

if __name__ == "__main__":
    article_dir="/search/odin/liruihong/article_data"
    file_list = ["20200701_15"]
    file_list = [os.path.join(article_dir,x) for x in file_list]
    vocab_list = ["/search/odin/liruihong/word2vec_embedding/keywords_vocab","/search/odin/liruihong/word2vec_embedding/tencent_ChineseEmbedding/Tencent_Chinese_vocab.txt"]
    vocab_dict_file = "/search/odin/liruihong/word-based-transformer/data/vocab_count.bin"
    gen_vocab(file_list, vocab_list, vocab_dict_file)
