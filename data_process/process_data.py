#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from tqdm import tqdm
import jieba
import random
import csv
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(messgae)s", datefmt="%Y-%m-%d_%H:%M:%S")


def load_vocab_dict(vocab_file):
    word_dict = set()
    with open(vocab_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            word = parts[0]
            word_dict.add(word)
    return word_dict


def generate_cutword_file(ori_file, output_file, vocab_file):
    # jieba.enable_parallel(30)
    # word_dict = load_vocab_dict(vocab_file)
    jieba.load_userdict(vocab_file)
    wfp = open(output_file, "w", encoding="utf-8")
    with open(ori_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for line in tqdm(lines):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            title = parts[0]
            content = parts[1]

            title_words = jieba.lcut(title)
            content_words = jieba.lcut(content)
            wfp.write("%s\t%s\n"%(" ".join(title_words), " ".join(content_words)))
    wfp.close()


def generate_train_data_from_articles(cutword_file, vocab_file, train_file, dev_file, example_nums=None, dev_ratio=0.1):
    count = 0
    examples = []
    idx = 0
    with open(cutword_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        total_num = len(lines)
        for line in tqdm(lines):
            if idx < 1000000:
                idx += 1
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            title_words = parts[0]
            content_words = parts[1]

            sample_idx = random.randint(0, total_num-1)
            negative_line = lines[sample_idx]
            neg_parts = negative_line.strip().split('\t')
            neg_title_words = neg_parts[0]
            neg_content_words = neg_parts[1]

            examples.append([title_words, content_words, 1])
            examples.append([title_words, neg_content_words, 0])
            #wfp.write("%s\t%s\t%d\n" % (title_words, content_words, 1))
            #wfp.write("%s\t%s\t%d\n" % (title_words, neg_content_words, 0))
            count += 2
            idx += 1
            if example_nums is not None and count >= example_nums:
                break

    random.shuffle(examples)
    total_num = len(examples)
    logging.info("total_num: %d" % (total_num))
    dev_num = int(total_num * dev_ratio)
    dev_examples = examples[0:dev_num]
    train_examples = examples[dev_num:]
    logging.info("train_num:%d, dev_num:%d" % (total_num-dev_num, dev_num))
    with open(train_file, "w", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter='\t')
        for example in train_examples:
            writer.writerow(example)
    with open(dev_file, "w", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for example in dev_examples:
            writer.writerow(example)


def preprocess_cutword_file():
    ori_file = "/search/odin/liruihong/article_data/random_201906_14"
    vocab_file = "/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
    output_file = "/search/odin/liruihong/word-based-transformer/data/cutword_article_201906_14"
    generate_cutword_file(ori_file, output_file, vocab_file)


def run_train_data_generation():
    cutword_file = "/search/odin/liruihong/word-based-transformer/data/cutword_article_20200701_15"
    vocab_file = "/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
    train_file = "/search/odin/liruihong/word-based-transformer/data/train_data_900k.tsv"
    dev_file = "/search/odin/liruihong/word-based-transformer/data/dev_data_100k.tsv"
    generate_train_data_from_articles(cutword_file, vocab_file, train_file, dev_file, example_nums=1000000, dev_ratio=0.1)

if __name__ == "__main__":
    #run_train_data_generation()
    preprocess_cutword_file()
