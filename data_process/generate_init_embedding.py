#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging 
import gensim
import numpy as np
import csv
from gensim import utils
logging.basicConfig(format='[%(levelname).1s %(asctime)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d_%H:%M:%S")


class MyCorpus(object):
    def __iter__(self):
        corpus_path = ""
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)



def train():
    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences, size=100, min_count=5, workers=16)
    model.build_vocab(corpus_file)
    model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

def load_vocab_file(vocab_file):
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as fp:
        for line in fp:
            word = line.strip().split('\t')[0]
            vocab.append(word)
    return vocab

def extract_vocab_embedding(embedding_file, vocab_file, vocab_embedding_file):
    
    vocab = load_vocab_file(vocab_file) 
    vocab_idx = {}
    for idx,w in enumerate(vocab):
        vocab_idx[w] = idx

    embedding_table = np.random.uniform(0.0,1.0,size=(len(vocab), 256))
    wfp = open(vocab_embedding_file, "w", encoding="utf-8") 
    writer = csv.writer(wfp, delimiter=' ')
    
    count = 0
    with open(embedding_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            parts = line.strip().split(' ')
            word = parts[0]
            vector = parts[1:]
            vector = np.asarray(vector, dtype=np.float32)
            if word in vocab_idx:
                count += 1
                embedding_table[vocab_idx[word]] = vector

    for idx, word in enumerate(vocab):
        vector = embedding_table[idx].tolist()
        writer.writerow([word] + vector)
    

if __name__ == "__main__":
    embedding_file = "/search/odin/liruihong/word2vec_embedding/trained_embedding/w2v_model_mincount1.txt"
    vocab_file = "/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
    vocab_embedding_file = "/search/odin/liruihong/word2vec_embedding/trained_embedding/final_vocab_embedding.txt"
    extract_vocab_embedding(embedding_file, vocab_file, vocab_embedding_file)

