#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging 
import gensim
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

