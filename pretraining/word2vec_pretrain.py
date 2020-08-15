#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging
import gensim
from gensim.models import Word2Vec
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname).1s %(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d_%H:%M:%S")


def run_pretraining(input_file, model_file, w2v_file):
    sentences = gensim.models.word2vec.LineSentence(input_file)
    # sentences = gensim.models.word2vec.PathLineSentences(input_dir)
    model = Word2Vec(sentences=sentences, size=256, window=5, min_count=5, sg=1, workers=20, iter=10, compute_loss=True)
    model.wv.save_word2vec_format(w2v_file)
    model.save(model_file)

def run_continue_training(model_file, input_file, output_model, output_w2v):
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = Word2Vec.load(model_file)
    model.train(sentences=sentences, eopchs=8, compute_loss=True)
    model.wv.save_word2vec_format(output_w2v)
    model.save(output_model)


if __name__ == "__main__":
    input_file = "/search/odin/liruihong/word-based-transformer/data/cutword_article_20200701_15"
    #input_file = "/search/odin/liruihong/word-based-transformer/data/cutword_article"
    model_file = "/search/odin/liruihong/word-based-transformer/model_output/pretrained_word2vec/w2v_model.bin"
    w2v_file = "/search/odin/liruihong/word-based-transformer/model_output/pretrained_word2vec/w2v_model.txt"
    run_pretraining(input_file, model_file, w2v_file)

    #output_model = "/search/odin/liruihong/word-based-transformer/model_output/pretrained_word2vec/model_w2v.bin10"
    #output_w2v= "/search/odin/liruihong/word-based-transformer/model_output/pretrained_word2vec/model_w2v_format.10"
    #run_continue_training(model_file, input_file, output_model, output_w2v)
