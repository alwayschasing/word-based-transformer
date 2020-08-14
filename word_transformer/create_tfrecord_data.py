import logging
import tensorflow as tf
import os
import math
import numpy as np
import collections
import sys
from data_struct import SenpairProcessor
from input_function import file_based_convert_examples_to_features
from config import BaseConfig
import tokenization
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")


def create_tfrecord_data(input_file, output_file, vocab_file):
    processor = SenpairProcessor()
    tokenizer = tokenization.Tokenizer(vocab_file=vocab_file, stop_words_file=None, use_pos=False)
    data_examples = processor.get_train_examples(input_file)

    tfrecord_file = FLAGS.cached_tfrecord
    file_based_convert_examples_to_features(
        data_examples, FLAGS.max_seq_length, tokenizer, tfrecord_file)
    logging.info("  Num examples = %d", len(data_examples))


if __name__ == "__main__":
    input_file = ""
    output_file = ""
    vocab_file = ""
    create_tfrecord_data(input_file, output_file, vocab_file)


