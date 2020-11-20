import logging
import tensorflow as tf
import os
import math
import numpy as np
import collections
import sys
from data_struct import SenpairProcessor
from input_function import file_based_convert_examples_to_features
import tokenization
import logging
import argparse
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")


def load_vocab(vocab_file):
    vocab = set()
    with open(vocab_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            word = parts[0]
            vocab.add(word)
    return vocab

def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer, output_file, set_type="train", label_type="int", single_text=False):
    writer = tf.python_io.TFRecordWriter(output_file)
    error_count = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, set_type)
        if "" == feature:
            error_count += 1
            continue

        def create_int_feature(values):
            feat = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feat

        def create_float_feature(values):
            feat = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return feat

        features = collections.OrderedDict()
        features["input_ids_a"] = create_int_feature(feature.input_ids_a)
        features["input_mask_a"] = create_int_feature(feature.input_mask_a)

        if single_text == False:
            features["input_ids_b"] = create_int_feature(feature.input_ids_b)
            features["input_mask_b"] = create_int_feature(feature.input_mask_b)

        if label_type == "int":
            features["labels"] = create_int_feature([int(feature.label)])
        else:
            features["labels"] = create_float_feature([float(feature.label)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    tf.logging.info("based_convert_error_case:%d" % (error_count))
    writer.close()


def create_masked_lm_predictions(tokens,masked_lm_prob,max_predictions_per_seq,vocab):
    cand_indexes = []
    for (i,token) in enumerate(tokens):
        if token == "[unknown]" or token not in vocab:
            continue
        cand_indexes.append(i)
    random.shuffle(cand_indexes)

    output_tokens = list(tokens)
    masked_lm = collections.namedtuple("masked_lm", ["index", "label"])
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None
        if random.random() > 0.8:
            masked_token = "[mask]"
        else:
            masked_token = tokens[index]
        output_tokens[index] = masked_token
        masked_lms.append(masked_lm(index=index, label=tokens[index]))
    
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_labels, masked_lm_positions)
    
def create_training_instance(ins_idx, tokens_a, tokens_b, label, max_seq_length, 
                             masked_lm_prob, max_predictions_per_seq, tokenizer, vocab):
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:max_seq_length]
    (tokens_a, masked_lm_labels_a, masked_lm_positions_a) = create_masked_lm_predictions(tokens_a,
                                                                                         masked_lm_prob,
                                                                                         max_predictions_per_seq,
                                                                                         vocab)
    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_mask_a = [1] * len(input_ids_a)
    masked_lm_ids_a = tokenizer.convert_tokens_to_ids(masked_lm_labels_a)
    masked_lm_weights_a = [1.0] * len(masked_lm_ids_a)
    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)
    
    while len(masked_lm_positions_a) < max_predictions_per_seq:
        masked_lm_positions_a.append(0)
        masked_lm_ids_a.append(0)
        masked_lm_weights_a.append(0)

    if len(tokens_b) > max_seq_length:
        tokens_b = tokens_b[0:max_seq_length]
    (tokens_b, masked_lm_labels_b, masked_lm_positions_b) = create_masked_lm_predictions(tokens_b,
                                                                                         masked_lm_prob,
                                                                                         max_predictions_per_seq,
                                                                                         vocab)
    input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
    input_mask_b = [1] * len(input_ids_b)
    masked_lm_ids_b = tokenizer.convert_tokens_to_ids(masked_lm_labels_b)
    masked_lm_weights_b = [1.0] * len(masked_lm_ids_b)
    while len(input_ids_b) < max_seq_length:
        input_ids_b.append(0)
        input_mask_b.append(0)

    while len(masked_lm_positions_b) < max_predictions_per_seq:
        masked_lm_positions_b.append(0)
        masked_lm_ids_b.append(0)
        masked_lm_weights_b.append(0)

    def create_int_feature(values):
        feat = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feat

    def create_float_feature(values):
        feat = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feat

    features = collections.OrderedDict()
    features["input_ids_a"] = create_int_feature(input_ids_a)
    features["input_mask_a"] = create_int_feature(input_mask_a)
    features["masked_lm_ids_a"] = create_int_feature(masked_lm_ids_a)
    features["masked_lm_positions_a"] = create_int_feature(masked_lm_positions_a)
    features["masked_lm_weights_a"] = create_float_feature(masked_lm_weights_a)

    features["input_ids_b"] = create_int_feature(input_ids_b)
    features["input_mask_b"] = create_int_feature(input_mask_b)
    features["masked_lm_ids_b"] = create_int_feature(masked_lm_ids_b)
    features["masked_lm_positions_b"] = create_int_feature(masked_lm_positions_b)
    features["masked_lm_weights_b"] = create_float_feature(masked_lm_weights_b)
    features["labels"] = create_int_feature([label])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    if ins_idx < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens_a: %s" % "#".join(tokens_a))
        tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        tf.logging.info("masked_lm_ids_a: %s" % " ".join([str(x) for x in masked_lm_ids_a]))
        tf.logging.info("masked_lm_positions_a: %s" % " ".join([str(x) for x in masked_lm_positions_a]))
        tf.logging.info("masked_lm_weights_a: %s" % " ".join([str(x) for x in masked_lm_weights_a]))
        tf.logging.info("tokens_b: %s" % "#".join(tokens_b))
        tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
        tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        tf.logging.info("masked_lm_ids_b: %s" % " ".join([str(x) for x in masked_lm_ids_b]))
        tf.logging.info("masked_lm_positions_b: %s" % " ".join([str(x) for x in masked_lm_positions_b]))
        tf.logging.info("masked_lm_weights_b: %s" % " ".join([str(x) for x in masked_lm_weights_b]))
        tf.logging.info("label: %s" % (str(label)))
    return tf_example

def create_pretraining_data_from_file(input_file, 
                                      output_file, 
                                      vocab_file, 
                                      max_seq_length,
                                      masked_lm_prob, 
                                      max_predictions_per_seq, 
                                      do_token=False):
    tokenizer = tokenization.Tokenizer(vocab_file=vocab_file, stop_words_file=None, use_pos=False)
    vocab = load_vocab(vocab_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    with open(input_file, 'r', encoding="utf-8") as fp:
        logging.info("create training data from %s" % (input_file))
        lines = fp.readlines()
        for idx, line in enumerate(tqdm(lines)):
            parts = line.strip().split('\t')
            text_a = parts[0]
            text_b = parts[1]
            # label = int(parts[2])
            label = 1
            if do_token == 1:
                tokens_a = tokenizer.tokenize(text_a)
                tokens_b = tokenizer.tokenize(text_b)
            else:
                tokens_a = text_a.split(' ')
                tokens_b = text_b.split(' ')
            instance = create_training_instance(idx, tokens_a, tokens_b, label, max_seq_length, 
                                                masked_lm_prob, max_predictions_per_seq, tokenizer, vocab)
            writer.write(instance.SerializeToString())
    writer.close() 


def create_tfrecord_data(input_file, output_file, vocab_file):
    processor = SenpairProcessor()
    tokenizer = tokenization.Tokenizer(vocab_file=vocab_file, stop_words_file=None, use_pos=False)
    data_examples = processor.get_train_examples(input_file)

    tfrecord_file = FLAGS.cached_tfrecord
    file_based_convert_examples_to_features(
        data_examples, FLAGS.max_seq_length, tokenizer, tfrecord_file)
    logging.info("  Num examples = %d", len(data_examples))


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    vocab_file = args.vocab_file
    max_seq_length = args.max_seq_length
    masked_lm_prob = args.masked_lm_prob
    max_prediction_per_seq = args.max_predictions_per_seq
    do_token = args.do_token
    create_pretraining_data_from_file(input_file, output_file, vocab_file, max_seq_length, masked_lm_prob, max_prediction_per_seq, do_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create_data")
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--max_seq_length', default=256, type=int, required=True)
    parser.add_argument('--masked_lm_prob', default=256, type=float, required=True)
    parser.add_argument('--max_predictions_per_seq', default=256, type=int, required=True)
    parser.add_argument('--do_token', type=int, default=0)
    args = parser.parse_args()
    main(args)


