import tensorflow as tf
import collections
import tokenization
from data_struct import InputFeatures


def convert_single_example(ex_index, example, max_seq_length, tokenizer, set_type="train", do_token=False):
    if do_token == True:
        tokens_a = tokenizer.tokenize(example.text_a)
    else:
        tokens_a = example.text_a.split(' ')
    tokens_b = None
    if example.text_b:
        if do_token == True:
            tokens_b = tokenizer.tokenize(example.text_b)
        else:
            tokens_b = example.text_b.split(' ')

    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:max_seq_length]

    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_mask_a = [1] * len(input_ids_a)

    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)

    if tokens_b is not None:
        if len(tokens_b) > max_seq_length:
            tokens_b = tokens_b[0:max_seq_length]

        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        input_mask_b = [1] * len(input_ids_b)
        while len(input_ids_b) < max_seq_length:
            input_ids_b.append(0)
            input_mask_b.append(0)

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("text_a:#%s#" % (example.text_a))
        tf.logging.info("tokens_a: #%s" % "#".join(tokens_a))
        tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
        if tokens_b is not None:
            tf.logging.info("text_b:#%s#" % (example.text_b))
            tf.logging.info("tokens_b: #%s" % "#".join(tokens_b))
            tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
            tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        tf.logging.info("label: %s" % (str(example.label)))
    
    if tokens_b is not None:
        feature = InputFeatures(
            input_ids_a=input_ids_a,
            input_mask_a=input_mask_a,
            input_ids_b=input_ids_b,
            input_mask_b=input_mask_b,
            label=example.label
        ) 
    else:
        feature = InputFeatures(
            input_ids_a=input_ids_a,
            input_mask_a=input_mask_a,
            label=example.label
        )
    return feature

def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer, output_file, 
                                            set_type="train", label_type="int", single_text=False, do_token=True):
    writer = tf.python_io.TFRecordWriter(output_file)
    error_count = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, set_type, do_token=do_token)
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


def file_based_input_fn_builder(input_file, seq_length, is_training, single_text=False):
    if single_text == False:
        name_to_features = {
            "input_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
            "input_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
            "labels": tf.FixedLenFeature([], tf.int64)
        }
    else:
        name_to_features = {
            "input_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
            "labels": tf.FixedLenFeature([], tf.int64)
        }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d
    return input_fn




