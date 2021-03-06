import logging
import tensorflow as tf
import os
import math
import numpy as np
import random
import collections
from models import BertModel
from models import BilstmAttnModel
from models import BilstmModel
from models.bert_transformer import get_shape_list
from models.bert_transformer import create_initializer
from models.bert_transformer import create_attention_mask_from_input_mask
from models.bert_transformer import optimization
from models.bert_transformer import get_assignment_map_from_checkpoint
from models.bert_transformer import get_activation
from models.bert_transformer import BertConfig
from data_struct import SenpairProcessor
from input_function import file_based_convert_examples_to_features
from input_function import file_based_input_fn_builder
from config import BaseConfig
import tokenization


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("NEG", 50, "neg num")
flags.DEFINE_string("model_name", "bert", "model name")
flags.DEFINE_string("config_file", None, "config_file path")
flags.DEFINE_string("output_dir", None, "output_dir path")
flags.DEFINE_string("vocab_file", None, "vocab_file path")
flags.DEFINE_string("stop_words_file", None, "stop_words_file path")
flags.DEFINE_string("embedding_table", None, "embedding_table path")
flags.DEFINE_bool("embedding_table_trainable", True, "embedding_table_trainable")
flags.DEFINE_string("input_file", None, "input_file path")
flags.DEFINE_string("cached_tfrecord", None, "cached tfrecord file path")
flags.DEFINE_string("gpu_id", "0", "gpu_id str")
flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")
flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length")
flags.DEFINE_integer("batch_size", 32, "Total batch size.")
flags.DEFINE_string("task_type", "classify", "task type name, classify or regression")

flags.DEFINE_bool("do_token", True, "Whether to do tokenize work.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run predict on the test set.")
flags.DEFINE_bool("single_text", False, "whether has single text of one instance")

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint")
flags.DEFINE_string("pred_model",None,"")
flags.DEFINE_string("eval_model",None,"")
flags.DEFINE_integer("save_checkpoints_steps", 1000,"")

flags.DEFINE_integer("num_train_epochs", 10, "num_train_steps")
flags.DEFINE_integer("num_warmup_steps", 10, "num_warmup_steps")


def word_attention_layer(input_tensor,
                         input_mask,
                         hidden_size,
                         query_act=None,
                         key_act=None,
                         value_act=None,
                         initializer_range=0.02,
                         pooling_strategy="mean"):
    """
    Args:
        input_tensor: Float Tensor of shape [batch_size, seq_length, hidden_size]
        input_mask: int Tensor of shape [batch_size, seq_length]
        hidden_size
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor,
                                   [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    shape_list = get_shape_list(input_tensor, expected_rank=[2, 3])
    batch_size = shape_list[0]
    seq_length = shape_list[1]

    query_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range)
    )

    #key_layer = tf.layers.dense(
    #    inputs=input_tensor,
    #    units=hidden_size,
    #    activation=key_act,
    #    name="key",
    #    kernel_initializer=create_initializer(initializer_range)
    #)

    #value_layer = tf.layers.dense(
    #    inputs=input_tensor,
    #    units=hidden_size,
    #    activation=value_act,
    #    name="value",
    #    kernel_initializer=create_initializer(initializer_range)
    #)
    value_layer = input_tensor

    query_shape_list = get_shape_list(query_layer, expected_rank=3)
    tf.logging.debug("query_layer shape: %s" % (str(query_shape_list)))
    # query shape [batch_size, seq_length, hidden_size]
    #attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.layers.dense(
        query_layer,
        units=1,
        activation=None,
        name="attn_weights",
        kernel_initializer=create_initializer(initializer_range)
    )
    weights = tf.get_variable(name="attn_weights", shape=[hidden_size], initializer=tf.random_normal_initializer(stddev=0.1))
    attention_scores = tf.tensordot(query_layer, weights, axes=1)
    # attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(hidden_size)))
    # attention_mask shape: [batch_size, seq_length, seq_length]
    #attention_mask = create_attention_mask_from_input_mask(input_tensor, input_mask)
    # expand for multi heads, [batch_size, 1, seq_length, seq_length]
    #attention_mask = tf.expand_dims(attention_mask, axis=[1])
    attention_mask = input_mask
    mask_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    # attention_score: [batch_size, num_heads, seq_length, seq_length]
    attention_scores += mask_adder

    # attention_probs shape: [batch_size, seq_length]
    attention_probs = tf.nn.softmax(attention_scores)
    # value_layer = tf.reshape(value_layer, [batch_size, seq_length, hidden_size])
    # value_layer shape : [batch_size, num_heads, seq_length, size_per_head]
    # value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.reduce_mean(tf.math.multiply(tf.reshape(attention_probs, [batch_size, seq_length, 1]), value_layer), axis=1)
    # context_layer shape : [batch_size, seq_length, hidden_size]
    # context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    # context_layer = tf.reshape(context_layer, [batch_size, seq_length, hidden_size])

    #if pooling_strategy == "mean":
    # input_mask_expanded = tf.cast(input_mask.expand(), float)
    #float_mask = tf.cast(input_mask, tf.float32)
    #float_mask = tf.reshape(float_mask, [batch_size, seq_length, 1])
    #tf.logging.debug("context_layer shape:%s" % (get_shape_list(context_layer)))
    #tf.logging.debug("float_mask shape:%s" % (get_shape_list(float_mask)))
    #pooling_output = tf.math.multiply(context_layer, float_mask)
    #tf.logging.debug("[check shape1 shape:%s" % (get_shape_list(pooling_output)))
    #pooling_output = tf.reduce_sum(context_layer, axis=1)
    #tf.logging.debug("[check shape2 shape:%s" % (get_shape_list(pooling_output)))
    #divisor = tf.reduce_sum(float_mask, axis=1)
    #tf.logging.debug("[check shape3 shape:%s" % (get_shape_list(divisor)))
    #pooling_output = pooling_output/divisor
    #tf.logging.debug("[check shape4 shape:%s" % (get_shape_list(pooling_output)))
    #tf.logging.debug("[check attention_probs shape:%s" % (get_shape_list(attention_probs)))
    tf.logging.debug("[check context_layer shape:%s" % (get_shape_list(context_layer)))
    tf.logging.debug("[check attention_probs shape:%s" % (get_shape_list(attention_probs)))
    return context_layer, attention_probs


def create_kwextraction_model(config,
                              input_ids,
                              input_mask,
                              is_training=None,
                              embedding_table=None,
                              hidden_size=None,
                              query_act=None,
                              key_act=None,
                              scope="title_kw",
                              model_name="bert"):
    with tf.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        if model_name == "bert":
            encode_model = BertModel(config=config,
                                    is_training=is_training,
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    embedding_table=embedding_table,
                                    use_one_hot_embeddings=False)

            sequence_output = encode_model.get_sequence_output()
            # attention_probs : [batch_size, seq_length]
            pooling_output, keyword_probs = word_attention_layer(input_tensor=sequence_output,
                                                                input_mask=input_mask,
                                                                hidden_size=hidden_size,
                                                                query_act=query_act,
                                                                key_act=key_act)
        elif model_name == "bilstm":
            encode_model = BilstmAttnModel(config=config,
                                          is_training=is_training,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          embedding_table=embedding_table)
            sequence_output = encode_model.get_sequence_output()
            pooling_output =  tf.reduce_mean(sequence_output, axis=-1)
            keyword_probs = encode_model.get_keyword_probs()
        elif model_name == "test":
            encode_model = BilstmModel(config,
                                       is_training=is_training,
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       embedding_table=embedding_table)
            pooling_output = encode_model.get_text_encoding()
            keyword_probs = None
        else:
            raise ValueError("model type error")

        return pooling_output, keyword_probs


def model_fn_builder(config,
                     learning_rate=1e-5,
                     task="classify",
                     single_text=False,
                     init_checkpoint=None,
                     num_train_steps=None,
                     num_warmup_steps=None,
                     embedding_table_value=None,
                     embedding_table_trainable=False,
                     classify_num=2,
                     model_name="bert"):
    """
    :param task: "classify" or "regression"
    :return:
    """
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        
        guid = features["guid"]
        input_ids_a = features["input_ids_a"]
        input_mask_a = features["input_mask_a"]
        #if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if single_text == False:
            input_ids_b = features["input_ids_b"]
            input_mask_b = features["input_mask_b"]
            labels = features["labels"]

        with tf.variable_scope("embedding_table", reuse=tf.compat.v1.AUTO_REUSE):
            embedding_table = tf.get_variable("embedding_table",
                                            shape=[config.vocab_size, config.embedding_size],
                                            trainable=embedding_table_trainable)

        def init_embedding_table(scoffold, sess):
            sess.run(embedding_table.initializer, {embedding_table.initial_value: embedding_table_value})

        scaffold = None
        if embedding_table_value is not None:
            scaffold = tf.train.Scaffold(init_fn=init_embedding_table)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        sen_a_output, sen_a_attention_probs = create_kwextraction_model(input_ids=input_ids_a,
                                                                        input_mask=input_mask_a,
                                                                        config=config,
                                                                        is_training=is_training,
                                                                        embedding_table=embedding_table,
                                                                        hidden_size=config.hidden_size,
                                                                        query_act=get_activation(config.query_act),
                                                                        key_act=config.key_act,
                                                                        scope="text_kw",
                                                                        model_name=model_name)

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            sen_b_output, sen_b_attention_probs = create_kwextraction_model(input_ids=input_ids_b,
                                                                            input_mask=input_mask_b,
                                                                            config=config,
                                                                            is_training=is_training,
                                                                            embedding_table=embedding_table,
                                                                            hidden_size=config.hidden_size,
                                                                            query_act=get_activation(config.query_act),
                                                                            key_act=config.key_act,
                                                                            scope="text_kw",
                                                                            model_name=model_name)

            total_loss = None
            with tf.variable_scope("sen_pair_loss"):
                query_encoder = sen_a_output
                doc_encoder = sen_b_output
                tmp = tf.tile(doc_encoder, [1, 1])
                doc_encoder_fd = sen_a_output
                for i in range(FLAGS.NEG):
                    rand = random.randint(1, FLAGS.batch_size + i) % FLAGS.batch_size
                    s1 = tf.slice(tmp, [rand, 0], [FLAGS.batch_size - rand, -1])
                    s2 = tf.slice(tmp, [0, 0], [rand, -1])
                    doc_encoder_fd = tf.concat([doc_encoder_fd, s1, s2], axis=0)
                query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_encoder), axis=1, keepdims=True)), [FLAGS.NEG + 1, 1])
                doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_encoder_fd), axis=1, keepdims=True))
                query_encoder_fd = tf.tile(query_encoder, [FLAGS.NEG + 1, 1])
                prod = tf.reduce_sum(tf.multiply(query_encoder_fd, doc_encoder_fd), axis=1, keepdims=True)
                norm_prod = tf.multiply(query_norm, doc_norm)
                cos_sim_raw = tf.truediv(prod, norm_prod)
                cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.NEG + 1, -1])) * 20
                
                prob = tf.nn.softmax(cos_sim)
                hit_prob = tf.slice(prob, [0, 0], [-1, 1])
                loss = -tf.reduce_mean(tf.log(hit_prob))
                correct_prediction = tf.cast(tf.equal(tf.argmax(prob, 1), 0), tf.float32)
                accuracy = tf.reduce_mean(correct_prediction)
                total_loss = loss

                if FLAGS.do_eval:
                    sen_a_norm = tf.sqrt(tf.reduce_sum(tf.square(sen_a_output), axis=1))
                    sen_b_norm = tf.sqrt(tf.reduce_sum(tf.square(sen_b_output), axis=1))
                    dotres = tf.reduce_sum(tf.multiply(sen_a_output, sen_b_output), axis=1)
                    tf.logging.debug("dot shape:%s" % (get_shape_list(dotres)))
                    tf.logging.debug("norm_a shape:%s" % (get_shape_list(sen_a_norm)))
                    tf.logging.debug("norm_b shape:%s" % (get_shape_list(sen_b_norm)))
                    normres = tf.multiply(sen_a_norm, sen_b_norm)
                    tf.logging.debug("norm shape:%s" % (get_shape_list(normres)))
                    cosine_val = tf.truediv(dotres, normres)
                    predictions = tf.sigmoid(cosine_val)
                    tf.logging.debug("cosine_val shape:%s" % (get_shape_list(cosine_val)))
                    #probabilities = tf.nn.softmax(cos_sim)
                    #predictions = tf.argmax(probabilities, axis=1)
                #if task == "classify":
                #    with tf.variable_scope("classify_loss"):
                #        concat_vector = tf.concat([sen_a_output, sen_b_output, tf.abs(sen_a_output-sen_b_output)], axis=1)
                #        logits = tf.layers.dense(concat_vector, classify_num, kernel_initializer=None)
                #        probabilities = tf.nn.softmax(logits)
                #        labels = tf.cast(labels, tf.int32)
                #        softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=config.label_nums),
                #                                                   logits=logits)
                #        total_loss = softmax_loss
                #elif task == "regression":
                #    with tf.variable_scope("regression_loss"):
                #        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)
                #        cosine_val = cosine_similarity(sen_a_output, sen_b_output)
                #        cosine_val = tf.reshape(cosine_val, [-1])
                #        labels = tf.cast(labels, tf.float32)
                #        total_loss = tf.losses.mean_squared_error(labels, cosine_val)
                #else:
                #    raise ValueError("task name error")

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            tf.logging.info("get assignment_map from checkpoint")
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimization.create_optimizer(
                #total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            #log_hook = tf.train.LoggingTensorHook({"total_loss":  total_loss, 
            #                                       "guid": guid,
            #                                       "input_ids_a": input_ids_a, 
            #                                       "input_mask_a": input_mask_a,
            #                                       "input_mask_num": tf.reduce_sum(input_mask_a, axis=-1),
            #                                       "input_ids_b": input_ids_b,
            #                                       "input_mask_b": input_mask_b,
            #                                       "input_mask_num": tf.reduce_sum(input_mask_b, axis=-1)}, every_n_iter=10)
            log_hook = tf.train.LoggingTensorHook({"total_loss":  total_loss, "accuracy": accuracy}, every_n_iter=10)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op,
                                                     training_hooks=[log_hook],
                                                     scaffold=scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            #if task == "classify":
            #    predictions = tf.argmax(probabilities, axis=1)
            #    output_spec = tf.estimator.EstimatorSpec(mode=mode,
            #                                            loss=total_loss,
            #                                            eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)},
            #                                            scaffold=scaffold)
            #elif task == "regression":
            #    predictions = cosine_val
            #    output_spec = tf.estimator.EstimatorSpec(mode=mode,
            #                                            loss=total_loss,
            #                                            eval_metric_ops={"auc": tf.metrics.auc(labels, cosine_val)},
            #                                            scaffold=scaffold)
            log_hook = tf.train.LoggingTensorHook({"total_loss":  total_loss, "accuracy": accuracy}, every_n_iter=1)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     evaluation_hooks=[log_hook],
                                                     scaffold=scaffold)
        else:
            if single_text:
                log_hook = tf.train.LoggingTensorHook({"guid": guid,
                                                   "sen_a_embedding": sen_a_output,
                                                   "keyword_probs_a": sen_a_attention_probs}, every_n_iter=1)
                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                        predictions={"sen_a_embedding": sen_a_output,
                                                                    "input_ids_a": input_ids_a,
                                                                    "keyword_probs_a": sen_a_attention_probs})
            else:
                log_hook = tf.train.LoggingTensorHook({"guid": guid,
                                                   "sen_a_embedding": sen_a_output,
                                                   "sen_a_attentions": sen_a_attention_probs,
                                                   "sen_b_embedding": sen_b_output,
                                                   "sen_b_attentions": sen_b_attention_probs}, every_n_iter=1)
                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                         predictions={"sen_a_embedding": sen_a_output,
                                                                      "input_ids_a": input_ids_a,
                                                                      "keyword_probs_a": sen_a_attention_probs,
                                                                      "sen_b_embedding": sen_b_output,
                                                                      "input_ids_b": input_ids_b,
                                                                      "keyword_probs_b": sen_b_attention_probs},
                                                         prediction_hooks=[log_hook])
        return output_spec
    return model_fn


def load_vocab_file(vocab_file):
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as fp:
        for line in fp:
            word = line.strip().split('\t')
            vocab.append(word)
    return vocab


def load_embedding_table(embedding_file, vocab_file):
    vectors = []
    vocab = load_vocab_file(vocab_file)
    with open(embedding_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split(' ')
            word = parts[0]
            vec = parts[1:]
            vectors.append(vec)

    embedding_table = np.asarray(vectors, dtype=np.float32)
    tf.logging.info("load embedding_table, shape is %s"%(str(embedding_table.shape)))
    assert len(vocab) == len(vectors)
    return embedding_table


def main(_):
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    processor = SenpairProcessor()

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    config = BaseConfig.from_json_file(FLAGS.config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.Tokenizer(vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file, use_pos=False)

    run_config = None
    num_train_steps = 0
    num_warmup_steps = 0
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.input_file)
        num_train_steps = int(
            len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = FLAGS.num_warmup_steps

        run_config = tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=num_train_steps/FLAGS.num_train_epochs,
            keep_checkpoint_max=5,
        )

    embedding_table = None
    if FLAGS.embedding_table is not None:
        embedding_table = load_embedding_table(FLAGS.embedding_table, FLAGS.vocab_file)

    model_fn = model_fn_builder(config=config,
                                learning_rate=FLAGS.learning_rate,
                                task=FLAGS.task_type,
                                single_text=FLAGS.single_text,
                                init_checkpoint=FLAGS.init_checkpoint,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                embedding_table_value=embedding_table,
                                embedding_table_trainable=False,
                                model_name=FLAGS.model_name)


    params = {"batch_size":FLAGS.batch_size}
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.output_dir,
                                       config=run_config,
                                       params=params)


    if FLAGS.do_train:
        if FLAGS.cached_tfrecord:
            train_file = FLAGS.cached_tfrecord
        else:
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, FLAGS.max_seq_length, tokenizer, train_file, do_token=FLAGS.do_token)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples  # 释放train_examples内存
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
        )

        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps)
    elif FLAGS.do_eval:
        dev_examples = processor.get_train_examples(FLAGS.input_file)
        if FLAGS.cached_tfrecord:
            dev_file = FLAGS.cached_tfrecord
        else:
            dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
        if not os.path.exists(dev_file):
            file_based_convert_examples_to_features(
                dev_examples, FLAGS.max_seq_length, tokenizer, dev_file)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(dev_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        del dev_examples
        eval_input_fn = file_based_input_fn_builder(
            input_file=dev_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False
        )

        if FLAGS.eval_model is not None:
            eval_model_path = os.path.join(FLAGS.output_dir, FLAGS.eval_model)
        else:
            eval_model_path = None

        result = estimator.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=eval_model_path
        )
        eval_output_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(eval_output_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    else:
        predict_examples = processor.get_test_examples(FLAGS.input_file)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, FLAGS.max_seq_length, tokenizer, predict_file,
                                                set_type="test", label_type="int", single_text=FLAGS.single_text)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            single_text=FLAGS.single_text
        )

        if FLAGS.pred_model is not None:
            pred_model_path = os.path.join(FLAGS.output_dir, FLAGS.pred_model)
        else:
            pred_model_path = None

        result = estimator.predict(
            input_fn=predict_input_fn,
            checkpoint_path=pred_model_path
        )

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                sen_a_embedding = prediction["sen_a_embedding"]
                input_ids_a = prediction["input_ids_a"]
                keyword_probs_a = prediction["keyword_probs_a"]
                if not FLAGS.single_text:
                    sen_b_embedding = prediction["sen_b_embedding"]
                    input_ids_b = prediction["input_ids_b"]
                    keyword_probs_b = prediction["keyword_probs_b"]
                    
                sorted_keyword_idx_a = np.argsort(-keyword_probs_a)
                extracted_keywords_a = []
                for idx in sorted_keyword_idx_a:
                    word_id = input_ids_a[idx]
                    word_prob = keyword_probs_a[idx]
                    word = tokenizer.convert_ids_to_tokens([word_id])[0]
                    extracted_keywords_a.append([word, word_prob])
                
                keyword_output_a = " ".join(["%s:%f" % (kw, prob) for kw, prob in extracted_keywords_a])
                text_output_a = " ".join(tokenizer.convert_ids_to_tokens(input_ids_a))

                writer.write("%s\t%s" % (keyword_output_a, text_output_a))
                writer.write("\n")

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO, format="[%(asctime)s-%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("do_token")
    tf.app.run()


