import logging
import tensorflow as tf
import os
import math
import numpy as np
import time
import random
import collections
from models import BertModel
from models import BilstmAttnModel
from models.bert_transformer import get_shape_list
from models.bert_transformer import create_initializer
from models.bert_transformer import optimization
from models.bert_transformer import get_assignment_map_from_checkpoint
from models.bert_transformer import get_activation
from input_function import pretrain_input_fn_builder
from config import BaseConfig
import tokenization
from models import layer_norm


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
flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length")
flags.DEFINE_integer("mask_num", 25, "The maximum mask num")
flags.DEFINE_integer("batch_size", 32, "Total batch size.")
flags.DEFINE_string("task_type", "classify", "task type name, classify or regression")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run predict on the test set.")

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint")
flags.DEFINE_string("pred_model",None,"")
flags.DEFINE_string("eval_model",None,"")
flags.DEFINE_integer("save_checkpoints_steps", 1000,"")

flags.DEFINE_integer("num_train_epochs", 10, "num_train_steps")
flags.DEFINE_integer("num_warmup_steps", 10, "num_warmup_steps")
flags.DEFINE_integer("num_train_steps", 10, "num_train_steps")



def gather_indexes(sequence_tensor,positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = tf.cast(sequence_shape[0], dtype=tf.int64)
    seq_length = sequence_shape[1]
    hidden_size = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0,batch_size, dtype=tf.int64) * seq_length, [-1,1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, hidden_size])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


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

    shape_list = get_shape_list(input_tensor, expected_rank=[2, 3])
    batch_size = shape_list[0]
    seq_length = shape_list[1]
    tf.logging.debug("[check_wordattn] %s" % (str(shape_list)))

    query_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range)
    )
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
    #weights = tf.get_variable(name="attn_weights", shape=[hidden_size], initializer=tf.random_normal_initializer(stddev=0.1))
    #attention_scores = tf.tensordot(query_layer, weights, axes=1)
    attention_scores = tf.reshape(attention_scores, [batch_size, seq_length])
    attention_mask = input_mask
    mask_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    # attention_score: [batch_size, num_heads, seq_length, seq_length]
    attention_scores += mask_adder
    # attention_probs shape: [batch_size, seq_length]
    attention_probs = tf.nn.softmax(attention_scores)
    context_layer = tf.reduce_sum(tf.math.multiply(tf.reshape(attention_probs, [batch_size, seq_length, 1]), value_layer), axis=1)
    context_layer = context_layer/tf.reduce_sum(tf.cast(input_mask, dtype=tf.float32), axis=-1, keepdims=True)
    return context_layer, attention_probs

def get_pred_label_output(config,input_tensor,label=None,is_training=False):
    with tf.variable_scope("pred_label"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[config.label_size, config.hidden_size],
            initializer=create_initializer(config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias",
            shape=[config.label_size],
            initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        label_probs = tf.nn.softmax(logits, axis=-1)
        if is_training:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            label = tf.reshape(label, [-1])
            one_hot_label = tf.one_hot(label, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_label * log_probs, axis = -1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, per_example_loss, label_probs)
        else:
            return label_probs

def build_negative_sample_weights(input_tensor,label_ids,weights,sample_scope,neg_num,rng,num_heads=None):
    input_shape = get_shape_list(input_tensor)
    tmp_batch_size = input_shape[0]
    neg_sample_ids = tf.random.uniform(shape=[tmp_batch_size,neg_num],
                                       minval=0,
                                       maxval=sample_scope-1,
                                       dtype=tf.int64)

    all_ids = tf.concat([label_ids,neg_sample_ids],axis=1)
    logits_weights = tf.gather(weights, all_ids)
    if num_heads:
        logits_weights = tf.tile(logits_weights,[1,1,num_heads])
    return logits_weights


def get_masked_lm_output(config, input_tensor, word_weights, positions,
                         label_ids, label_weights, rng):
    batch_size = tf.shape(input_tensor)[0]
    position_size = tf.shape(label_weights)[1]

    input_tensor = gather_indexes(input_tensor, positions)
    tf.logging.debug("[check2]input_tensor:%s" % (str(input_tensor.shape.as_list())))
    with tf.variable_scope("lm_predictions"):
        with tf.variable_scope("lm_out_layer"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=config.hidden_size,
                activation=get_activation("tanh"),
                kernel_initializer=create_initializer(0.1)
            )
            input_tensor = layer_norm(input_tensor)

            word_bias = tf.get_variable(
                "lm_output_bias",
                shape=[config.vocab_size],
                initializer=tf.zeros_initializer())

            neg_sample_num = config.neg_sample_num
            # reshape label_ids from [batch_size,position_size] to [batch_size*position_size]
            label_ids = tf.reshape(label_ids, [-1, 1])
            # negative shape [batch_size*position_size, 1 + neg_sample_num, hidden_size]
            negative_sample_weights = build_negative_sample_weights(input_tensor, label_ids, word_weights, config.vocab_size, neg_sample_num, rng)
            negative_sample_bias = build_negative_sample_weights(input_tensor,label_ids, word_bias,config.vocab_size,neg_sample_num,rng)

            input_tensor = tf.expand_dims(input_tensor, axis=1)
            # change input_tensor shape to: [batch_size*position_size,1,hidden_size],
            # in order to get logits as shape [batch_size*position_size, 1, 1 + neg_sample_num]

            # reshape negative_sample_weights, res: [batch_size*position_size, hidden_size, 1 + neg_sample_num]
            negative_sample_weights = tf.transpose(negative_sample_weights, perm=[0, 2, 1])

            # input_tensor:[batch_size*position_size , 1, hidden_size]
            # negative_sample_weights:[batch_size*position_size, hidden_size, 1 + neg_sample_num]
            # logits: [batch_size, position_size, 1 + neg_sample_num]
            tf.logging.debug("[check1]input_tensor:%s, negative_sample_weights:%s" % 
                             (str(input_tensor.shape.as_list()), str(negative_sample_weights.shape.as_list())))
            logits = tf.matmul(input_tensor, negative_sample_weights)
            # reshape to [batch_size*position_size, 1 + neg_sample_ids]
            logits = tf.reshape(logits, [-1,1+neg_sample_num])
            tf.logging.debug("logits shape:%s",str(logits.shape.as_list()))
            tf.logging.debug("negative_smaple_bias:%s", str(negative_sample_bias.shape.as_list()))
            logits = tf.add(logits, negative_sample_bias)

            neg_label = tf.zeros(
                shape=[batch_size*position_size, neg_sample_num],
                dtype=tf.float32,
                name="neg_label")
            pos_label = tf.ones(
                shape=[batch_size*position_size, 1],
                dtype=tf.float32,
                name="pos_label")

            one_hot_labels = tf.concat([pos_label, neg_label], axis=1)


            #per_example_loss shape: [bactch_size*position_size,1 + neg_sample_ids]
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels,
                                                                       logits=logits,
                                                                       name="pred_lm_loss")
            tf.logging.debug("per_example_loss shape:%s"%(per_example_loss.shape.as_list()))

            per_example_loss = tf.reduce_sum(tf.reshape(per_example_loss, shape=[batch_size,position_size,1 + neg_sample_num]), axis=[-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real predictions and 0.0 for the
            # padding predictions.
            # per_example_loss shape: [batch_size*positions,1 + neg_sample_num]
            per_example_loss = tf.reduce_sum(per_example_loss * label_weights)
            numerator = tf.reduce_sum(per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

    return (loss, per_example_loss)



def create_pretraining_model(config,
                              input_ids,
                              input_mask,
                              is_training=None,
                              embedding_table=None,
                              hidden_size=None,
                              query_act=None,
                              key_act=None,
                              scope="text_kw",
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
            # pooling_output =  tf.reduce_mean(sequence_output, axis=-1)

        elif model_name == "bilstm":
            encode_model = BilstmAttnModel(config=config,
                                          is_training=is_training,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          embedding_table=embedding_table)
            sequence_output = encode_model.get_sequence_output()
            pooling_output =  tf.reduce_mean(sequence_output, axis=-1)
            keyword_probs = encode_model.get_keyword_probs()
        else:
            raise ValueError("model type error")
        return sequence_output, pooling_output, keyword_probs


def model_fn_builder(config,
                     learning_rate=1e-5,
                     task="classify",
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

        input_ids_a = features["input_ids_a"]
        input_mask_a = features["input_mask_a"]
        masked_lm_ids_a = features["masked_lm_ids_a"]
        masked_lm_positions_a = features["masked_lm_positions_a"]
        masked_lm_weights_a = features["masked_lm_weights_a"]

        input_ids_b = features["input_ids_b"]
        input_mask_b = features["input_mask_b"]
        masked_lm_ids_b = features["masked_lm_ids_b"]
        masked_lm_positions_b = features["masked_lm_positions_b"]
        masked_lm_weights_b = features["masked_lm_weights_b"]
        labels = features["labels"]

        embedding_table = tf.get_variable("embedding_table",
                                          shape=[config.vocab_size, config.embedding_size],
                                          trainable=embedding_table_trainable)
        embedding_weights = tf.get_variable("embedding_weights",
                                            shape=[config.vocab_size, config.hidden_size],
                                            trainable=True)

        def init_embedding_table(scoffold, sess):
            sess.run(embedding_table.initializer, {embedding_table.initial_value: embedding_table_value})

        scaffold = None
        if embedding_table_value is not None:
            scaffold = tf.train.Scaffold(init_fn=init_embedding_table)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        seq_a_output, sen_a_output, sen_a_attention_probs = create_pretraining_model(input_ids=input_ids_a,
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
            seq_b_output, sen_b_output, sen_b_attention_probs = create_pretraining_model(input_ids=input_ids_b,
                                                                           input_mask=input_mask_b,
                                                                           config=config,
                                                                           is_training=is_training,
                                                                           embedding_table=embedding_table,
                                                                           hidden_size=config.hidden_size,
                                                                           query_act=get_activation(config.query_act),
                                                                           key_act=config.key_act,
                                                                           scope="text_kw",
                                                                           model_name=model_name)

            total_loss = 0
            with tf.variable_scope("mask_lm_loss", reuse=tf.compat.v1.AUTO_REUSE):
                rng = random.Random(time.time())
                (masked_lm_loss_a, masked_lm_example_loss_a) = get_masked_lm_output(
                    config, seq_a_output, embedding_weights, masked_lm_positions_a,
                    masked_lm_ids_a, masked_lm_weights_a, rng)

                (masked_lm_loss_b, masked_lm_example_loss_b) = get_masked_lm_output(
                    config, seq_b_output, embedding_weights, masked_lm_positions_b,
                    masked_lm_ids_b, masked_lm_weights_b, rng)

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
                sim_loss = loss
                #if task == "classify":
                #    with tf.variable_scope("classify_loss"):
                #        concat_vector = tf.concat([sen_a_output, sen_b_output, tf.abs(sen_a_output-sen_b_output)], axis=1)
                #        fc_output = tf.layers.dense(concat_vector, config.hidden_size, activation=tf.tanh, kernel_initializer=create_initializer(0.1))
                #        #logits = tf.layers.dense(fc_output, classify_num, activation=tf.tanh, kernel_initializer=None)
                #        cls_weights = tf.get_variable("cls_weights", 
                #                                      shape=[config.hidden_size],
                #                                      initializer=create_initializer(0.1))
                #        logits = tf.tensordot(fc_output, cls_weights, axes=1)
                #        #probabilities = tf.nn.softmax(logits)
                #        #labels = tf.cast(labels, tf.int32)
                #        #classify_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=config.label_nums),
                #        probabilities = tf.math.sigmoid(logits)
                #        labels = tf.cast(labels, dtype=tf.float32)
                #        classify_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
                #elif task == "regression":
                #    with tf.variable_scope("regression_loss"):
                #        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)
                #        cosine_val = cosine_similarity(sen_a_output, sen_b_output)
                #        cosine_val = tf.reshape(cosine_val, [-1])
                #        labels = tf.cast(labels, tf.float32)
                #        regression_loss= tf.losses.mean_squared_error(labels, cosine_val)
                #else:
                #    raise ValueError("task name error")

            #if task == "regression":
            #    total_loss = masked_lm_loss_a + masked_lm_loss_b + regression_loss
            #elif task == "classify":
            #    total_loss = masked_lm_loss_a + masked_lm_loss_b + classify_loss
            #else:
            #    raise ValueError("task error, loss define error")
            total_loss = masked_lm_loss_a + masked_lm_loss_b + sim_loss

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
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
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            log_hook = tf.train.LoggingTensorHook({"total_loss": total_loss, "part_loss": sim_loss}, every_n_iter=10)
            tf.summary.scalar("part_loss", sim_loss)
            tf.summary.scalar("masked_lm_loss_a", masked_lm_loss_a)
            tf.summary.scalar("masked_lm_loss_b", masked_lm_loss_b)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=10,
                output_dir=FLAGS.output_dir,
                summary_writer=None,
                summary_op=tf.summary.merge_all()
            )
            output_spec = tf.estimator.EstimatorSpec(mode = mode,
                                                     loss=total_loss,
                                                     train_op=train_op,
                                                     training_hooks=[log_hook, summary_hook],
                                                     scaffold=scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            if task == "classify":
                predictions = tf.argmax(probabilities, axis=1)
                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                        loss=total_loss,
                                                        eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)},
                                                        scaffold=scaffold)
            elif task == "regression":
                predictions = cosine_val
                output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                        loss=total_loss,
                                                        eval_metric_ops={"auc": tf.metrics.auc(labels, cosine_val)},
                                                        scaffold=scaffold)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={"sen_embedding": sen_a_output,
                                                                  "input_ids": input_ids_a,
                                                                  "word_attention_probs": sen_a_attention_probs})
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

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    config = BaseConfig.from_json_file(FLAGS.config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.Tokenizer(vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file, use_pos=False)

    run_config = None
    if FLAGS.do_train:
        num_train_steps = FLAGS.num_train_steps
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
                                init_checkpoint=FLAGS.init_checkpoint,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                embedding_table_value=None,
                                embedding_table_trainable=False,
                                model_name=FLAGS.model_name)


    params = {"batch_size": FLAGS.batch_size}
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.output_dir,
                                       config=run_config,
                                       params=params)


    if FLAGS.do_train:
        train_file = FLAGS.input_file
        tf.logging.info("***** Running training *****")
        train_input_fn = pretrain_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            mask_num=FLAGS.mask_num,
            is_training=True
        )
        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps)
    elif FLAGS.do_eval:
        dev_file = FLAGS.input_file
        eval_input_fn = pretrain_input_fn_builder(
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


if __name__ == "__main__":
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("input_file")
    tf.app.run()


