import logging
import tensorflow as tf
import os
import math
from .models import BertModel
from .models.bert_transformer import get_shape_list
from .models.bert_transformer import create_initializer
from .models.bert_transformer import create_attention_mask_from_input_mask
from .models.bert_transformer import optimization
from .models.bert_transformer import get_assignment_map_from_checkpoint
from .models.bert_transformer import BertConfig
from .data_struct import SenpairProcessor
from .input_function import file_based_convert_examples_to_features
from .input_function import file_based_input_fn_builder
from . import tokenization


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config_file", None,
    "config_file path"
)

flags.DEFINE_string(
    "",
)

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

    key_layer = tf.layers.dense(
        inputs=input_tensor,
        units=hidden_size,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range)
    )

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
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(hidden_size)))
    # attention_mask shape: [batch_size, seq_length, seq_length]
    attention_mask = create_attention_mask_from_input_mask(input_tensor, input_mask)
    # expand for multi heads, [batch_size, 1, seq_length, seq_length]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])
    mask_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    # attention_score: [batch_size, num_heads, seq_length, seq_length]
    attention_scores += mask_adder

    # attention_probs shape: [batch_size, seq_length]
    attention_probs = tf.nn.softmax(attention_scores)
    # value_layer = tf.reshape(value_layer, [batch_size, seq_length, hidden_size])
    # value_layer shape : [batch_size, num_heads, seq_length, size_per_head]
    # value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    context_layer = tf.matmul(attention_probs, value_layer)
    # context_layer shape : [batch_size, seq_length, hidden_size]
    # context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    # context_layer = tf.reshape(context_layer, [batch_size, seq_length, hidden_size])

    #if pooling_strategy == "mean":
    # input_mask_expanded = tf.cast(input_mask.expand(), float)
    pooling_output = tf.reduce_sum(tf.matmul(context_layer, input_mask), axis=1)/tf.reduce_sum(input_mask, axis=1)
    tf.logging.debug("pooling_output: %s"%(get_shape_list(pooling_output)))
    return pooling_output, attention_probs


def create_kwextraction_model(input_ids,
                              input_mask,
                              config=None,
                              is_training=None,
                              embedding_table=None,
                              hidden_size=None,
                              query_act=None,
                              key_act=None):

    encode_model = BertModel(config=config,
                             is_training=is_training,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             embedding_table=embedding_table,
                             use_one_hot_embeddings=False)

    sequence_output = encode_model.get_sequence_output()
    # attention_probs : [batch_size, seq_length]
    pooling_output, attention_probs = word_attention_layer(input_tensor=sequence_output,
                                                           input_mask=input_mask,
                                                           hidden_size=hidden_size,
                                                           query_act=query_act,
                                                           key_act=key_act)

    return pooling_output, attention_probs


def model_fn_builder(config,
                     learning_rate=1e-5,
                     task="classify",
                     num_train_steps=10,
                     num_warmup_steps=1,
                     init_checkpoint=None,
                     embedding_table_value=None,
                     embedding_table_trainable=False,
                     classify_num=2):
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
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            input_ids_b = features["input_ids_b"]
            input_mask_b = features["input_mask_b"]
            labels = features["labels"]

        embedding_table = tf.get_variable("embedding_table",
                                          shape=[config.vocab_size, config.vocab_vec_size],
                                          trainable=embedding_table_trainable)

        def init_embedding_table(scoffold, sess):
            sess.run(embedding_table.initializer, {embedding_table.initial_value: embedding_table_value})

        if embedding_table_value is not None:
            scaffold = tf.train.Scaffold(init_fn=init_embedding_table)
        else:
            scaffold = None

        sen_a_output, sen_a_attention_probs = create_kwextraction_model(input_ids=input_ids_a,
                                                                        input_mask=input_mask_a,
                                                                        config=None,
                                                                        is_training=None,
                                                                        embedding_table=None,
                                                                        hidden_size=None,
                                                                        query_act=None)

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            sen_b_output, sen_b_attention_probs = create_kwextraction_model(input_ids=input_ids_b,
                                                                            input_mask=input_mask_b,
                                                                            config=None,
                                                                            is_training=None,
                                                                            embedding_table=None,
                                                                            hidden_size=None,
                                                                            query_act=None)

            total_loss = None
            with tf.variable_scope("sen_pair_loss"):
                if task == "classify":
                    with tf.variable("classify_loss"):
                        concat_vector = tf.concat([sen_a_output, sen_b_output, tf.abs(sen_a_output-sen_b_output)], axis=1)
                        logits = tf.layers.dense(concat_vector, classify_num, kernel_initializer=None)
                        probabilities = tf.nn.softmax(logits)
                        softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels),
                                                                   logits=logits)
                        total_loss = softmax_loss
                elif task == "regression":
                    with tf.variabel("regression_loss"):
                        cosine_similarity = tf.keras.losses.CosineSimilarity(axis=1)
                        cosine_val = cosine_similarity(sen_a_output, sen_b_output)
                        total_loss = tf.losses.mean_squared_error(labels, cosine_val)
                else:
                    raise ValueError("task name error")

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
        scaffold = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            log_hook = tf.train.LoggingTensorHook({"total_loss": total_loss}, every_n_iter=10)
            output_spec = tf.estimator.EstimatorSpec(mode = mode,
                                                     loss=total_loss,
                                                     train_op=train_op,
                                                     training_hooks=[log_hook],
                                                     scaffold=scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            if task == "classify":
                predictions = probabilities
            elif task == "regression":
                predictions = cosine_val
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     eval_metric_ops={"auc": tf.metrics.auc(labels, cosine_val)},
                                                     scaffold=scaffold)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={"sen_embedding": sen_a_output,
                                                                  "word_attention_probs": sen_a_attention_probs})
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processor = SenpairProcessor()

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    config = BertConfig.from_json_file(FLAGS.config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.Tokenizer(vocab_file=FLAGS.vocab_file, stop_words_file=FLAGS.stop_words_file, use_pos=False)

    run_config = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        run_config = tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=num_train_steps/FLAGS.num_train_epochs,
            keep_checkpoint_max=5,
        )

    model_fn = model_fn_builder(config=config,
                                learning_rate=1e-5,
                                task="regression",
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                init_checkpoint=FLAGS.init_checkpoint,
                                embedding_table_value=None,
                                embedding_table_trainable=False)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.output_dir,
                                       config=run_config,
                                       params=None)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, FLAGS.max_seq_length, tokenizer, train_file)
        del train_examples  # 释放train_examples内存
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True
        )

        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps)

    elif FLAGS.do_eval:
        pass
    else:
        pass



