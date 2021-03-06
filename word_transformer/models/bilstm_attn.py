import tensorflow as tf
from .util import get_shape_list
from .util import create_initializer
from .util import assert_rank


class BilstmAttnModel(object):
    def __init__(self, config, is_training, input_ids, input_mask, embedding_table=None, scope="bilstm_attn"):
        with tf.variable_scope(scope):
            with tf.variable_scope("embeddings"):
                self.embedding_output, self.embedding_table = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.embedding_size,
                    embedding_table=embedding_table,
                    initializer_range=config.initializer_range)

            pre_output = self.embedding_output
            for idx in range(config.num_hidden_layers):
                with tf.variable_scope("bilstm_layer_%d" % (idx)):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=config.hidden_size,
                        initializer=None,
                        activation=None,
                        dtype=tf.float32)
                    bw_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=config.hidden_size,
                        initializer=None,
                        activation=None,
                        dtype=tf.float32)
                    seq_length = tf.reduce_sum(input_mask, axis=-1)
                    bilstm_output, bilstm_output_states = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_cell,
                        cell_bw=bw_cell,
                        inputs=pre_output,
                        sequence_length=seq_length,
                        initial_state_fw=None,
                        initial_state_bw=None,
                        dtype=tf.float32)
                    pre_output = tf.concat(bilstm_output, axis=-1)

            with tf.variable_scope("kw_attn"):
                query_layer = tf.layers.dense(
                    inputs=pre_output,
                    units=config.hidden_size,
                    activation=config.query_act,
                    kernel_initializer=create_initializer(config.initializer_range),
                    name="attn_query")
                value_layer = pre_output

                weights = tf.get_variable(name="attn_weights", shape=[config.hidden_size],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
                attention_scores = tf.tensordot(query_layer, weights, axes=1)
                attention_mask = input_mask
                mask_adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
                attention_scores += mask_adder
                attention_probs = tf.nn.softmax(attention_scores)
                shape_list = get_shape_list(value_layer)
                batch_size = shape_list[0]
                context_layer = tf.math.multiply(tf.reshape(attention_probs, [batch_size, config.max_seq_length, 1]), value_layer)
                # return context_layer, attention_probs
                self.sequence_output = context_layer
                self.keyword_probs = attention_probs

    def get_sequence_output(self):
        return self.sequence_output

    def get_keyword_probs(self):
        return self.keyword_probs

    def get_sentence_output(self):
        pass


def embedding_lookup(input_ids,
                    vocab_size,
                    embedding_size=128,
                    embedding_table=None,
                    initializer_range=0.02,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      embedding_table: embedding_table
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.
    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    if embedding_table is None:
        embedding_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range),
            trainable=True)

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    tf.logging.debug("[check embedding_lookup] output:%s, embedding_table:%s" % (output.shape, embedding_table.shape)) 
    return (output, embedding_table)



