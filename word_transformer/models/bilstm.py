import tensorflow as tf
from .util import get_shape_list
from .util import create_initializer
from .util import assert_rank



class BilstmModel(object):
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
                    pre_output = tf.nn.tanh(pre_output)

            context_output = pre_output
            # final_output = tf.layers.dense(context_output, config.hidden_size, activation=tf.tanh)
            self.text_encoding = tf.reduce_mean(context_output, axis=1)

    def get_sequence_output(self):
        return self.sequence_output

    def get_keyword_probs(self):
        return self.keyword_probs

    def get_text_encoding(self):
        return self.text_encoding


def embedding_lookup(input_ids,
                    vocab_size,
                    embedding_size=128,
                    embedding_table=None,
                    initializer_range=0.02,
                    word_embedding_name="word_embeddings"):

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
        trainable=True)

    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    tf.logging.debug("[check embedding_lookup] output:%s, embedding_table:%s" % (output.shape, embedding_table.shape))
    return (output, embedding_table)



