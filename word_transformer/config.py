import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BaseConfig(object):
    def __init__(self,
                 vocab_size=0,
                 embedding_size=0,
                 hidden_size=256,
                 num_hidden_layers=2,
                 num_attention_heads=4,
                 intermediate_size=512,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=256,
                 initializer_range=0.02,
                 query_act="tanh",
                 key_act="tanh"):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.query_act = query_act
        self.key_act = key_act

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BaseConfig` from a Python dictionary of parameters."""
        config = BaseConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BaseConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


