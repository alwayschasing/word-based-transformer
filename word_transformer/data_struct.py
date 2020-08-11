import pickle
import tensorflow as tf
import logging
from .models.bert_transformer import tokenization

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 input_ids_a,
                 input_mask_a,
                 input_ids_b=None,
                 input_mask_b=None,
                 label=None):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.label = label


class DataProcessor(object):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        lines = []
        f = open(input_file, "r")
        for line in f:
            parts = line.strip().split('\t')
            lines.append(parts)
        return lines


class SenpairProcessor(DataProcessor):
    def __init__(self):
        pass

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = ""
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = 1
            else:
                if len(line) < 3:
                    logging.error("[data error] line %d, %s" % (i, line))
                    raise ValueError("data format error, parts less 3")
                label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_train_examples(self, data_path):
        return self._create_examples(self._read_tsv(data_path), "train")

    def get_dev_examples(self, data_path):
        return self._create_example(self._read_tsv(data_path), "dev")

    def get_test_examples(self, data_path):
        return self._create_example(self._read_tsv(data_path), "test")

