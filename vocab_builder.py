import os
import operator
import collections
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import compat


COLUMN_NAMES = ['TITLE']
VOCAB_SIZE = 2**13
MAX_SEQUENCE_LENGTH = 20
OUTPUT_DIR = '/tmp'
VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, 'vocab.txt')
DEFAULT_STRIP_REGEX = r'[’!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'


class Vocabulary():
  def __init__(self, csv_file):
    self.csv_file = csv_file


  def load_data(self):
    dataframe = pd.read_csv(self.csv_file, usecols=COLUMN_NAMES)
    dataframe = dataframe.dropna()
    return list(dataframe['TITLE'])
    
    
  def _preprocess(self, inputs):
    if ragged_tensor.is_ragged(inputs):
      lowercase_inputs = ragged_functional_ops.map_flat_values(
          gen_string_ops.string_lower, inputs)
      lowercase_inputs = array_ops.identity(lowercase_inputs)
    else:
      lowercase_inputs = gen_string_ops.string_lower(inputs)
    inputs = string_ops.regex_replace(lowercase_inputs, DEFAULT_STRIP_REGEX,
                                        " ")
    tokens = ragged_string_ops.string_split_v2(inputs)
    return tokens
    
    
  def _token_counts_from_preprocess_corpus(self, tokens):
    values = self._preprocess(tokens)
    
    if ragged_tensor.is_ragged(values):
      values = values.to_list()
    if isinstance(values, ops.EagerTensor):
      values = values.numpy()
    if isinstance(values, np.ndarray):
      values = values.tolist()

    accumulator = collections.namedtuple(
        "Accumulator", ["count_dict", "metadata"])

    count_dict = collections.defaultdict(int)
    metadata = [0]
    accumulator = accumulator(count_dict, metadata)

    for document in values:
      current_doc_id = accumulator.metadata[0]
      for token in document:
        accumulator.count_dict[compat.as_str_any(token)] += 1
      accumulator.metadata[0] += 1

    return accumulator
    
    
  def extract(self, inputs):
    accumulator = self._token_counts_from_preprocess_corpus(inputs)
    vocab_counts, _ = accumulator

    sorted_counts = sorted(
        vocab_counts.items(), key=operator.itemgetter(1, 0), reverse=True)
    vocab_data = (
        sorted_counts[:VOCAB_SIZE] if VOCAB_SIZE else sorted_counts)
    vocab = [data[0] for data in vocab_data]
    
    return vocab
    
    
  def create_vocabulary(self):
    inputs = self.load_data()
    vocab_list = self.extract(inputs)
    vocab_list.insert(VOCAB_SIZE-2, '<OOV>')
    word_index = dict(zip(vocab_list, list(range(1, len(vocab_list)+1))))
    
    tf.io.gfile.mkdir(OUTPUT_DIR)  
    global VOCAB_FILE_PATH; VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR,'vocab.txt')
    with tf.io.gfile.GFile(VOCAB_FILE_PATH, 'wb') as f:
      for word, index in word_index.items():
        if index < VOCAB_SIZE: 
          f.write("{},{}\n".format(word, index))