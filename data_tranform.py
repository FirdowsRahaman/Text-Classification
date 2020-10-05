import pandas as pd
import tensorflow as tf

from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_string_ops


COLUMN_NAMES = ['TITLE', 'CATEGORY']
CLASSES = {'business': 0, 'science and technology': 1, 'health': 2, 'entertainment': 3}
MAX_SEQUENCE_LENGTH = 20
OUTPUT_DIR = '/tmp'
VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, 'vocab.txt')


def load_data(data_path):
  dataframe = pd.read_csv(data_path, usecols=COLUMN_NAMES)
  dataframe = dataframe.dropna()
  return (list(dataframe['TITLE']), np.array(dataframe['CATEGORY'].map(CLASSES)))


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


def text_transform(text):
  token = _preprocess(text)
  words = token.to_tensor(shape=(None, MAX_SEQUENCE_LENGTH))
  init = tf.lookup.TextFileInitializer(filename=VOCAB_FILE_PATH, 
                              key_dtype=tf.string, 
                              key_index=0,#tf.lookup.TextFileIndex.WHOLE_LINE,
                              value_dtype=tf.int64, 
                              value_index=1,#tf.lookup.TextFileIndex.LINE_NUMBER,
                              vocab_size=None,
                              delimiter=',')
  
  table = tf.lookup.StaticVocabularyTable(initializer=init, num_oov_buckets=1)             
  word2numbers = table.lookup(words)
  return (word2numbers)