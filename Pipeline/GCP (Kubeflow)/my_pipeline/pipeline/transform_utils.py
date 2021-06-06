import tensorflow as tf
import tensorflow_transform as tft


_FEATURE_KEY = 'review'
_LABEL_KEY = 'sentiment'

# using LSTM model
_VOCAB_SIZE = 8000
_MAX_LEN = 300


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def _tokenize_review(review): # using LSTM model
  """Tokenize the reviews by spliting the reviews.
  Source: https://github.com/tensorflow/tfx/blob/v1.0.0-rc1/tfx/examples/imdb/imdb_utils_native_keras.py
  Constructing a vocabulary. Map the words to their frequency index in the
  vocabulary.
  Args:
    review: tensors containing the reviews. (batch_size/None, 1)
  Returns:
    Tokenized and padded review tensors. (batch_size/None, _MAX_LEN)
  """
  review_sparse = tf.strings.split(tf.reshape(review, [-1])).to_sparse()
  # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
  # with convention and use mask_zero in keras.embedding layer, set oov value
  # to _VOCAB_SIZE and padding value to -1. Then add 1 to all the tokens.
  review_indices = tft.compute_and_apply_vocabulary(
      review_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
  dense = tf.sparse.to_dense(review_indices, default_value=-1)
  # TFX transform expects the transform result to be FixedLenFeature.
  padding_config = [[0, 0], [0, _MAX_LEN]]
  dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
  padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
  padded += 1
  return padded

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  trimmed_text = tf.strings.strip(inputs[_FEATURE_KEY])
  return {
#       _FEATURE_KEY: _fill_in_missing(trimmed_text),
      _FEATURE_KEY: _fill_in_missing(_tokenize_review(inputs[_FEATURE_KEY])), # using LSTM model
      'label': _fill_in_missing(inputs[_LABEL_KEY])
  }
