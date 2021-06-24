from typing import List, Text

import absl
import kerastuner
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult


_MODEL_PATH = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
_PREPROCESSOR_PATH = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
_FEATURE_KEY = 'review'
_SEQ_LENGTH = 64
_TRAIN_BATCH_SIZE = 16
_EVAL_BATCH_SIZE = 16
_EPOCHS = 3

# using LSTM model
_EMBEDDING_UNITS = 32
_HIDDEN_UNITS = 32
_LSTM_UNITS = 32
_VOCAB_SIZE = 8000


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# TFX Trainer will call this function.
def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.
  Args:
    file_pattern: List of paths or patterns of materialized transformed input
      tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch
  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key='label')

  return dataset.prefetch(tf.data.AUTOTUNE)

def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    reshaped_examples = tf.reshape(serialized_tf_examples, [-1, 1])
    transformed_features = model.tft_layer({_FEATURE_KEY: reshaped_examples})

    outputs = model(transformed_features)
    return {'outputs': outputs}

  return serve_tf_examples_fn

def get_strategy():
  """Detect hardware and return appropriate distribution strategy."""
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    absl.logging.info(f'Running on TPU {tpu.master()}')
  except ValueError:
    tpu = None

  if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
  else:
    strategy = tf.distribute.get_strategy()
  
  return strategy

def preprocess_model_fn(seq_length=_SEQ_LENGTH):
  """Returns Model mapping string features to encoder inputs.

  Args:
    seq_length: an integer that defines the sequence length of the inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors and
    returns a dict of tensors for input to model.
  """
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, 
                                     name=_FEATURE_KEY)

  # Tokenize the text to word pieces.
  preprocessor = hub.load(_PREPROCESSOR_PATH)
  tokenizer = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
  tokenized_inputs = tokenizer(text_input)

  # Pack inputs. The details (start/end token ids, dict of output tensors)
  # are model-dependent, so this gets loaded from the SavedModel.
  packer = hub.KerasLayer(preprocessor.bert_pack_inputs,
                          arguments={'seq_length': seq_length}, 
                          name='packer')
  model_inputs = packer([tokenized_inputs])
  return tf.keras.Model(text_input, model_inputs)

def model_fn(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Return a BERT model built for sentence classification.
  Args:
    learning_rate: learning rate value that will be used for training.
  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  preprocess_model = preprocess_model_fn(_SEQ_LENGTH)

  encoder = hub.KerasLayer(_MODEL_PATH, trainable=True)
  
  outputs = encoder(preprocess_model.output)
  pooled_output = outputs['pooled_output']

  x = tf.keras.layers.Dropout(hparams.get('dropout'))(pooled_output)
  output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(preprocess_model.inputs, output)

  model.compile(optimizer=tf.keras.optimizers.Adam(hparams.get('learning_rate')),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])  
  model.summary(print_fn=absl.logging.info)

  return model

def model_fn_temp(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a LSTM Keras model for classifying imdb data.
  Reference: https://www.tensorflow.org/tutorials/text/text_classification_rnn
  Returns:
    A Keras Model.
  """
  model = tf.keras.Sequential([
          tf.keras.layers.Embedding(_VOCAB_SIZE + 2,
                                    _EMBEDDING_UNITS,
                                    name=_FEATURE_KEY),
          tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(_LSTM_UNITS, dropout=hparams.get('dropout'))
          ),
          tf.keras.layers.Dense(_HIDDEN_UNITS, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
      ])

  model.compile(optimizer=tf.keras.optimizers.Adam(hparams.get('learning_rate')),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])  
  model.summary(print_fn=absl.logging.info)

  return model

def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kerastuner.HyperParameters()
  hp.Choice('learning_rate', [3e-4, 1e-4, 3e-5], default=3e-5)
  hp.Choice('dropout', [0.0, 0.25, 0.5], default=0.5)
  return hp

# TFX Tuner will call this function.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # RandomSearch is a subclass of kerastuner.Tuner which inherits from
  # BaseTuner.
  tuner = kerastuner.RandomSearch(
#       model_fn,
      model_fn_temp, # using LSTM model
      max_trials=3,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=kerastuner.Objective('val_binary_accuracy', 'max'),
#       directory=fn_args.working_dir,
      project_name='bert_tuning')
  
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = _input_fn(fn_args.train_files, 
                            tf_transform_output, 
                            batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(fn_args.eval_files, 
                           tf_transform_output, 
                           batch_size=_EVAL_BATCH_SIZE)

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })

# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.
  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, 
                            tf_transform_output, 
                            batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(fn_args.eval_files, 
                           tf_transform_output, 
                           batch_size=_EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()

  absl.logging.info(f'HyperParameters for training: {hparams.get_config()}')

  strategy = get_strategy()
  with strategy.scope():
#     model = model_fn(hparams)
    model = model_fn_temp(hparams) # using LSTM model

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, 
      update_freq='batch'
  )

  model.fit(
      train_dataset,
      epochs=_EPOCHS,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps, 
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name=_FEATURE_KEY)),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)