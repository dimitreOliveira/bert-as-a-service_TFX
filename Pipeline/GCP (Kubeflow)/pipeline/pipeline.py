from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text

import os
from tfx.orchestration import pipeline
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2

from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    module_path: Text,
    enable_tuning = False,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> pipeline.Pipeline:

  components = []

  # Output 2 splits: train:eval=4:1
  output = example_gen_pb2.Output(
               split_config=example_gen_pb2.SplitConfig(splits=[
                   example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
                   example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
               ]))
  example_gen = CsvExampleGen(input_base=data_path, output_config=output)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], 
                         infer_feature_shape=True)
  components.append(schema_gen)

  example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(module_path))
  components.append(transform)

#   if enable_tuning:
#     tuner = Tuner(
#         module_file=os.path.abspath(module_path),
#         examples=transform.outputs['transformed_examples'],
#         transform_graph=transform.outputs['transform_graph'],
#         train_args=trainer_pb2.TrainArgs(num_steps=5),
#         eval_args=trainer_pb2.EvalArgs(num_steps=2))
#   components.append(tuner)

#   # Uses user-provided Python function that implements a model using TF-Learn.
#   trainer_args = {
#       'run_fn': run_fn,
#       'transformed_examples': transform.outputs['transformed_examples'],
#       'schema': schema_gen.outputs['schema'],
#       'transform_graph': transform.outputs['transform_graph'],
#       'train_args': train_args,
#       'eval_args': eval_args,
#   }
#   if ai_platform_training_args is not None:
#     trainer_args['custom_config'] = {
#         tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
#             ai_platform_training_args,
#     }
#     trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**trainer_args)
#   else:
#     trainer = tfx.components.Trainer(**trainer_args)
#   components.append(trainer)

  trainer = Trainer(
      module_file=os.path.abspath(module_path),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use ImporterNode to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of ImporterNode,
      #   hparams_importer = ImporterNode(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
#      hyperparameters=(tuner.outputs['best_hyperparameters']
#                         if enable_tuning else None),
      train_args=trainer_pb2.TrainArgs(num_steps=20),
      eval_args=trainer_pb2.EvalArgs(num_steps=5))
  components.append(trainer)



  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
