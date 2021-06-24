from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Text

import os
import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2

from tfx.orchestration import pipeline
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.proto import example_gen_pb2, pusher_pb2, infra_validator_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_blessed_model_resolver


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    transform_module_path: Text,
    train_module_path: Text,
    train_args,
    eval_args,
    eval_accuracy_threshold,
    serving_model_dir,
    enable_tuning = False,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> pipeline.Pipeline:

  # Output 2 splits: train:eval=4:1
  output = example_gen_pb2.Output(
               split_config=example_gen_pb2.SplitConfig(splits=[
                   example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
                   example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
               ]))
  example_gen = CsvExampleGen(input_base=data_root, output_config=output)


  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])


  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], 
                         infer_feature_shape=True)


  example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])


  transform = Transform(
    module_file=transform_module_path,
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'])


  if enable_tuning:
    tuner = Tuner(
        module_file=train_module_path,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=train_args,
        eval_args=eval_args)


  trainer = Trainer(
      module_file=train_module_path,
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
     hyperparameters=(tuner.outputs['best_hyperparameters']
                        if enable_tuning else None),
      train_args=train_args,
      eval_args=eval_args)


  model_resolver = Resolver(
    strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
  ).with_id('latest_blessed_model_resolver')


  # Use TFMA to compute a evaluation statistics over features of a model and
  # validate them against a baseline.
  eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='sentiment')],
        slicing_specs=[tfma.SlicingSpec(), 
                       tfma.SlicingSpec(feature_keys=['sentiment'])],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            # Increase this threshold when training on complete
                            # dataset.
                            lower_bound={'value': eval_accuracy_threshold}),
                        # Change threshold will be ignored if there is no
                        # baseline model repipeline_rootsolved from MLMD (first run).
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -eval_accuracy_threshold})))
            ])
        ])

  evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
#     baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)


  # Performs infra validation of a candidate model to prevent unservable model
  # from being pushed. In order to use InfraValidator component, persistent
  # volume and its claim that the pipeline is using should be a ReadWriteMany
  # access mode.
  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=infra_validator_pb2.ServingSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(
              tags=['latest']),
          kubernetes=infra_validator_pb2.KubernetesConfig()),
      request_spec=infra_validator_pb2.RequestSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec())
  )


  pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
#     infra_blessing=infra_validator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir))
  )


  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          transform,
#           tuner,
          trainer,
#           model_resolver,
          evaluator,
#           infra_validator, # not supported
          pusher,
      ],
      # Change this value to control caching of execution results. Default value
      # is `False`.
      enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
