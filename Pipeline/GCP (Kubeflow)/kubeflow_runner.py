from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

from pipeline import configs
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.utils import telemetry_utils

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# GCS path: `gs://{GCS_BUCKET_NAME}/bert-aas/data/`. Using a GCS path is
# recommended for KFP.
#
# One can optionally choose to use a data source located inside of the container
# built by the template, by specifying
# DATA_PATH = 'data'. Note that Dataflow does not support use container as a
# dependency currently, so this means CsvExampleGen cannot be used with Dataflow.

DATA_PATH = 'gs://{}/bert-aas/data/'.format(configs.GCS_BUCKET_NAME)

ROOT_PATH = os.path.join(os.path.expanduser("~"), "imported", 
                         configs.PIPELINE_NAME)
MODULE_PATH = os.path.join(ROOT_PATH, 'pipeline', 'bert_aas_utils.py')


def run():
  """Define a kubeflow pipeline."""

  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  # If you use Kubeflow, metadata will be written to MySQL database inside
  # Kubeflow cluster.
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      tfx_image=configs.PIPELINE_IMAGE)
#   pod_labels = kubeflow_dag_runner.get_default_pod_labels()
#   pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: 'tfx-template'})
  kubeflow_dag_runner.KubeflowDagRunner(
      config=runner_config, 
#       pod_labels_to_attach=pod_labels
  ).run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          module_path=MODULE_PATH,
          enable_tuning=configs.ENABLE_TUNNING,
      ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
