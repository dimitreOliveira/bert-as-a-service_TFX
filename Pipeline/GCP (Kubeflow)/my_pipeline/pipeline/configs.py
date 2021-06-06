# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = 'my_pipeline'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = ''
except ImportError:
  GOOGLE_CLOUD_PROJECT = ''

# Specify your GCS bucket name here. You have to use GCS to store output files
# when running a pipeline with Kubeflow Pipeline on GCP or when running a job
# using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
# This bucket is created automatically when you deploy KFP from marketplace.
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

PREPROCESSING_FN = 'models.transform_utils.preprocessing_fn'
RUN_FN = 'models.train_utils.run_fn'

TRAIN_NUM_STEPS = 10
EVAL_NUM_STEPS = 5
EVAL_ACCURACY_THRESHOLD = 0.01
ENABLE_TUNNING = True
