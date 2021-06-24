PIPELINE_NAME = 'bert-aas-vertex-pipelines'

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.yaml'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = ''
except ImportError:
  GOOGLE_CLOUD_PROJECT = ''

# Specify your GCS bucket name here.
GCS_BUCKET_NAME = 'bert-aas-storage'

GOOGLE_CLOUD_REGION = 'us-central1'

# Path to various pipeline artifact.
PIPELINE_ROOT = f'gs://{GCS_BUCKET_NAME}/pipeline_root/{PIPELINE_NAME}'

# Paths for users' Python module.
MODULE_ROOT = f'gs://{GCS_BUCKET_NAME}/pipeline_module/{PIPELINE_NAME}'

# Paths for input data.
DATA_ROOT = f'gs://{GCS_BUCKET_NAME}/data/{PIPELINE_NAME}'

# This is the path where your model will be pushed for serving.
SERVING_MODEL_DIR = f'gs://{GCS_BUCKET_NAME}/serving_model/{PIPELINE_NAME}'

# Pipeline parameters
TRAIN_NUM_STEPS = 10
EVAL_NUM_STEPS = 5
EVAL_ACCURACY_THRESHOLD = 0.01
ENABLE_TUNNING = False
