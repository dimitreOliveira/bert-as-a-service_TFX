<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/bert_icon.png?raw=true" width="800" height="300">

> [image source](https://jalammar.github.io/illustrated-bert/)

---

## BERT as a service

#### This repository is designed to demonstrate a simple yet complete machine learning solution that uses a [BERT](https://github.com/google-research/bert) model for text sentiment analysis using a [TensorFlow Extended](https://www.tensorflow.org/tfx) end-to-end pipeline, and making use of some of the best practices from the [MLOps](https://en.wikipedia.org/wiki/MLOps) domain, it will cover steps from data ingestion until model serving and consuming it either with REST or gRPC requests.

---

### Content
- Pipelines
  - Notebook (Google Colab)
    - BERT from TF HUB [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(TF_HUB).ipynb) [[colab]](https://colab.research.google.com/drive/1_9zttTBbQaLDDAo8VbC9bgOKnw_vQNCf?usp=sharing)
    - BERT from HuggingFace [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(HuggingFace).ipynb) [[colab]](https://colab.research.google.com/drive/1XE5HqqMUihxX3DD7gsejYRmv5aUDRCyG?usp=sharing)
  - Local (AirFlow) `TODO`
  - GCP (KubeFlow) `TODO`
- Documentation [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/tree/main/Documentation)
- Data [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/tree/main/Data)

---

#### Pipeline description

<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/tfx_diagram.png?raw=true" width="1000" height="300">

> [image source](https://www.tensorflow.org/tfx)

The end-to-end [TFX](https://www.tensorflow.org/tfx) pipeline will cover most of the main areas of a machine learning solution, from data ingestion and validation to model training and serving, those steps are further described below, this repository also aims to provide different options for managing the pipeline, this will be done using orchestrators, the orchestrators covered will be [AirFlow](https://airflow.apache.org/), [KubeFlow](https://www.kubeflow.org/) and an interactive option that can be used at `Google Colab` for demonstration purposes.

- `ExampleGen` is the initial input component of a pipeline that ingests and optionally splits the input dataset.
  - Reads the `IMDB` dataset stored as a `CSV` file and spits the data into train (`2/3`) and validation (`1/3`).
- `StatisticsGen` calculates statistics for the dataset.
  - Generate statistics for text and label distribution.
- `SchemaGen` examines the statistics and creates a data schema.
- `ExampleValidator` looks for anomalies and missing values in the dataset.
  - Validates the input data based on the `SchemaGen`'s schema.
- `Transform` performs feature engineering on the dataset.
  - Input missing data and do basic data pre-processing.
- `Trainer` trains the model.
  - Train the custom pre-trained `BERT` model, this model also has a built-in text tokenizer.
- `Resolver` performs model validation.
  - Resolve a model to be used as a baseline for model validation.
- `Evaluator` performs deep analysis of the training results and helps you validate your exported models, ensuring that they are "good enough" to be pushed to production.
  - Evaluate the model's accuracy over the complete dataset and across different data slices, also evaluate new models against a baseline.
- `Pusher` deploys the model on a serving infrastructure.
  - Export the model for serving if the new model improved over the baseline.

---

#### Model description

At the modeling part, we are going to use the [BERT](https://github.com/google-research/bert) model, for better performance we will use [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), this means that we are using a model that was pre-trained on another task (usually a task that is more generic or similar), from the pre-trained model we will use all layers until the output of the last embedding, to be more specific only the output from the `CLS` token, shown in the image below, then we add a classifier layer at the top, this classifier layer will be responsible for classifying the input text as being `positive` or `negative`, this task is also known as sentiment analysis, and is very common in natural language processing.

<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/bert_sent_diagram.png?raw=true" width="600" height="400">

> [image source](https://github.com/chrisjmccormick/chrisjmccormick.github.io/blob/master/_posts/2019-07-22-BERT-fine-tuning.md)

---

#### Dataset description

The dataset used for training and evaluating the model is the known [IMDB review dataset](https://ai.stanford.edu/~amaas/data/sentiment/), this dataset has 25,000 movies reviews, being either `negative (label 0)` or `positive (label 1)`, this dataset was slightly processed to be used here, labels have been encoded to be integers (0 or 1), and for faster experimentation, the data was reduced to have only 5,000 samples.
