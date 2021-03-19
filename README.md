<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/bert_icon.png?raw=true" width="800" height="300">

---

## BERT as a service

#### This repository is designed to demonstrate a simple yet complete machine learning solution that uses a [BERT](https://github.com/google-research/bert) model for text sentiment analysis using a [TensorFlow Extended](https://www.tensorflow.org/tfx) end-to-end pipeline, covering steps from data ingestion until model serving, and making use of some best practice from the [MLOps](https://en.wikipedia.org/wiki/MLOps) domain.

---

### Content
- Notebook (Google Colab)
  - BERT from TF HUB [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(TF_HUB).ipynb)
  - BERT from HuggingFace [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(HuggingFace).ipynb)
- Local (AirFlow) `TODO`
- GCP (KubeFlow) `TODO`

---

#### Pipeline description `TODO`

<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/tfx_diagram.png?raw=true" width="1000" height="300">

The end-to-end [TFX](https://www.tensorflow.org/tfx) pipeline will cover most of the main areas of a machine learning solution, from data ingestion and validation to model training and serving, those steps are further described below, this repository also aims to provide different options for managing the pipeline, this will be done using orchestrators, the orchestrators covered will be [AirFlow](https://airflow.apache.org/), [KubeFlow](https://www.kubeflow.org/) and an interactive option that can be used at Google Colab for demonstration purposes.

- ExampleGen is the initial input component of a pipeline that ingests and optionally splits the input dataset.
- StatisticsGen calculates statistics for the dataset.
- SchemaGen examines the statistics and creates a data schema.
- ExampleValidator looks for anomalies and missing values in the dataset.
- Transform performs feature engineering on the dataset.
- Trainer trains the model.
- Tuner tunes the hyperparameters of the model.
- Evaluator performs deep analysis of the training results and helps you validate your exported models, ensuring that they are "good enough" to be pushed to production.
- InfraValidator checks the model is actually servable from the infrastructure, and prevents bad model from being pushed.
- Pusher deploys the model on a serving infrastructure.

---

#### Model description

For the modeling part, we are going to use the [BERT](https://github.com/google-research/bert) model, for better performance we are using [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), this means that we are using a model that was pre-trained on another task (usually a task that is more generic or similar), from the pre-trained model we will use the layers until the output of the embedding, to be more specific only the output from the `CLS` token, shown in the image below, then we add a classifier layer at the top, this classifier layer will be responsible for classifying the input text as being `positive` or `negative`, this task is also known as sentiment analysis, and is very common in natural language processing.

<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/bert_sent_diagram.png?raw=true" width="400" height="250">
