### References:
- [TFX homepage](https://www.tensorflow.org/tfx)
- [TFX YouTube channel](https://www.youtube.com/playlist?list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F)
- [TFX BERT example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/bert)
- [TFX IMDB example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/imdb)
- [Train and serve a TensorFlow model with TensorFlow Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
- [Part 2: Fast, scalable and accurate NLP: Why TFX is a perfect match for deploying BERT](https://blog.tensorflow.org/2020/06/part-2-fast-scalable-and-accurate-nlp.html)
- [Faster TensorFlow models in Hugging Face Transformers](https://huggingface.co/blog/tf-serving?utm_campaign=Hugging%2BFace&utm_medium=email&utm_source=Hugging_Face_7)
- [TFX TensorFlow in Production tutorials](https://www.tensorflow.org/tfx/tutorials)
  - [TFX Keras Component Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras#pusher)
  - [TFX Airflow Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/airflow_workshop)
  - [TFX on Cloud AI Platform Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)

---

### TODO list:
- Store and load data using `GCS` to enable training with `TPU`.
- Add more features to the data (e.g. length, is_question, and word_count), to improve `StatisticsGen` and `Evaluator` outputs.
- Leverage `Pusher` and also deploy models for Tensorflow Lite and Tensorflow JS.
- Develop a small application to serve and consume the deployed model.
- Look into the possibilities to add unity tests as part of the pipeline.
- Experiment with custom `TFX` components.
- Experiment with other components (e.g. `BulkInferrer`).
