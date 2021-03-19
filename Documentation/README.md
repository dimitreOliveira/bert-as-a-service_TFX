### References:
- [TFX homepage](https://www.tensorflow.org/tfx)
- [Faster TensorFlow models in Hugging Face Transformers](https://huggingface.co/blog/tf-serving?utm_campaign=Hugging%2BFace&utm_medium=email&utm_source=Hugging_Face_7)
---

### TODO list:
- Experiment with data from different sources and formats (e.g. `CSV`, `TXT`, and `TFRecord`).
- Store and load data using `GCS` to enable training with `TPU`.
- Add more features to the data (e.g. length, is_question, and word_count), to improve `StatisticsGen` and `Evaluator` outputs.
- Leverage `Pusher` and also deploy models for Tensorflow Lite and Tensorflow JS.
- Develop a small application to serve and consume the deployed model.
- Look into the possibilities to add unity testes as part of the pipeline.
- Experiment with custom `TFX` components.
