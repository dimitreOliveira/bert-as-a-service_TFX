<img src="https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Assets/colab_icon.png?raw=true" width="600" height="200">

### Google Colab pipeline

**BERT from TF HUB** [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(TF_HUB).ipynb) [[colab]](https://colab.research.google.com/drive/1_9zttTBbQaLDDAo8VbC9bgOKnw_vQNCf?usp=sharing)
- Model: BERT base uncased (english)
- Data: IMDB movie review (5,000 samples)
- Pre-processing: Text trimming, Tokenizer (sequence length of 128, lower case)
- Training: `epochs`: 3, `batch size`: 32, `learning rate`: 1e-5, `loss`: binary crossentropy.

**BERT from HuggingFace** [[link]](https://github.com/dimitreOliveira/bert-as-a-service_TFX/blob/main/Pipeline/Notebook%20(Google%20Colab)/BERT_as_a_service_TFX_Colab_(HuggingFace).ipynb) [[colab]](https://colab.research.google.com/drive/1XE5HqqMUihxX3DD7gsejYRmv5aUDRCyG?usp=sharing)
- Model: BERT base uncased (english)
- Data: IMDB movie review (5,000 samples)
- Pre-processing: Text trimming, Tokenizer (sequence length of 128, lower case)
- Training: `epochs`: 3, `batch size`: 32, `learning rate`: 1e-5, `loss`: binary crossentropy.
