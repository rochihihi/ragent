# Local memory embedding model

RAgent bundles data from `Xenova/paraphrase-multilingual-MiniLM-L12-v2`, revision
`2c4055b12046f11709e9df2c122e59ffbdc2f900`, quantized ONNX export of the
Sentence Transformers multilingual MiniLM model. Authors: Nils Reimers and
Iryna Gurevych / Sentence Transformers; ONNX conversion: Xenova.

- Base model and Apache-2.0 license: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Export: https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2
- Reference: Reimers and Gurevych (2019), Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks. https://arxiv.org/abs/1908.10084

The weights/tokenizer are redistributed unchanged under Apache-2.0 (see LICENSE).
RAgent applies attention-mask mean pooling and L2 normalization locally using
ONNX Runtime (CPU); it does not train the model or execute downloaded Python code.
The application's MIT license does not relicense these model artifacts.
