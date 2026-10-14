# Weaviate reranker inference service

`app.py`, `crossencoder.py`, `meta.py`, and `download.py` are unmodified copies
from [weaviate/reranker-transformers](https://github.com/weaviate/reranker-transformers)
at commit `200383773a04411a52c0bd626e51b4072b076a8f`. The upstream BSD
license is included in `LICENSE`.

This project's Dockerfile installs the pinned CPU PyTorch wheel before the
upstream Python dependencies. The upstream published amd64 image includes a
5.9 GB compressed layer even when running on CPU; this local build keeps the
same Weaviate inference API and model without that GPU-sized image.
