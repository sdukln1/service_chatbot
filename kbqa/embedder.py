from __future__ import annotations

import threading

from kbqa.config import CONFIG, resolve_model

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(resolve_model(CONFIG.embedding_model))
    return _model


class Embedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        embeddings = model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        model = _get_model()
        embedding = model.encode(
            BGE_QUERY_INSTRUCTION + text, normalize_embeddings=True
        )
        return embedding.tolist()
