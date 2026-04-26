"""
Lazy singleton for SentenceTransformer to avoid duplicate loads
(Verifier FAISS path + DisagreementDetector share one model instance).
"""
from __future__ import annotations

_encoder = None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(model_name)
    return _encoder
