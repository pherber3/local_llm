from functools import lru_cache
from langchain_ollama import OllamaEmbeddings
from typing import Any

# @lru_cache()
# def get_embeddings(**kwargs: dict[str, Any]) -> HuggingFaceEmbeddings:
#     """Get cached embedding model instance."""
#     return HuggingFaceEmbeddings(
#         model_name="nvidia/NV-Embed-v2",
#         model_kwargs={'device': 'mps', "trust_remote_code": True},
#         encode_kwargs={'normalize_embeddings': True}
#     )


@lru_cache()
def get_embeddings(
    model: str = "nomic-embed-text", **kwargs: dict[str, Any]
) -> OllamaEmbeddings:
    """
    Get cached embedding model instance.
    """
    return OllamaEmbeddings(model=model, **kwargs)
