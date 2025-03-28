import numpy as np
from src.embeddings import get_embeddings


def test_embeddings_creation():
    embeddings = get_embeddings()
    assert embeddings is not None


def test_embedding_dimensions():
    embeddings = get_embeddings()
    test_text = "This is a test document"
    result = embeddings.embed_query(test_text)
    assert isinstance(result, list)
    assert len(result) == 768  # nomic dim


def test_embedding_similarity():
    embeddings = get_embeddings()
    text1 = "Python programming"
    text2 = "Python coding"
    text3 = "Database management"

    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    emb3 = embeddings.embed_query(text3)

    # Convert to numpy arrays for cosine similarity
    sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

    # Similar texts should have higher similarity
    assert sim_12 > sim_13


def test_embeddings_caching():
    # Test that the lru_cache is working
    emb1 = get_embeddings()
    emb2 = get_embeddings()
    assert emb1 is emb2
