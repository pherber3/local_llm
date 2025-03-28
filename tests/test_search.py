import pytest
from src.search import WebSearcher
from dotenv import load_dotenv

@pytest.fixture(scope="module")
def web_searcher():
    load_dotenv()
    return WebSearcher()

def test_search_functionality(web_searcher):
    query = "What is Python programming?"
    response = web_searcher.search(query)
    results = response["results"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)
    assert all('title' in r and 'content' in r for r in results)

def test_format_results(web_searcher):
    results = [
        {"title": "Test1", "content": "Content1"},
        {"title": "Test2", "content": "Content2"}
    ]
    formatted = web_searcher.format_results(results)
    assert isinstance(formatted, str)
    assert "Test1" in formatted
    assert "Content1" in formatted

def test_search_caching(web_searcher):
    query = "What is machine learning?"
    # First call
    results1 = web_searcher.search(query)
    # Second call (should use cache)
    results2 = web_searcher.search(query)
    assert results1 == results2

@pytest.mark.skip(reason="Only run when testing error handling")
def test_invalid_api_key():
    with pytest.raises(ValueError):
        WebSearcher(api_key="invalid_key")