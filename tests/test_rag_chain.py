import pytest
from unittest.mock import patch, MagicMock
from src.rag_chain import RAGChain
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

@pytest.fixture
def mock_vectorstore():
    store = MagicMock()
    store.as_retriever.return_value = MagicMock()
    store.as_retriever().get_relevant_documents.return_value = [
        MagicMock(page_content="Test document content")
    ]
    return store

@pytest.fixture
def mock_web_results():
    return [{"title": "Web Result 1", "content": "Web content 1"}]

def test_chain_initialization(mock_vectorstore):
    chain = RAGChain(mock_vectorstore)
    assert chain.vectorstore == mock_vectorstore
    assert chain.model is not None

@patch("langchain_community.chat_models.ChatOllama")
def test_local_only_query(MockChatOllama, mock_vectorstore):
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = AIMessage(content="Local response")
    MockChatOllama.return_value = mock_chat

    chain = RAGChain(mock_vectorstore)
    response = chain("test question")
    assert "Local response" in response

@patch("langchain_community.chat_models.ChatOllama")
def test_web_search_query(MockChatOllama, mock_vectorstore, mock_web_results):
    # Mock the chat model responses
    mock_chat = MagicMock()
    mock_chat.invoke.side_effect = [
        AIMessage(content="NEED_WEB_SEARCH"),
        AIMessage(content="Combined response")
    ]
    MockChatOllama.return_value = mock_chat

    chain = RAGChain(mock_vectorstore)
    response = chain("test question")
    assert "Combined response" in response

def test_chain_error_handling(mock_vectorstore):
    chain = RAGChain(mock_vectorstore)
    mock_vectorstore.as_retriever().get_relevant_documents.side_effect = Exception(
        "Test error"
    )
    with pytest.raises(Exception):
        chain("test question")