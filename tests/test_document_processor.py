import pytest
from pathlib import Path
from src.document_processor import DocumentProcessor
import tempfile

@pytest.fixture
def processor():
    return DocumentProcessor()

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_test_files(tmpdirname)
        yield tmpdirname

def create_test_files(temp_dir):
    test_files = {
        'test1.py': 'def test_function():\n    pass',
        'test2.md': '# Test Document\nThis is a test.',
        'nested/test3.txt': 'Nested test content'
    }
    
    for filepath, content in test_files.items():
        full_path = Path(temp_dir) / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

def test_load_directory(processor, temp_dir):
    docs = processor.load_directory(temp_dir)
    assert len(docs) == 3
    assert all(doc.page_content for doc in docs)

def test_process_documents(processor, temp_dir):
    docs = processor.load_directory(temp_dir)
    splits = processor.process_documents(docs)
    assert len(splits) >= len(docs)
    assert all(len(split.page_content) <= 1000 for split in splits)

def test_create_vectorstore(processor, temp_dir):
    docs = processor.load_directory(temp_dir)
    splits = processor.process_documents(docs)
    
    # Create a temporary directory for the vectorstore
    with tempfile.TemporaryDirectory() as vector_dir:
        vectorstore = processor.create_vectorstore(
            splits,
            persist_dir=vector_dir
        )
        
        results = vectorstore.similarity_search("test")
        assert len(results) > 0

def test_refresh_vectorstore(processor, temp_dir):
    docs = processor.load_directory(temp_dir)
    splits = processor.process_documents(docs)
    
    with tempfile.TemporaryDirectory() as vector_dir:
        vectorstore = processor.create_vectorstore(
            splits,
            persist_dir=vector_dir,
        )
        
        # Add new file
        new_file = Path(temp_dir) / "new_test.py"
        new_file.write_text("def new_function():\n    pass")
        
        processor.refresh_vectorstore(temp_dir, vectorstore)
        results = vectorstore.similarity_search("new_function")
        assert len(results) > 0