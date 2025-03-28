from pathlib import Path
from typing import Optional
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PythonLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_chroma import Chroma
from .embeddings import get_embeddings


class DocumentProcessor:
    """Processes documents while maintaining complete file context."""
    
    def __init__(self):
        self.file_contents = {}  # Cache of full file contents
        
    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """Load all supported files from a directory."""
        dir_path = Path(dir_path)
        documents = []
        
        # Define file type handlers
        handlers = {
            "**/*.py": self._process_python_file,
            "**/*.md": self._process_markdown_file,
            "**/*.txt": self._process_text_file
        }
        
        # Process each file type
        for glob_pattern, handler in handlers.items():
            for file_path in dir_path.glob(glob_pattern):
                try:
                    doc = handler(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def _process_python_file(self, file_path: Path) -> Optional[Document]:
        """Process a Python file, keeping complete context."""
        try:
            # Use PythonLoader to get initial document with metadata
            loader = PythonLoader(str(file_path))
            doc = loader.load()[0]
            
            # Store the complete file content
            self.file_contents[file_path.name] = doc.page_content
            
            # Create a searchable summary
            summary = self._create_file_summary(doc.page_content, file_path.name)
            
            return Document(
                page_content=summary,
                metadata={
                    "file_type": "python",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "is_summary": True,
                    "full_content_available": True
                }
            )
        except Exception as e:
            print(f"Error in Python processing {file_path}: {e}")
            return None
    
    def _process_markdown_file(self, file_path: Path) -> Optional[Document]:
        """Process a Markdown file."""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            doc = loader.load()[0]
            self.file_contents[file_path.name] = doc.page_content
            
            # For markdown, use first few lines and headers as summary
            import re
            headers = re.findall(r'^#{1,6}\s+.+', doc.page_content, re.MULTILINE)
            first_para = doc.page_content.split('\n\n')[0]
            summary = f"File: {file_path.name}\n{'=' * (len(file_path.name) + 6)}\n\n"
            if headers:
                summary += "Headers:\n" + "\n".join(headers) + "\n\n"
            summary += f"Preview:\n{first_para[:500]}..."
            
            return Document(
                page_content=summary,
                metadata={
                    "file_type": "markdown",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "is_summary": True,
                    "full_content_available": True
                }
            )
        except Exception as e:
            print(f"Error in Markdown processing {file_path}: {e}")
            return None
    
    def _process_text_file(self, file_path: Path) -> Optional[Document]:
        """Process a text file."""
        try:
            loader = TextLoader(str(file_path))
            doc = loader.load()[0]
            self.file_contents[file_path.name] = doc.page_content
            
            # For text files, use first few lines as summary
            summary = f"File: {file_path.name}\n{'=' * (len(file_path.name) + 6)}\n\n"
            summary += f"Preview:\n{doc.page_content[:500]}..."
            
            return Document(
                page_content=summary,
                metadata={
                    "file_type": "text",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "is_summary": True,
                    "full_content_available": True
                }
            )
        except Exception as e:
            print(f"Error in text processing {file_path}: {e}")
            return None
    
    def _create_file_summary(self, content: str, file_name: str) -> str:
        """Create a searchable summary of Python file's key elements."""
        import ast
        
        try:
            tree = ast.parse(content)
            elements = []
            
            # Start with the filename
            file_header = f"File: {file_name}\n{'=' * (len(file_name) + 6)}"
            elements.append(file_header)
            
            # Get imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            if imports:
                elements.append("Imports:\n" + "\n".join(imports))
            
            # Get class definitions with their methods
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_summary = [f"class {node.name}:"]
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Include function args for better searchability
                            args = ast.unparse(item.args)
                            args = args[args.find('('):args.find(')')+1]
                            methods.append(f"    def {item.name}{args}")
                    class_summary.extend(methods)
                    elements.append("\n".join(class_summary))
                
                # Get top-level functions
                elif isinstance(node, ast.FunctionDef):
                    args = ast.unparse(node.args)
                    args = args[args.find('('):args.find(')')+1]
                    elements.append(f"def {node.name}{args}")
            
            return "\n\n".join(elements)
            
        except Exception as e:
            print(f"Error creating summary for {file_name}: {e}")
            return f"File: {file_name}\n\nError parsing file: {str(e)}"
    
    def get_full_content(self, file_name: str) -> Optional[str]:
        """Retrieve the complete content of a file."""
        return self.file_contents.get(file_name)
    
    def create_vectorstore(
        self, 
        docs: list[Document], 
        persist_dir: Optional[str | Path] = None
    ) -> Chroma:
        """Create a vectorstore with the file summaries."""
        kwargs = {"persist_directory": str(persist_dir)} if persist_dir else {}
        
        return Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
            **kwargs
        )
    
    def refresh_vectorstore(self, dir_path: str | Path, vectorstore: Chroma) -> None:
        """Refresh the vectorstore with latest changes."""
        docs = self.load_directory(dir_path)
        vectorstore.add_documents(docs)