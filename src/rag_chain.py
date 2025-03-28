from dataclasses import dataclass, field
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

from .search import WebSearcher
from .document_processor import DocumentProcessor


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatContext:
    def __init__(self, max_messages: int = 10):
        self.messages: list[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str, timestamp: datetime | None = None):
        """Add a message with an optional specific timestamp."""
        message = Message(
            role=role, content=content, timestamp=timestamp or datetime.now()
        )
        self.messages.append(message)
        print(f"[DEBUG] Added message to context - Role: {role}")
        total_messages = len(self.messages)
        active_messages = min(total_messages, self.max_messages)
        print(
            f"[DEBUG] Context size - Active: {active_messages}/{self.max_messages}, Total stored: {total_messages}"
        )

    def get_context_string(self) -> str:
        return "\n".join(
            [
                f"{msg.role}: {msg.content}"
                for msg in self.messages[-self.max_messages :]
            ]
        )


class RAGChain:
    def __init__(
        self,
        vectorstore: Chroma,
        doc_processor: DocumentProcessor,
        model_name: str = "deepseek-r1:14b",
        k_docs: int = 3,
        temperature: float = 0.6,
        max_history: int = 5,
        project_description: str = "No project description provided.",
    ):
        self.vectorstore = vectorstore
        self.doc_processor = doc_processor
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.web_searcher = WebSearcher()
        self.chat_context = ChatContext(max_messages=max_history)
        self.k_docs = k_docs
        self.project_description = project_description
        self.rag_enabled = True

        # Initialize prompts as class attributes
        self.local_prompt = ChatPromptTemplate.from_template("""
        You are a helpful research assistant for a project associated with a Python codebase.

        About this project:
        {project_description}                                                     

        First check if you can answer using just the local codebase context or your own knowledge.
        If you can answer confidently using only this context, do so.
        If you need a web search or external knowledge to provide a complete answer, respond with exactly "NEED_WEB_SEARCH".
                                                             
        Previous conversation:
        {chat_history}
        
        Relevant code files (if any):
        {code_context}
        
        Question about the project: {question}
        
        When answering:
        1. Reference specific files and code structures you see in the context
        2. If you describe functionality, make sure it matches the actual implementation shown
        3. If you're unsure about something or can't find it in the context, say so
        4. Focus on the actual code implementation rather than making assumptions
        """)

        self.web_prompt = ChatPromptTemplate.from_template("""
        Answer the question using both the codebase context and web search results.
        Make sure to consider both sources of information in your response.
        
        About this project:
        {project_description}                                                  

        Previous conversation:
        {chat_history}
        
        Relevant code files:
        {code_context}
        
        Web results:
        {web_results}
        
        Question: {question}
        """)

        self.conversation_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Use the conversation history and your knowledge to provide informed responses.
        
        About this project:
        {project_description}
        
        Previous conversation:
        {chat_history}
        
        Question: {question}
        """)

        print(f"[DEBUG] Initialized RAGChain with {max_history} max history")
        self.chain = self._build_chain()

    def _should_search_codebase(self, question: str) -> bool:
        """Determine if a question likely needs codebase context."""
        # Keywords that suggest code-related queries
        code_indicators = {
            "code",
            "file",
            "function",
            "class",
            "method",
            "implementation",
            "module",
            "import",
            "variable",
            "define",
            "declaration",
            "return",
            "parameter",
            "argument",
            "error",
            "bug",
            "issue",
            "fix",
            ".py",
            "python",
            "script",
        }

        question_lower = question.lower()

        # Check for code-related keywords
        has_code_keywords = any(
            indicator in question_lower for indicator in code_indicators
        )

        # Check for specific file mentions
        has_file_mention = any(
            filename.lower() in question_lower
            for filename in self.doc_processor.file_contents.keys()
        )

        return has_code_keywords or has_file_mention

    def _format_code_context(self, file_contents: dict[str, str]) -> str:
        """Format multiple files into a readable context."""
        if not file_contents:
            return "No relevant code files found."

        sections = []
        for filename, content in file_contents.items():
            sections.append(f"=== {filename} ===\n{content}\n")
        return "\n\n".join(sections)

    def _get_relevant_files(self, question: str) -> dict[str, str]:
        """Get relevant files based on the question."""
        # First check if any specific files are mentioned
        lower_question = question.lower()
        mentioned_files = {}

        # Look for specifically mentioned files first
        for filename in self.doc_processor.file_contents.keys():
            if filename.lower() in lower_question:
                content = self.doc_processor.get_full_content(filename)
                if content:
                    mentioned_files[filename] = content

        # If specific files were mentioned, prioritize those
        if mentioned_files:
            return mentioned_files

        # Otherwise, use vector similarity to find relevant files
        docs = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k_docs}
        ).invoke(question)

        relevant_files = {}
        for doc in docs:
            filename = doc.metadata.get("file_name")
            if filename and doc.metadata.get("full_content_available"):
                content = self.doc_processor.get_full_content(filename)
                if content:
                    relevant_files[filename] = content

        return relevant_files

    def process_response(self, inputs: dict) -> str:
        try:
            print("[DEBUG] Processing response...")
            chat_history = self.chat_context.get_context_string()

            # Store the user's question
            self.chat_context.add_message("user", inputs["question"])

            if not self.rag_enabled:
                # Simple conversation mode without RAG or web search
                print("[DEBUG] RAG disabled, using conversation-only mode")
                conversation_chain = (
                    self.conversation_prompt | self.model | StrOutputParser()
                )
                final_response = conversation_chain.invoke(
                    {
                        "question": inputs["question"],
                        "chat_history": chat_history,
                        "project_description": self.project_description,
                    }
                )
            else:
                # Only search codebase if RAG is enabled and question seems code-related
                code_context = "No code context needed for this question."
                if self._should_search_codebase(inputs["question"]):
                    print(
                        "[DEBUG] Question appears code-related, searching codebase..."
                    )
                    relevant_files = self._get_relevant_files(inputs["question"])
                    code_context = self._format_code_context(relevant_files)
                else:
                    print(
                        "[DEBUG] Question doesn't appear code-related, skipping codebase search"
                    )

                local_chain = self.local_prompt | self.model | StrOutputParser()
                local_response = local_chain.invoke(
                    {
                        "question": inputs["question"],
                        "code_context": code_context,
                        "chat_history": chat_history,
                        "project_description": self.project_description,
                    }
                )

                if "NEED_WEB_SEARCH" in local_response:
                    print(
                        "[DEBUG] Local context insufficient, performing web search..."
                    )
                    try:
                        web_results = self.web_searcher.search(inputs["question"])
                        formatted_results = self.web_searcher.format_results(
                            web_results
                        )

                        web_chain = self.web_prompt | self.model | StrOutputParser()
                        final_response = web_chain.invoke(
                            {
                                "question": inputs["question"],
                                "code_context": code_context,
                                "web_results": formatted_results,
                                "chat_history": chat_history,
                                "project_description": self.project_description,
                            }
                        )
                    except Exception as e:
                        final_response = f"Error during web search: {str(e)}"
                else:
                    final_response = local_response

            self.chat_context.add_message("assistant", final_response)
            return final_response

        except Exception as e:
            error_msg = f"Error processing response: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    def toggle_rag(self) -> bool:
        """Toggle RAG mode on/off."""
        self.rag_enabled = not self.rag_enabled
        return self.rag_enabled

    def get_rag_status(self) -> bool:
        """Get current RAG mode status."""
        return self.rag_enabled

    def _build_chain(self):
        chain = RunnablePassthrough() | self.process_response | StrOutputParser()
        return chain

    def __call__(self, question: str) -> str:
        """Process a question and return the response"""
        print(f"[DEBUG] Processing question: {question[:50]}...")
        response = self.chain.invoke({"question": question})
        print("[DEBUG] Response generated")
        return response
