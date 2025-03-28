from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.rag_chain import RAGChain
from src.utils import load_config, ensure_directory

load_dotenv()


class ChatSession:
    def __init__(self, config_path: str | Path = "./config.yaml"):
        self.config = load_config(config_path)
        self.processor = self._initialize_processor()
        self.vectorstore = self._initialize_vectorstore()
        self.chain = self._initialize_chain()
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        print("[DEBUG] Chat session initialized with ID:", self.session_id)

    def _initialize_processor(self) -> DocumentProcessor:
        print("[DEBUG] Initializing document processor...")
        return DocumentProcessor()

    def _initialize_vectorstore(self):
        print("[DEBUG] Loading documents from:", self.config["codebase_path"])
        docs = self.processor.load_directory(self.config["codebase_path"])
        print(f"[DEBUG] Loaded {len(docs)} documents")

        # No more splits needed - docs are already processed
        persist_dir = ensure_directory(self.config["persist_directory"])
        return self.processor.create_vectorstore(docs, persist_dir)

    def _initialize_chain(self):
        print("[DEBUG] Initializing RAG chain...")
        return RAGChain(
            vectorstore=self.vectorstore,
            doc_processor=self.processor,
            model_name=self.config["model_name"],
            k_docs=self.config["k_docs"],
            temperature=0.6,
            max_history=10,
            project_description=self.config.get(
                "project_description", "No project description provided."
            ),
        )

    def refresh_context(self):
        """Refresh the vectorstore with latest changes from codebase"""
        print("[DEBUG] Refreshing vectorstore...")
        self.processor.refresh_vectorstore(
            self.config["codebase_path"], self.vectorstore
        )
        print("\nVectorstore refreshed with latest changes")

    def save_session(self):
        """Save the current chat session history with ISO format timestamps"""
        history_dir = ensure_directory("./chat_history")
        history_file = history_dir / f"session_{self.session_id}.json"

        messages = self.chain.chat_context.messages
        print(f"[DEBUG] Saving {len(messages)} messages to session history")

        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "model_name": self.config["model_name"],
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),  # Save in ISO format
                }
                for msg in messages
            ],
        }

        with open(history_file, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"\nChat history saved to {history_file}")

    def load_session(self, session_id: str) -> bool:
        """Load a previous chat session preserving original timestamps"""
        history_file = Path("./chat_history") / f"session_{session_id}.json"
        if not history_file.exists():
            print(f"\nSession {session_id} not found")
            return False

        print(f"[DEBUG] Loading session from {history_file}")
        with open(history_file, "r") as f:
            session_data = json.load(f)

        # Clear existing context and load historical messages
        self.chain.chat_context.messages.clear()
        for msg in session_data["messages"]:
            # Parse the ISO format timestamp and preserve it
            timestamp = datetime.fromisoformat(msg["timestamp"])
            self.chain.chat_context.add_message(
                msg["role"],
                msg["content"],
                timestamp=timestamp,  # Pass the original timestamp
            )

        print(f"[DEBUG] Loaded {len(session_data['messages'])} messages")
        print(f"\nLoaded chat history from session {session_id}")
        return True

    def debug_context(self) -> str:
        """Return debug information about current context"""
        messages = self.chain.chat_context.messages
        debug_info = [
            "\n=== Chat Context Debug Info ===",
            f"Current session ID: {self.session_id}",
            f"Session start time: {self.session_start}",
            f"Messages in context: {len(messages)}",
            "\nMessage Timeline:",
        ]

        for i, msg in enumerate(messages, 1):
            debug_info.append(
                f"\n{i}. [{msg.timestamp}] {msg.role}:"
                f"\n   Content length: {len(msg.content)} chars"
                f"\n   Preview: {msg.content[:100]}..."
            )

        return "\n".join(debug_info)


def display_help():
    """Display available commands"""
    print("\nAvailable commands:")
    print("  /help     - Show this help message")
    print("  /refresh  - Refresh the vectorstore with latest changes")
    print("  /save     - Save current chat session")
    print("  /load ID  - Load a previous chat session by ID")
    print("  /clear    - Clear current chat context")
    print("  /debug    - Show debug information about current context")
    print("  /quit     - Exit the program")


def main():
    session = ChatSession()
    print("\nChat session initialized. Type /help for available commands.")

    while True:
        try:
            question = input("\nQuestion: ").strip()

            # Handle commands
            if question.startswith("/"):
                parts = question.split()
                command = parts[0].lower()

                if command == "/quit":
                    print("[DEBUG] Saving session before exit...")
                    session.save_session()
                    break
                elif command == "/help":
                    display_help()
                elif command == "/refresh":
                    session.refresh_context()
                elif command == "/save":
                    session.save_session()
                elif command == "/load" and len(parts) > 1:
                    session.load_session(parts[1])
                elif command == "/clear":
                    session.chain.chat_context.messages.clear()
                    print("\nChat context cleared")
                elif command == "/debug":
                    print(session.debug_context())
                else:
                    print("\nUnknown command. Type /help for available commands.")
                continue

            # Process regular questions
            if question:
                print("[DEBUG] Processing question...")
                response = session.chain(question)
                print("\nResponse:", response)
                print(
                    f"[DEBUG] Context size: {len(session.chain.chat_context.messages)} messages"
                )

        except KeyboardInterrupt:
            print("\nSaving session before exit...")
            session.save_session()
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
