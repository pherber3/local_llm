from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from datetime import datetime
from pathlib import Path
import uvicorn
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
logger.info(f"Added project root to path: {PROJECT_ROOT}")

try:
    from main import ChatSession

    logger.info("Successfully imported ChatSession")
except ImportError as e:
    logger.error(f"Failed to import ChatSession: {e}")
    raise

app = FastAPI()

# Mount static files
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

# Store active chat sessions
chat_sessions = {}


@app.get("/")
async def get_html():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


async def handle_command(
    websocket: WebSocket, command: str, chat_session: ChatSession, data: dict = None
):
    """Handle different command types."""
    try:
        if command == "help":
            help_text = """Available commands:\n
            /refresh  - Refresh the context with latest changes\n
            /save    - Save current chat session\n
            /load    - Load a previous chat session\n
            /debug   - Show debug information
            /toggle_rag - Toggle between RAG and conversation-only modes
            """
            await websocket.send_json(
                {
                    "type": "system",
                    "content": help_text,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        elif command == "refresh":
            chat_session.refresh_context()
            await websocket.send_json(
                {
                    "type": "system",
                    "content": "Context refreshed with latest changes",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        elif command == "save":
            chat_session.save_session()
            await websocket.send_json(
                {
                    "type": "system",
                    "content": "Session saved successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        elif command == "load":
            if not data:
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": "No session data provided",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            try:
                logger.info(f"Loading session data: {data}")

                # Extract messages from the session data
                messages_to_load = data.get("messages", [])
                logger.info(f"Found {len(messages_to_load)} messages to load")

                # Clear existing messages
                chat_session.chain.chat_context.messages.clear()

                # Load messages from provided session data
                for msg in messages_to_load:
                    try:
                        timestamp = datetime.fromisoformat(msg["timestamp"])
                        chat_session.chain.chat_context.add_message(
                            msg["role"], msg["content"], timestamp=timestamp
                        )
                        logger.info(f"Loaded message: {msg['role']} at {timestamp}")
                    except Exception as e:
                        logger.error(f"Error loading individual message: {str(e)}")

                # Get all loaded messages
                loaded_messages = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                    for msg in chat_session.chain.chat_context.messages
                ]

                logger.info(f"Sending {len(loaded_messages)} messages back to client")

                # First send confirmation
                await websocket.send_json(
                    {
                        "type": "system",
                        "content": f"Successfully loaded {len(loaded_messages)} messages from session {data.get('session_id')}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Then send the session data
                await websocket.send_json(
                    {
                        "type": "session_loaded",
                        "messages": loaded_messages,
                        "session_info": {
                            "session_id": data.get("session_id"),
                            "start_time": data.get("start_time"),
                            "model_name": data.get("model_name"),
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                error_msg = f"Error loading session: {str(e)}"
                logger.error(error_msg)
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        elif command == "debug":
            debug_info = chat_session.debug_context()
            await websocket.send_json(
                {
                    "type": "system",
                    "content": debug_info,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        elif command == "toggle_rag":
            new_state = chat_session.chain.toggle_rag()
            await websocket.send_json(
                {
                    "type": "rag_status",
                    "enabled": new_state,
                    "content": f"RAG mode {'enabled' if new_state else 'disabled'}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "content": f"Unknown command: {command}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        logger.error(f"Error handling command {command}: {str(e)}")
        await websocket.send_json(
            {
                "type": "error",
                "content": f"Error executing command: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }
        )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logger.info(f"New WebSocket connection request for session {session_id}")
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session {session_id}")

    # Initialize chat session if it doesn't exist
    if session_id not in chat_sessions:
        logger.info(f"Creating new ChatSession for {session_id}")
        try:
            chat_sessions[session_id] = ChatSession()
            logger.info("ChatSession created successfully")
            
            # Send combined initial status
            await websocket.send_json({
                "type": "init",
                "rag_enabled": chat_sessions[session_id].chain.get_rag_status(),
                "content": "Connected to server. Ready to chat!",
                "timestamp": datetime.now().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error creating ChatSession: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "content": f"Failed to create chat session: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            })
            await websocket.close()
            return

    chat_session = chat_sessions[session_id]

    try:
        while True:
            # Wait for messages
            message = await websocket.receive_text()
            logger.info(f"Received message: {message}")

            try:
                data = json.loads(message)
                logger.info(f"Parsed message data: {data}")

                if data["type"] == "message":
                    try:
                        logger.info(f"Processing message: {data['content']}")
                        response = chat_session.chain(data["content"])
                        logger.info(
                            f"Got response from model: {response[:100]}..."
                        )  # Log first 100 chars

                        await websocket.send_json(
                            {
                                "type": "response",
                                "content": response,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        logger.info("Response sent back to client")
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        await websocket.send_json(
                            {
                                "type": "error",
                                "content": f"Error processing message: {str(e)}",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                elif data["type"] == "command":
                    logger.info(f"Processing command: {data['command']}")
                    command_data = data.get("data")  # Get additional data if provided
                    await handle_command(
                        websocket, data["command"], chat_session, command_data
                    )

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message as JSON: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": "Invalid message format",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # Clean up session on error
        if session_id in chat_sessions:
            del chat_sessions[session_id]


if __name__ == "__main__":
    # Ensure required directories exist
    Path("chat_history").mkdir(exist_ok=True)

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
