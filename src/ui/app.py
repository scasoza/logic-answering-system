"""
FastAPI Web Application for the Logic Answering System.

Provides a web interface with:
- Text input for questions
- Audio input with transcription
- Real-time tree view of the reasoning process
- Collapsible sections for exploring prompts/responses
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from src.core.models import SystemState
from src.pipelines.orchestrator import Orchestrator


# Store active sessions
active_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    yield
    # Cleanup on shutdown
    active_sessions.clear()


app = FastAPI(
    title="Logic Answering System",
    description="A reasoning system that uses LLMs to approach truth through iterative verification",
    lifespan=lifespan,
)

# Mount static files
static_path = Path(__file__).parent.parent.parent / "static"
templates_path = Path(__file__).parent.parent.parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.pending_queries: dict[str, asyncio.Queue] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[session_id] = websocket
        self.pending_queries[session_id] = asyncio.Queue()

    def disconnect(self, session_id: str):
        if session_id in self.connections:
            del self.connections[session_id]
        if session_id in self.pending_queries:
            del self.pending_queries[session_id]

    async def send_state(self, session_id: str, state: SystemState):
        """Send state update to a specific session."""
        if session_id in self.connections:
            try:
                await self.connections[session_id].send_json({
                    "type": "state_update",
                    "data": {
                        "status": state.context.status,
                        "tree_view": state.tree_view,
                        "cost_summary": state.cost_summary.model_dump(mode="json"),
                        "final_answer": state.context.final_answer,
                        "termination_reason": state.context.termination_reason,
                        "is_complete": state.is_complete,
                    }
                })
            except Exception:
                pass

    async def send_query(self, session_id: str, query_type: str, question: str) -> str:
        """Send a query to the user and wait for response."""
        if session_id not in self.connections:
            return ""

        try:
            await self.connections[session_id].send_json({
                "type": "user_query",
                "query_type": query_type,
                "question": question,
            })

            # Wait for response
            response = await self.pending_queries[session_id].get()
            return response
        except Exception:
            return ""

    async def receive_query_response(self, session_id: str, response: str):
        """Receive a response to a pending query."""
        if session_id in self.pending_queries:
            await self.pending_queries[session_id].put(response)


manager = WebSocketManager()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/question")
async def submit_question(
    question: str = Form(...),
    session_id: str = Form(...),
):
    """Submit a question for processing."""
    # Store session
    active_sessions[session_id] = {
        "question": question,
        "state": None,
    }

    return {"status": "accepted", "session_id": session_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(session_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "start_processing":
                question = data.get("question", "")
                if question:
                    # Start processing in background
                    asyncio.create_task(
                        process_question_with_updates(session_id, question)
                    )

            elif data.get("type") == "query_response":
                response = data.get("response", "")
                await manager.receive_query_response(session_id, response)

            elif data.get("type") == "continue_feedback":
                feedback = data.get("feedback", "")
                if session_id in active_sessions and active_sessions[session_id].get("state"):
                    asyncio.create_task(
                        continue_with_feedback(session_id, feedback)
                    )

    except WebSocketDisconnect:
        manager.disconnect(session_id)


async def process_question_with_updates(session_id: str, question: str):
    """Process a question and send updates via WebSocket."""

    async def on_state_update(state: SystemState):
        await manager.send_state(session_id, state)

    async def user_query_callback(query_type: str, question: str) -> str:
        return await manager.send_query(session_id, query_type, question)

    orchestrator = Orchestrator(
        user_query_callback=user_query_callback,
        on_state_update=on_state_update,
    )

    try:
        state = await orchestrator.process_question(question)
        active_sessions[session_id]["state"] = state
    except Exception as e:
        await manager.connections[session_id].send_json({
            "type": "error",
            "message": str(e),
        })


async def continue_with_feedback(session_id: str, feedback: str):
    """Continue processing with user feedback."""

    async def on_state_update(state: SystemState):
        await manager.send_state(session_id, state)

    async def user_query_callback(query_type: str, question: str) -> str:
        return await manager.send_query(session_id, query_type, question)

    orchestrator = Orchestrator(
        user_query_callback=user_query_callback,
        on_state_update=on_state_update,
    )

    state = active_sessions[session_id]["state"]
    if state:
        try:
            new_state = await orchestrator.continue_with_feedback(state, feedback)
            active_sessions[session_id]["state"] = new_state
        except Exception as e:
            await manager.connections[session_id].send_json({
                "type": "error",
                "message": str(e),
            })


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using Whisper."""
    try:
        import whisper
    except ImportError:
        return {
            "error": "Whisper not installed. Install with: pip install -r requirements-audio.txt"
        }

    try:
        import tempfile

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load whisper model (will be cached after first load)
        model = whisper.load_model("base")

        # Transcribe
        result = model.transcribe(tmp_path)

        # Clean up
        Path(tmp_path).unlink()

        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/logs")
async def get_logs():
    """Get list of cost log files."""
    logs_path = Path("logs")
    if not logs_path.exists():
        return {"logs": []}

    logs = []
    for f in logs_path.glob("*.json"):
        logs.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })

    return {"logs": sorted(logs, key=lambda x: x["modified"], reverse=True)}


@app.get("/api/logs/{log_name}")
async def get_log(log_name: str):
    """Get a specific log file."""
    log_path = Path("logs") / log_name
    if not log_path.exists():
        return {"error": "Log not found"}

    with open(log_path) as f:
        return json.load(f)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
