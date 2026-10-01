# main.py
# The FastAPI entry point - defines our API endpoints.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import collection
from fastapi.responses import FileResponse        # For returning index.html

# Import our custom modules
from ingest import extract_text, chunk_text, store_chunks
from rag import generate_answer

# Create the FastAPI app
app = FastAPI(title="RAG Chatbot")

# ---------------------------
# CORS (Cross-Origin Resource Sharing)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production, replace "*" with your real domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Request/Response Models
# ---------------------------
class ChatRequest(BaseModel):
    """The body of a /chat request."""
    question: str                       # The user's question
    session_id: str = "default"         # Which session's docs to search


class ChatResponse(BaseModel):
    """The response from /chat."""
    answer: str                         # The generated answer
    sources: list[str]                  # The chunks used as context


# ---------------------------
# ROUTE 1: Serve the frontend UI at root
# ---------------------------
@app.get("/")
def serve_ui():
    """
    Serve the frontend HTML file at the root URL.
    Opening http://127.0.0.1:8000/ shows the chat UI.
    """
    return FileResponse("./index.html")


# ---------------------------
# ROUTE 2: Health check (moved from /)
# ---------------------------
@app.get("/health")
def home():
    """Health-check endpoint (used to verify the API is up)."""
    return {"message": "Chatbot backend is alive!"}


# ---------------------------
# ROUTE 3: Upload a document
# ---------------------------
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = "default"         # Optional query param
):
    """
    Accept a PDF, DOCX, or TXT file, extract its text,
    split it into chunks, and store it in ChromaDB under the given session.
    """
    try:
        file_bytes = await file.read()                          # Read into memory
        text = extract_text(file.filename, file_bytes)          # Extract text

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file."
            )

        chunks = chunk_text(text)                               # Split into chunks
        count = store_chunks(chunks, file.filename, session_id) # Store with session tag

        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_stored": count,
            "session_id": session_id
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ---------------------------
# ROUTE 4: Chat
# ---------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Accept a question, retrieve context (filtered by session_id),
    ask Gemini, return the answer.
    """
    try:
        result = generate_answer(request.question, session_id=request.session_id)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ---------------------------
# ROUTE 5: Reset the database (dev only)
# ---------------------------
@app.delete("/reset")
def reset_database():
    """
    Delete all stored chunks. Useful during development
    to start with a clean slate.
    """
    all_items = collection.get()            # Returns everything
    ids = all_items["ids"]                  # List of all chunk IDs

    if ids:
        collection.delete(ids=ids)          # Removes those chunks

    return {"message": f"Deleted {len(ids)} chunks."}

# ---------------------------
# ROUTE 6: Database status
# ---------------------------
@app.get("/status")
def database_status():
    """
    Return how many chunks are currently in ChromaDB.
    Used by the frontend to show whether documents are loaded.
    """
    all_items = collection.get()             # Fetch everything
    count = len(all_items["ids"])            # How many chunks we have
    sources = set()                          # Unique source filenames

    # Loop through metadata and collect unique source names
    for meta in all_items.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])

    return {
        "chunks": count,
        "sources": list(sources),            # e.g. ["cv.pdf", "notes.txt"]
        "empty": count == 0
    }