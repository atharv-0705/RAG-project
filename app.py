import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from create_database import create_database
from rag import ask

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG AI Assistant",
    description="Upload PDFs and ask questions powered by Mistral AI + ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int


class StatusResponse(BaseModel):
    status: str
    has_database: bool
    db_directory: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    """Serve the main UI."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=StatusResponse, tags=["System"])
async def health_check():
    """Check API health and database status."""
    has_db = os.path.exists(DB_DIR) and bool(os.listdir(DB_DIR))
    return StatusResponse(
        status="ok",
        has_database=has_db,
        db_directory=DB_DIR,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. This will replace the existing database and
    create a new ChromaDB vector store from the uploaded document.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Build vector database
    try:
        chunks = create_database(file_path, db_dir=DB_DIR)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database creation failed: {str(e)}")

    return UploadResponse(
        message="PDF processed and database created successfully.",
        filename=file.filename,
        chunks=chunks,
    )


@app.post("/ask", response_model=QueryResponse, tags=["Query"])
async def ask_question(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    Returns the AI-generated answer and the source chunks used.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, docs = ask(request.question, db_dir=DB_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    sources = [
        {
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "N/A"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.delete("/reset", tags=["Documents"])
async def reset_database():
    """Delete the current ChromaDB vector store and all uploads."""
    errors = []
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            errors.append(f"DB: {str(e)}")

    if os.path.exists(UPLOAD_DIR):
        try:
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        except Exception as e:
            errors.append(f"Uploads: {str(e)}")

    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    return JSONResponse(content={"message": "Database and uploads cleared successfully."})
