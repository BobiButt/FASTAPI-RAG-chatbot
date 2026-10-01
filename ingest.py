# ingest.py
# This file handles: reading uploaded files -> extracting text -> splitting into chunks

import os                          # For file path operations
import io                          # For handling file streams in memory
from pypdf import PdfReader        # For reading PDF files
from docx import Document          # For reading DOCX files (from python-docx)
from database import collection    # Our ChromaDB collection from database.py

# ----------------------------------------
# PART 1: EXTRACT TEXT FROM DIFFERENT FILE TYPES
# ----------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file (given as raw bytes)."""
    # Wrap the raw bytes in a BytesIO stream so PdfReader can read it
    pdf_stream = io.BytesIO(file_bytes)
    # Create a PDF reader object
    reader = PdfReader(pdf_stream)
    # Loop through every page and collect its text
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # 'or ""' handles blank pages
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract all text from a DOCX file."""
    # Wrap the raw bytes in a stream
    docx_stream = io.BytesIO(file_bytes)
    # Open the DOCX document
    doc = Document(docx_stream)
    # Loop through every paragraph and collect its text
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"  # Add a newline after each paragraph
    return text


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain TXT file."""
    # Decode the bytes into a UTF-8 string. If some characters fail,
    # 'ignore' tells Python to skip them instead of crashing.
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Route the file to the correct extractor based on its extension."""
    # Get the file extension in lowercase (e.g., ".pdf")
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        # If it's not one of our supported types, raise an error
        raise ValueError(f"Unsupported file type: {ext}")


# ----------------------------------------
# PART 2: SPLIT TEXT INTO CHUNKS
# ----------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text into smaller overlapping chunks.
    
    Why overlap? Because if we cut in the middle of a sentence,
    the meaning could be lost. Overlapping ensures context isn't broken
    between chunks.
    """
    chunks = []                     # Empty list to hold our chunks
    start = 0                       # Where the current chunk starts
    text_length = len(text)         # Total length of the text

    # Keep looping until we've covered the whole text
    while start < text_length:
        end = start + chunk_size    # Where this chunk ends
        chunk = text[start:end]     # Slice out the chunk
        chunks.append(chunk)        # Add it to our list
        start = end - overlap       # Move forward, but back up by 'overlap'

    return chunks                   # Return list of chunks


# ----------------------------------------
# PART 3: STORE CHUNKS IN CHROMADB
# ----------------------------------------

# ----------------------------------------
# PART 3: STORE CHUNKS IN CHROMADB
# ----------------------------------------

def store_chunks(chunks: list[str], source_name: str, session_id: str = "default") -> int:
    """
    Store a list of text chunks in ChromaDB.
    - source_name: the original filename (e.g., "cv.pdf")
    - session_id:  which user/session this belongs to (for filtering later)
    Returns the number of chunks stored.
    """
    # Build unique IDs for each chunk.
    # We include session_id so two users uploading "cv.pdf" don't collide.
    ids = [f"{session_id}_{source_name}_chunk_{i}" for i in range(len(chunks))]

    # Metadata stored with each chunk - this is what we filter by later
    metadatas = [
        {"source": source_name, "session_id": session_id}
        for _ in chunks
    ]

    # Add everything to ChromaDB (it auto-generates embeddings for us)
    collection.add(
        documents=chunks,    # The actual text of each chunk
        ids=ids,             # Unique IDs (now session-scoped)
        metadatas=metadatas  # Extra info (source file + session)
    )

    return len(chunks)