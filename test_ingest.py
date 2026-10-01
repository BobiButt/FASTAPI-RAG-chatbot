# test_ingest.py
# A quick test to make sure ingestion works end-to-end.

from ingest import extract_text, chunk_text, store_chunks

# Fake a text file (as raw bytes) - same format an upload would give us
sample_text = b"This is a test document. " * 50  # 50 copies = ~1350 chars

# 1. Extract text
text = extract_text("sample.txt", sample_text)
print(f"Extracted {len(text)} characters")

# 2. Chunk it
chunks = chunk_text(text, chunk_size=200, overlap=20)
print(f"Created {len(chunks)} chunks")

# 3. Store in ChromaDB
count = store_chunks(chunks, "sample.txt")
print(f"Stored {count} chunks in ChromaDB")