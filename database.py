# database.py
# This file sets up ChromaDB - our local vector database.
# ChromaDB stores data in files (like SQLite), so no external server is needed.

import chromadb  # Import the ChromaDB library

# Create a "persistent client" - this means data is saved to disk,
# not lost when the program closes.
# 'path' tells ChromaDB where to store its files (our vector_db folder).
client = chromadb.PersistentClient(path="./vector_db")

# Get or create a "collection". A collection is like a table in SQL -
# it's where our document chunks and their vectors will live.
# If it doesn't exist, ChromaDB creates it automatically.
collection = client.get_or_create_collection(
    name="documents",  # The name of our collection
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
    # Cosine similarity = a standard way to measure how "close" two vectors are
)

# This 'collection' object is what other files will import and use
# to add documents or search them.