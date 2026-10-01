# rag.py
# Core RAG logic: retrieve relevant chunks from ChromaDB,
# send them + the question to Gemini, and return the answer.
#
# Includes a model fallback system: if the primary Gemini model is
# unavailable (503) or fails, it automatically tries the next one.

import os                              # For reading environment variables
from google import genai               # NEW SDK (google-genai)
from google.genai import types         # For GenerateContentConfig
from dotenv import load_dotenv         # For loading the .env file
from database import collection        # Our ChromaDB collection

# Load variables from .env into the environment
load_dotenv()

# Create a Gemini client using our API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------------------
# MODEL FALLBACK LIST
# ----------------------------------------
# We try these models in order. If the first is busy (503), we try the next.
# The first entry comes from .env (GEMINI_MODEL), with a safe default.
FALLBACK_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),  # primary (from .env)
    "gemini-3.7-flash",                              # fallback #1
    "gemini-3.6-flash",                              # fallback #2
    "gemini-3.5-flash",                              # fallback #3
    "gemini-flash-latest",                           # fallback #4 (auto-latest)
    "gemini-2.5-flash",                              # fallback #5 (older)
]

# The system instruction tells the model how to behave
# The system instruction tells the model how to behave
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions strictly "
    "based on the provided context.\n\n"
    "FORMATTING RULES:\n"
    "- Answer directly. Do NOT start with phrases like 'Based on the context', "
    "'According to the document', or 'The context says'.\n"
    "- Use markdown formatting: **bold** for key terms, bullet lists for items.\n"
    "- Keep answers concise and well-structured.\n"
    "- If the answer is not in the context, say 'I don't know based on the "
    "provided document.' Do not make up information."
)


def retrieve_context(question: str, session_id: str = "default", top_k: int = 3) -> list[str]:
    """
    Find the most relevant chunks from ChromaDB for a given question,
    but ONLY from the given session_id (so users don't see each other's docs).
    """
    results = collection.query(
        query_texts=[question],               # Question (auto-embedded by ChromaDB)
        n_results=top_k,                      # How many top matches
        where={"session_id": session_id}      # Filter: only this session
    )
    # results["documents"] is a list-of-lists; we sent one query, so take [0]
    return results["documents"][0]


def generate_answer(question: str, session_id: str = "default") -> dict:
    """
    Full RAG flow: retrieve context -> ask Gemini -> return answer + sources.
    Tries each model in FALLBACK_MODELS until one succeeds.
    """
    # Step 1: Retrieve chunks ONLY from this session
    context_chunks = retrieve_context(question, session_id=session_id)

    # If nothing was found for this session, tell the user
    if not context_chunks:
        return {
            "answer": "I don't have any documents for this session yet. Please upload a file first.",
            "sources": []
        }

    # Step 2: Combine chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    # Step 3: Build the final prompt
    prompt = f"""Use the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # Step 4: Try each model until one succeeds
    last_error = None                  # Keep track of the last error we saw

    for model_name in FALLBACK_MODELS:
        try:
            # Attempt to generate content with this model
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )

            # If we reach this line, the call succeeded.
            # Return the answer + the chunks it was based on.
            return {
                "answer": response.text,
                "sources": context_chunks
            }

        except Exception as e:
            # This model failed (busy, 503, quota, etc.) - try the next one
            last_error = e
            print(f"[rag] Model {model_name} failed: {e}")
            continue

    # Step 5: Every model failed - return a friendly message
    return {
        "answer": (
            "All AI models are currently busy. "
            "Please try again in a moment.\n\n"
            f"(Last error: {last_error})"
        ),
        "sources": context_chunks
    }