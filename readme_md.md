# RAG Chatbot 🤖📄

> A document-aware chatbot that reads your **PDF**, **DOCX**, and **TXT** files and answers questions **strictly based on their content** — no guessing, no hallucinations.

Built with **FastAPI**, **ChromaDB**, and **Google Gemini** using a **Retrieval-Augmented Generation (RAG)** pipeline.

🔗 **Repository:** [github.com/BobiButt/FASTAPI-RAG-chatbot](https://github.com/BobiButt/FASTAPI-RAG-chatbot)

## 👋 About the Author

Hi, I'm **Mubeen Butt** (aka **BOBI**) — a full-stack web developer who loves building tools that make AI feel practical and personal.

* 🌐 **Full-Stack Developer:** Laravel, PHP, JavaScript, React
* 🐍 **Python Enthusiast:** FastAPI, Data pipelines
* 💡 **Motivation:** I built this project to demonstrate how a compact RAG pipeline can turn any document into an interactive, question-answering assistant.

Feel free to fork, star, or reach out!

* 🐙 **GitHub:** [@BobiButt](https://github.com/BobiButt)

## 📖 About the Project

**RAG Chatbot** lets you upload documents and ask questions about them in plain English. Behind the scenes, it executes a clean, local pipeline:

1. **Extraction:** Extracts raw text from your uploaded **PDF / DOCX / TXT** file.
2. **Chunking:** Splits the text into manageable overlapping blocks.
3. **Embedding:** Converts each chunk into a vector using a local embedding model (`all-MiniLM-L6-v2`).
4. **Storage:** Saves vectors and metadata in a **local ChromaDB** database.
5. **Retrieval & Generation:** When queried, it finds the most contextually similar chunks and passes them to **Google Gemini** to formulate an accurate answer.

### ✨ Features

| Feature | Description |
| :--- | :--- |
| 📤 **Multi-Format Upload** | PDF, DOCX, and TXT supported out of the box |
| 🔍 **Semantic Search** | Finds relevant content by **meaning**, not just keyword matching |
| 🤖 **Gemini-Powered** | Leverages Google's latest Gemini models |
| 🔁 **Smart Model Fallback** | Automatically handles and switches through built-in fallback models if one encounters traffic limits (503) |
| 🌑 **Dark-Themed UI** | Modern chat interface featuring rich markdown rendering |
| 📊 **Live Status Bar** | Tracks loaded chunk counts and active file references |
| 🗑️ **Quick Reset** | Wipe the vector database clean with a single click |
| 💾 **Local Vector DB** | ChromaDB keeps data locally on disk — no external server fees |
| 🔒 **Session Isolation** | Optional `session_id` to separate document contexts |
| 📝 **Markdown Replies** | Bot outputs support bolding, lists, headings, and inline code blocks |

## 🧠 How It Works (RAG Pipeline)

RAG = **Retrieval-Augmented Generation**. Instead of relying on pre-trained memory alone, the LLM answers strictly using the factual content of your documents.

### 📥 Ingestion Flow (Upload)

```
[User Uploads File] ➔ [Extract Text] ➔ [Split into Chunks] ➔ [Vectorize Embeddings] ➔ [Store in ChromaDB]
```

### 💬 Chat Flow (Query)

```
[User Submits Question] ➔ [Vectorize Query] ➔ [Retrieve Top Chunks] ➔ [Build Prompt] ➔ [Gemini Generation]
```

## 🚀 Quick Start Guide

### 🐧 Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/BobiButt/FASTAPI-RAG-chatbot.git
cd FASTAPI-RAG-chatbot/backend

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Configure your environment variables
nano .env  # Add your GEMINI_API_KEY (see Configuration section)

# 5. Launch the development server
uvicorn main:app --reload
```

### 🪟 Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/BobiButt/FASTAPI-RAG-chatbot.git
cd FASTAPI-RAG-chatbot\backend

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Configure your environment variables
notepad .env  # Add your GEMINI_API_KEY

# 5. Launch the development server
uvicorn main:app --reload
```

> **Tip:** Type `deactivate` whenever you want to exit your virtual environment.

### 🌐 Accessing the Application

* **Chat Interface:** http://127.0.0.1:8000/
* **Interactive API Docs (Swagger):** http://127.0.0.1:8000/docs

## ⚙️ Configuration (`.env`)

Create a file named `.env` inside the `backend/` folder containing your compulsory API key:

```env
GEMINI_API_KEY=your_google_ai_studio_key_here
```

*(Note: Model selection and failover options are handled directly via built-in fallback routines in code.)*

Get your **free** Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

> ⚠️ **Security Note:** Never commit your `.env` file. It is already excluded via `.gitignore`.

## 📦 Requirements & Dependencies

The `requirements.txt` file includes:

```
fastapi
uvicorn
python-multipart
python-dotenv
google-genai
chromadb
pypdf
python-docx
```

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page chat UI |
| `GET` | `/health` | System health check |
| `GET` | `/status` | Returns active chunk counts and loaded files |
| `POST` | `/upload` | Upload a document (`multipart/form-data`) |
| `POST` | `/chat` | Submit a query (`JSON body`) |
| `DELETE` | `/reset` | Clear all data chunks from the vector database |

## 💡 Example Usage (cURL)

**1. Upload a document:**

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@my_cv.pdf"
```

**2. Query the document:**

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are this candidates core skills?"}'
```

## 🛠 Tech Stack Breakdown

| Layer | Technology Choice |
| :--- | :--- |
| **Web Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **AI / LLM** | Google Gemini (`google-genai` SDK) |
| **Vector Store** | ChromaDB (Persistent, local) |
| **Embeddings** | `all-MiniLM-L6-v2` (Local execution) |
| **Parsing Engines** | `pypdf`, `python-docx` |
| **Frontend UI** | HTML5, Modern CSS, Vanilla JavaScript |

## 📁 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI app & routing entrypoint
│   ├── rag.py               # Core retrieval logic, automated fallbacks & Gemini wrapper
│   ├── ingest.py            # Text parsing, chunking & vector insertion
│   ├── database.py          # ChromaDB connection & client configuration
│   ├── requirements.txt     # Python package declarations
│   ├── .env                 # Environment variables (ignored)
│   ├── .gitignore           # Ignored files configuration
│   ├── vector_db/           # Local ChromaDB persistent storage directory
│   └── data/                # Document storage directory
└── frontend/
    └── index.html           # Unified chat interface layout
```

## 🐛 Troubleshooting Guide

| Issue | Resolution |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'docx'` | Ensure your virtual environment is active, then run `pip install python-docx`. |
| `502 / 503 Model Unavailable` | Google API servers are busy; the fallback handler inside `rag.py` will automatically retry with backup models. |
| `Status shows 'No documents loaded'` | Upload a valid document first before querying the chat interface. |
| Windows: `Scripts\Activate.ps1 cannot be loaded` | Run PowerShell as Administrator and execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |

## 🗺️ Roadmap & Future Plans

* [x] Full functional RAG pipeline
* [x] Multi-format file support (PDF, DOCX, TXT)
* [x] Dark-mode chat UI with complete markdown parsing
* [x] Live system status bar and database reset controls
* [x] Programmatic model fallback for high traffic protection
* [ ] Token-by-token streaming response integration
* [ ] Inline source citations displayed under outputs
* [ ] User authentication and dedicated vault sessions
* [ ] Containerization via Docker

## 📜 License

Distributed under the **MIT License**. Feel free to use, modify, and distribute.

⭐ **If this project helped you out, consider leaving a star on GitHub!**