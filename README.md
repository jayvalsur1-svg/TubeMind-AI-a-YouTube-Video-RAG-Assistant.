# TubeMind AI --- YouTube Video RAG Assistant

TubeMind AI is a YouTube transcript-based Retrieval-Augmented Generation
(RAG) assistant. It lets users ask questions about a YouTube video and
generates answers using relevant sections retrieved from the video's
transcript.

**Scope:** The backend works with available YouTube
transcripts/captions. It does not directly understand video frames,
visuals, or audio without transcripts.

## Features

-   Accepts a YouTube video URL or video ID.
-   Extracts available transcripts, with fallback attempts for manual,
    generated, or translated English captions.
-   Splits transcript text into chunks of 1,000 characters with 200
    characters of overlap.
-   Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
-   Stores transcript chunks in a per-video Chroma vector collection.
-   Retrieves the five most relevant chunks for a question.
-   Generates context-based answers using a Groq-hosted Qwen chat model.
-   Provides FastAPI endpoints for asking questions and checking backend
    health.
-   Caches retrievers in memory while the server is running.

## Architecture

``` text
YouTube URL / Video ID
        |
        v
Transcript Extraction
        |
        v
Text Chunking (1000 / 200)
        |
        v
Sentence Transformer Embeddings
        |
        v
ChromaDB Vector Store
        |
        v
Question from User
        |
        v
Retrieve Top 5 Relevant Chunks
        |
        v
LangChain RAG Chain
        |
        v
Groq Qwen Chat Model
        |
        v
Answer
```

### Workflow

1.  The backend identifies the video from its URL or ID.
2.  It attempts to retrieve an available transcript.
3.  The transcript is split into overlapping chunks.
4.  Each chunk is embedded and stored in a Chroma collection associated
    with the video.
5.  For each question, the retriever selects the five most relevant
    chunks.
6.  The retrieved context and question are passed through a LangChain
    chain to the Groq model.
7.  The API returns the answer, question, and video ID.

## Technology Stack

  Component               Technology
  ----------------------- ----------------------------------
  Language                Python
  Backend API             FastAPI
  RAG orchestration       LangChain
  Vector database         ChromaDB
  Embeddings              Sentence Transformers
  Embedding model         `all-MiniLM-L6-v2`
  LLM provider            Groq
  Chat model              `qwen/qwen3.8-27b`
  Transcript extraction   `youtube-transcript-api`
  Text splitting          `RecursiveCharacterTextSplitter`

## Project Structure

The exact repository layout may differ. A typical layout is:

``` text
TubeMind-AI/
├── main.py
├── frontend.html
├── .env
├── requirements.txt
└── README.md
```

Update this section to match the actual filenames in your repository.

## Installation

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository includes a `requirements.txt`:

``` bash
pip install -r requirements.txt
```

Otherwise, install the packages used by the backend and resolve versions
against your implementation:

``` bash
pip install fastapi uvicorn python-dotenv langchain langchain-core langchain-community langchain-groq langchain-chroma chromadb sentence-transformers youtube-transcript-api
```

### 4. Configure environment variables

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key
```

Use the exact environment-variable name expected by your Python code.
Never commit your real API key.

### 5. Run the backend

If the FastAPI application object is named `app` in `main.py`:

``` bash
uvicorn main:app --reload
```

Open the local address printed by Uvicorn. Interactive API documentation
is normally available at `http://127.0.0.1:8000/docs`.

Adjust the command if your entry-point filename or application variable
differs.

## API

### `GET /health`

Checks whether the backend is responding.

### `POST /ask`

Submits a question about a YouTube video. The endpoint accepts a video
URL or video ID and a question. Confirm the exact request field names
against the FastAPI request model in your source code.

Example request shape:

``` json
{
  "video_id": "YOUTUBE_VIDEO_ID",
  "question": "What are the main ideas explained in this video?"
}
```

Example response shape:

``` json
{
  "video_id": "YOUTUBE_VIDEO_ID",
  "question": "What are the main ideas explained in this video?",
  "answer": "The generated answer based on the retrieved transcript context."
}
```

These examples illustrate the intended shape; verify the exact schema in
your implementation.

## Current RAG Configuration

  Setting                                                      Value
  ----------------------- ------------------------------------------
  Chunk size                                                   1,000
  Chunk overlap                                                  200
  Retrieved chunks                                                 5
  Embedding model           `sentence-transformers/all-MiniLM-L6-v2`
  Chat model                                      `qwen/qwen3.8-27b`
  Temperature                                                    0.2
  Maximum output tokens                                        1,000

## Limitations and Considerations

-   **Transcript dependent:** Videos need an accessible transcript or
    caption track. Some videos may not have usable captions.
-   **No direct visual understanding:** The system does not inspect
    frames, slides, diagrams, or visual demonstrations.
-   **Transcript quality:** Incomplete or inaccurate captions can affect
    answer quality.
-   **Retrieval limitations:** Relevant information may be missed if it
    is not present in the retrieved chunks.
-   **In-memory cache:** Retriever cache data is held in process memory
    and may be lost when the server restarts.
-   **Deployment hardening:** Review CORS settings, input validation,
    request limits, and API-key handling before exposing the service
    publicly.

## Possible Future Improvements

-   Add transcript timestamps and source citations to answers.
-   Add reranking and metadata-aware retrieval.
-   Implement persistent caching.
-   Add rate limiting and stricter CORS configuration.
-   Improve multilingual transcript handling.
-   Build an evaluation set to measure retrieval relevance and answer
    quality.
-   Explore multimodal video understanding as a separate capability.

## Author

**Jay Valsur**

-   GitHub: [jayvalsur1-svg](https://github.com/jayvalsur1-svg)
-   LinkedIn: [Jay
    Valsur](https://www.linkedin.com/in/jay-valsur-40b4b7381/)

------------------------------------------------------------------------

Review the setup instructions, API schema, and project structure against
the current repository before publishing this README.
