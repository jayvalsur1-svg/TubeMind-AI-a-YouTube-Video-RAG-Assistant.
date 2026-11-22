import os
import re
import sys
import socket
import threading
import webbrowser
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
    YouTubeTranscriptApiException
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="TubeMind RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("[WARNING] GROQ_API_KEY is not set in environment or .env file!")

groq_model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
model = ChatGroq(
    model=groq_model_name,
    groq_api_key=groq_api_key,
    max_tokens=1000,
    temperature=0.2
)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="""You are TubeMind AI, an expert YouTube video analyst.
Answer the user's question accurately and concisely based on the transcript context provided below.
If the context does not contain enough information to answer the question, state what is mentioned in the context and clarify what information is missing.

Transcript Context:
{context}

Question: {question}

Helpful Answer:""",
    input_variables=["context", "question"]
)

retriever_cache = {}

class QueryRequest(BaseModel):
    video_id: str
    question: str

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_video_id(input_str: str) -> str:
    """Extract standard 11-char YouTube ID from URL or raw ID string."""
    if not input_str:
        return ""
    clean = input_str.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", clean):
        return clean
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?|shorts|live)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, clean)
    if match:
        return match.group(1)
    return ""

def sanitize_collection_name(video_id: str) -> str:
    """Chroma collection names must be 3-63 chars: alphanumeric, _, - only, starting & ending with alphanumeric."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", video_id).strip("_")
    if not clean:
        clean = "video"
    name = f"yt_{clean}_rag"
    if len(name) > 63:
        name = name[:63].rstrip("_")
    return name

def fetch_transcript_text(video_id: str) -> str:
    """
    Fetch transcript using youtube-transcript-api.
    Supports direct English fetch, manual transcripts, generated transcripts,
    and automatic translation/fallback.
    """
    ytt = YouTubeTranscriptApi()
    transcript_data = None

    try:
        transcript_data = ytt.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        pass

    if transcript_data is None:
        try:
            transcript_list = ytt.list(video_id)
        except TranscriptsDisabled:
            raise HTTPException(
                status_code=404,
                detail=f"Captions/subtitles are disabled for video '{video_id}'."
            )
        except NoTranscriptFound:
            raise HTTPException(
                status_code=404,
                detail=f"No captions or subtitles found for video '{video_id}'."
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to access captions for video '{video_id}': {str(e)}"
            )

        transcript_obj = None

        try:
            transcript_obj = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except Exception:
            pass

        if transcript_obj is None:
            try:
                transcript_obj = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            except Exception:
                pass

        if transcript_obj is None:
            try:
                transcript_obj = transcript_list.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                pass

        if transcript_obj is None:
            first_t = next(iter(transcript_list), None)
            if first_t is not None:
                if getattr(first_t, "is_translatable", False):
                    try:
                        transcript_obj = first_t.translate("en")
                    except Exception:
                        transcript_obj = first_t
                else:
                    transcript_obj = first_t

        if transcript_obj is None:
            raise HTTPException(
                status_code=404,
                detail=f"No usable captions or translatable subtitles available for video '{video_id}'."
            )

        try:
            transcript_data = transcript_obj.fetch()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch caption data for video '{video_id}': {str(e)}"
            )

    parts = []
    for item in transcript_data:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text", "")
        elif isinstance(item, str):
            text = item
        if text:
            parts.append(str(text))

    combined = " ".join(parts).strip()
    if not combined:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript content is empty for video '{video_id}'."
        )
    return combined

def get_retriever(video_id: str):
    if video_id in retriever_cache:
        return retriever_cache[video_id]

    transcript = fetch_transcript_text(video_id)
    chunks = splitter.create_documents([transcript])
    collection_name = sanitize_collection_name(video_id)

    try:
        vector_database = Chroma.from_documents(
            chunks,
            embedding,
            collection_name=collection_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build vector index for video '{video_id}': {str(e)}"
        )

    retriever = vector_database.as_retriever(search_kwargs={"k": 5})
    retriever_cache[video_id] = retriever
    return retriever

@app.get("/")
def serve_frontend():
    frontend_path = os.path.join(BASE_DIR, "frontend.html")
    if not os.path.exists(frontend_path):
        return HTMLResponse("<h2>frontend.html not found in server directory.</h2>", status_code=404)
    return FileResponse(frontend_path, media_type="text/html")

@app.get("/frontend.html")
def serve_frontend_alias():
    return serve_frontend()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TubeMind RAG API", "cached_videos": list(retriever_cache.keys())}

@app.post("/ask")
def ask_video(request: QueryRequest):
    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set in .env. Please configure your Groq API key to proceed."
        )

    clean_id = extract_video_id(request.video_id)
    if not clean_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube Video ID or URL. Please provide a valid 11-character video ID or YouTube URL."
        )

    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = get_retriever(clean_id)

    try:
        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        mainchain = parallel | prompt | model | parser
        result = mainchain.invoke(clean_question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer with Groq LLM: {str(e)}"
        )

    return {
        "video_id": clean_id,
        "question": clean_question,
        "answer": result
    }

def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_available_port(host: str, preferred_port: int = 8000) -> int:
    if is_port_available(host, preferred_port):
        return preferred_port
    for p in [8001, 8002, 8080, 8501, 5000]:
        if is_port_available(host, p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    host = "127.0.0.1"
    env_port = int(os.getenv("PORT", 8000))
    port = find_available_port(host, env_port)
    if port != env_port:
        print(f"[INFO] Port {env_port} is already in use. Automatically switched to available port {port}.")

    frontend_url = f"http://{host}:{port}/"
    print("=" * 64)
    print(" TubeMind AI - YouTube Video RAG Assistant")
    print(f" * Backend API:   http://{host}:{port}")
    print(f" * Frontend UI:   {frontend_url}")
    print("=" * 64)

    def open_browser():
        print(f"\n[TubeMind] Automatically opening frontend in browser: {frontend_url}\n")
        webbrowser.open(frontend_url)

    threading.Timer(1.2, open_browser).start()
    uvicorn.run(app, host=host, port=port)
