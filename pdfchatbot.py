import os
import re
import pickle
import threading
import webbrowser
from typing import Optional, List
from dotenv import load_dotenv
import fitz
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".faiss_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
app = FastAPI(title="DocuMind AI - Hybrid PDF RAG Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("[WARNING] GROQ_API_KEY not found in .env or environment!")
model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0, max_tokens=1000)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
parser = StrOutputParser()

template = PromptTemplate(
    template="""You are a strict and knowledgeable technical assistant.
Answer the question thoroughly and accurately using ONLY the provided context snippets below.
If the context does not contain enough detail to fully answer, state what you know from the context and mention what is missing.
Always cite the relevant page numbers whenever mentioned in the context.

Context:
---
{context}
---

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

document_cache = {}

class QueryRequest(BaseModel):
    question: str
    filename: Optional[str] = None

def sanitize_name(filename: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(filename)[0]).lower()
    return clean.strip("_") or "doc"

def list_available_pdfs() -> List[dict]:
    results = []
    for item in os.listdir(BASE_DIR):
        if item.lower().endswith(".pdf"):
            full_path = os.path.join(BASE_DIR, item)
            size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
            safe_name = sanitize_name(item)
            disk_cached = os.path.exists(os.path.join(CACHE_DIR, safe_name, "index.faiss"))
            in_memory = item in document_cache
            
            page_count = 0
            chunk_count = 0
            if in_memory:
                page_count = document_cache[item].get("page_count", 0)
                chunk_count = len(document_cache[item].get("chunks", []))
            elif disk_cached:
                try:
                    chunks_file = os.path.join(CACHE_DIR, safe_name, "chunks.pkl")
                    if os.path.exists(chunks_file):
                        with open(chunks_file, "rb") as f:
                            saved_chunks = pickle.load(f)
                            chunk_count = len(saved_chunks)
                            page_count = max((c.metadata.get("page", 0) for c in saved_chunks), default=0)
                except Exception:
                    pass

            if page_count == 0:
                try:
                    doc = fitz.open(full_path)
                    page_count = len(doc)
                    doc.close()
                except Exception:
                    page_count = 1

            results.append({
                "filename": item,
                "size_mb": size_mb,
                "pages": page_count,
                "chunks": chunk_count,
                "is_indexed": in_memory or disk_cached
            })
    return results

def get_or_create_index(filename: str):
    if filename in document_cache:
        return document_cache[filename]

    pdf_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file '{filename}' not found.")

    safe_name = sanitize_name(filename)
    sub_cache = os.path.join(CACHE_DIR, safe_name)
    faiss_index_file = os.path.join(sub_cache, "index.faiss")
    chunks_file = os.path.join(sub_cache, "chunks.pkl")

    if os.path.exists(faiss_index_file) and os.path.exists(chunks_file):
        try:
            print(f"[DocuMind] Loading pre-cached FAISS index for '{filename}'...")
            vector_store = FAISS.load_local(sub_cache, embedding, allow_dangerous_deserialization=True)
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)

            bm25 = BM25Retriever.from_documents(chunks)
            bm25.k = 4
            mmr_retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}
            )
            doc = fitz.open(pdf_path)
            pages = [
                Document(page_content=page.get_text(), metadata={"page": i + 1, "source": filename})
                for i, page in enumerate(doc) if page.get_text().strip()
            ]
            doc.close()

            doc_data = {
                "vector_store": vector_store,
                "bm25": bm25,
                "mmr_retriever": mmr_retriever,
                "docs": pages,
                "chunks": chunks,
                "page_count": len(pages),
                "filename": filename
            }
            document_cache[filename] = doc_data
            return doc_data
        except Exception as e:
            print(f"[DocuMind] Error loading disk cache for '{filename}': {e}. Re-indexing...")
    print(f"[DocuMind] Indexing PDF '{filename}' (Extracting pages & computing embeddings)...")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(Document(page_content=text, metadata={"page": i + 1, "source": filename}))
    doc.close()

    if not pages:
        raise HTTPException(status_code=400, detail=f"No readable text found in PDF '{filename}'.")

    chunks = splitter.split_documents(pages)
    if not chunks:
        chunks = pages
    vector_store = FAISS.from_documents(chunks, embedding=embedding)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}
    )
    try:
        os.makedirs(sub_cache, exist_ok=True)
        vector_store.save_local(sub_cache)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        print(f"[DocuMind] Saved FAISS cache to '{sub_cache}'.")
    except Exception as e:
        print(f"[DocuMind] Warning: Could not save cache to disk: {e}")

    doc_data = {
        "vector_store": vector_store,
        "bm25": bm25,
        "mmr_retriever": mmr_retriever,
        "docs": pages,
        "chunks": chunks,
        "page_count": len(pages),
        "filename": filename
    }
    document_cache[filename] = doc_data
    return doc_data

def dynamic_context_retriever(doc_data: dict, query: str):
    q_lower = query.lower()
    docs = doc_data["docs"]
    mmr_retriever = doc_data["mmr_retriever"]
    bm25 = doc_data["bm25"]

    if any(term in q_lower for term in ["summary", "overview", "topics", "table of contents", "what is this book about"]):
        intro_pages = [doc for doc in docs if doc.metadata.get("page", 0) <= 6]
        if intro_pages:
            combined = "\n\n".join(f"[Page {doc.metadata.get('page')}]: {doc.page_content}" for doc in intro_pages)
            sources = [{"page": doc.metadata.get("page", 1), "snippet": doc.page_content[:200]} for doc in intro_pages]
            return combined, sources

    dense_hits = mmr_retriever.invoke(query)
    sparse_hits = bm25.invoke(query)
    seen = set()
    combined_texts = []
    sources = []
    for doc in dense_hits + sparse_hits:
        content = doc.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            page_num = doc.metadata.get("page", 1)
            combined_texts.append(f"[Page {page_num}]: {content}")
            sources.append({"page": page_num, "snippet": content[:240]})
    final_context = "\n\n".join(combined_texts[:6])
    return final_context, sources[:6]

@app.get("/")
def serve_frontend():
    frontend_path = os.path.join(BASE_DIR, "frontend.html")
    if not os.path.exists(frontend_path):
        return HTMLResponse("<h2>frontend.html not found</h2>", status_code=404)
    return FileResponse(frontend_path, media_type="text/html")

@app.get("/frontend.html")
def serve_frontend_alias():
    return serve_frontend()

@app.get("/health")
def health_check():
    all_pdfs = list_available_pdfs()
    default_doc = "The Ultimate Python Handbook.pdf"
    if not any(p["filename"] == default_doc for p in all_pdfs) and all_pdfs:
        default_doc = all_pdfs[0]["filename"]

    return {
        "status": "ok",
        "default_document": default_doc,
        "indexed_documents": list(document_cache.keys()),
        "total_documents": len(all_pdfs)
    }

@app.get("/documents")
def get_documents():
    return {"documents": list_available_pdfs()}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(BASE_DIR, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    doc_data = get_or_create_index(file.filename)
    return {
        "status": "success",
        "message": f"Successfully uploaded and indexed '{file.filename}'.",
        "filename": file.filename,
        "pages": doc_data["page_count"],
        "chunks": len(doc_data["chunks"])
    }

@app.post("/ask")
def ask_document(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    filename = request.filename
    if not filename:
        all_pdfs = list_available_pdfs()
        if any(p["filename"] == "The Ultimate Python Handbook.pdf" for p in all_pdfs):
            filename = "The Ultimate Python Handbook.pdf"
        elif all_pdfs:
            filename = all_pdfs[0]["filename"]
        else:
            raise HTTPException(status_code=404, detail="No PDF documents found in workspace.")

    doc_data = get_or_create_index(filename)
    context_text, sources = dynamic_context_retriever(doc_data, question)
    chain = template | model | parser
    answer = chain.invoke({"context": context_text, "question": question})
    return {
        "answer": answer,
        "question": question,
        "filename": filename,
        "sources": sources
    }

def open_browser():
    url = "http://127.0.0.1:8000"
    print(f"\n[DocuMind] Automatically launching frontend at {url} ...\n")
    webbrowser.open(url)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("=" * 64)
    print(" DocuMind AI - Hybrid PDF RAG Assistant (FastAPI)")
    print(f" * Backend API:   http://{host}:{port}")
    print(f" * Frontend UI:   http://{host}:{port}/")
    print("=" * 64)

    default_pdf = "The Ultimate Python Handbook.pdf"
    if os.path.exists(os.path.join(BASE_DIR, default_pdf)):
        try:
            get_or_create_index(default_pdf)
        except Exception as e:
            print(f"[DocuMind] Notice during warm-up: {e}")
    threading.Timer(1.2, open_browser).start()
    uvicorn.run(app, host=host, port=port)