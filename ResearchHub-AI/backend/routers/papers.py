"""
Papers Router Module
Handles document ingestion and semantic search endpoints.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from utils.document_loader import DocumentLoader
from utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/papers", tags=["papers"])

# Initialize document loader and vector store
document_loader = None
vector_store = None


def initialize_papers_router(db_path: str, collection_name: str, data_dir: str):
    """
    Initialize the papers router with document loader and vector store.

    Args:
        db_path: Path to vector database
        collection_name: Name of the ChromaDB collection
        data_dir: Path to data directory with PDFs
    """
    global document_loader, vector_store
    document_loader = DocumentLoader(data_dir=data_dir)
    vector_store = VectorStore(db_path=db_path, collection_name=collection_name)
    logger.info("Papers router initialized")


class SearchRequest(BaseModel):
    """Search request model."""

    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Individual search result model."""

    rank: int
    document: str
    similarity: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""

    status: str
    query: str
    results_count: int
    results: List[SearchResult]


class IngestionResponse(BaseModel):
    """Ingestion response model."""

    status: str
    message: str
    documents_ingested: int


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF file to the data directory and ingest it into the vector store.
    """
    try:
        if not document_loader or not vector_store:
            raise HTTPException(status_code=500, detail="Router not initialized")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        def sanitize_filename(name: str) -> str:
            cleaned = re.sub(r"[<>:\\\"/|?*]", "_", name)
            cleaned = cleaned.strip().rstrip(". ")
            return cleaned or "uploaded.pdf"

        data_dir = Path(document_loader.data_dir)
        safe_name = sanitize_filename(file.filename)
        file_path = data_dir / safe_name

        counter = 1
        original_name = safe_name
        while file_path.exists():
            name_parts = original_name.rsplit(".", 1)
            file_path = data_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
            counter += 1

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Saved uploaded file: {file_path}")

        text = document_loader.load_pdf(str(file_path))
        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "document_type": "pdf",
        }
        chunks = document_loader.chunk_text(text, metadata)

        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

        result = vector_store.ingest_documents(chunks)

        if result["status"] == "success":
            logger.info(
                f"Successfully uploaded and ingested {file_path.name}: {result['count']} chunks"
            )
            return {
                "status": "success",
                "message": "File uploaded and ingested successfully",
                "filename": file_path.name,
                "documents_ingested": result["count"],
            }

        raise HTTPException(status_code=500, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents() -> Dict[str, Any]:
    """Ingest PDF documents from the data directory into the vector store."""
    try:
        if not document_loader or not vector_store:
            raise HTTPException(status_code=500, detail="Router not initialized")

        logger.info("Starting document ingestion process...")

        documents = document_loader.load_documents_from_directory()

        if not documents:
            logger.warning("No documents found to ingest")
            return {
                "status": "warning",
                "message": "No PDF documents found in data directory. Please add PDFs to the data folder.",
                "documents_ingested": 0,
            }

        result = vector_store.ingest_documents(documents)

        if result["status"] == "success":
            logger.info(f"Successfully ingested {result['count']} document chunks")
            return {
                "status": "success",
                "message": result["message"],
                "documents_ingested": result["count"],
            }

        raise HTTPException(status_code=500, detail=result["message"])

    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for similar documents using semantic search."""
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")

        top_k = min(max(1, top_k), 20)
        logger.info(f"Searching for query: '{query}' with top_k={top_k}")

        results = vector_store.query_similar_documents(query, top_k=top_k)

        formatted_results = [
            SearchResult(
                rank=result["rank"],
                document=result["document"],
                similarity=result["similarity"],
                metadata=result["metadata"],
            )
            for result in results
        ]

        logger.info(f"Found {len(formatted_results)} similar documents")

        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
