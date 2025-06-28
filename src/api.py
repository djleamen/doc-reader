"""
FastAPI web application for the RAG Document Q&A system.
"""
import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

from src.config import settings
from src.rag_engine import RAGEngine, ConversationalRAG, QueryResult
from src.document_processor import DocumentProcessor


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None
    include_sources: bool = True
    include_scores: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    confidence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any]


class DocumentUploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks_added: int
    errors: List[str]


class IndexStatsResponse(BaseModel):
    index_name: str
    stats: Dict[str, Any]


# Global RAG engine instance
rag_engine = None
conversational_rag = None


def get_rag_engine() -> RAGEngine:
    """Dependency to get RAG engine instance."""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


def get_conversational_rag() -> ConversationalRAG:
    """Dependency to get conversational RAG engine instance."""
    global conversational_rag
    if conversational_rag is None:
        conversational_rag = ConversationalRAG()
    return conversational_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the application on startup."""
    # Create necessary directories
    os.makedirs(settings.documents_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)
    os.makedirs(settings.logs_dir, exist_ok=True)
    
    logger.info("RAG Document Q&A API started successfully")
    yield
    # Cleanup code can go here if needed


# Create FastAPI app
app = FastAPI(
    title="RAG Document Q&A API",
    description="A powerful document Q&A system using Retrieval-Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Document Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "upload_documents": "/upload-documents",
            "query": "/query",
            "conversational_query": "/conversational-query",
            "index_stats": "/index-stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-document-qa"}


@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    index_name: str = Query("default", description="Name of the index to add documents to"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Upload and process documents for the RAG system."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    errors = []
    total_chunks = 0
    
    # Create temporary directory for uploaded files
    temp_dir = Path(settings.documents_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files
        file_paths = []
        for file in files:
            if not file.filename:
                errors.append("File with no filename skipped")
                continue
            
            # Check file extension
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            if file_extension not in settings.supported_formats_list:
                errors.append(f"Unsupported file format: {file.filename}")
                continue
            
            # Save file
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
            processed_files.append(file.filename)
        
        # Process documents
        if file_paths:
            # Get document processor to count chunks
            doc_processor = DocumentProcessor()
            
            for file_path in file_paths:
                try:
                    documents = doc_processor.process_document(file_path)
                    total_chunks += len(documents)
                except Exception as e:
                    errors.append(f"Error processing {Path(file_path).name}: {str(e)}")
            
            # Add to RAG engine
            try:
                rag_engine.add_documents(file_paths)
            except Exception as e:
                errors.append(f"Error adding documents to index: {str(e)}")
        
        return DocumentUploadResponse(
            message=f"Processed {len(processed_files)} files successfully",
            files_processed=processed_files,
            total_chunks_added=total_chunks,
            errors=errors
        )
    
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    index_name: str = Query("default", description="Name of the index to query"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Query the document index with a question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_engine.query(
            question=request.question,
            k=request.k,
            include_sources=request.include_sources,
            include_scores=request.include_scores
        )
        
        # Convert source documents to serializable format
        source_docs = None
        if request.include_sources and result.source_documents:
            source_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result.source_documents
            ]
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            source_documents=source_docs,
            confidence_scores=result.confidence_scores if request.include_scores else None,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/conversational-query", response_model=QueryResponse)
async def conversational_query(
    request: QueryRequest,
    index_name: str = Query("default", description="Name of the index to query"),
    conv_rag: ConversationalRAG = Depends(get_conversational_rag)
):
    """Query with conversation context."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = conv_rag.conversational_query(
            question=request.question,
            k=request.k,
            include_sources=request.include_sources,
            include_scores=request.include_scores
        )
        
        # Convert source documents to serializable format
        source_docs = None
        if request.include_sources and result.source_documents:
            source_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result.source_documents
            ]
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            source_documents=source_docs,
            confidence_scores=result.confidence_scores if request.include_scores else None,
            metadata=result.metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing conversational query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing conversational query: {str(e)}")


@app.delete("/conversation")
async def clear_conversation(
    conv_rag: ConversationalRAG = Depends(get_conversational_rag)
):
    """Clear conversation history."""
    conv_rag.clear_conversation()
    return {"message": "Conversation history cleared"}


@app.get("/index-stats", response_model=IndexStatsResponse)
async def get_index_stats(
    index_name: str = Query("default", description="Name of the index"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Get statistics about the document index."""
    try:
        stats = rag_engine.get_index_stats()
        return IndexStatsResponse(
            index_name=index_name,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting index stats: {str(e)}")


@app.delete("/index")
async def clear_index(
    index_name: str = Query("default", description="Name of the index to clear"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Clear the document index."""
    try:
        rag_engine.clear_index()
        return {"message": f"Index '{index_name}' cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported document formats."""
    return {
        "supported_formats": settings.supported_formats_list,
        "max_file_size_mb": settings.max_document_size_mb
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
