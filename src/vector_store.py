"""
Vector database management for document embeddings.
"""
import os
import pickle
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

from src.config import settings


class VectorStoreManager(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(VectorStoreManager):
    """FAISS-based vector store implementation."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS vector store."""
        logger.info(f"Adding {len(documents)} documents to FAISS store")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        self.documents.extend(documents)
        logger.info(f"Total documents in store: {len(self.documents)}")

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents using FAISS."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} similar documents for query")
        return results

    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Search with similarity scores."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search_with_score(query, k=k)

        logger.info(f"Found {len(results)} similar documents with scores")
        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = path / "faiss_index"
        self.vector_store.save_local(str(faiss_path))

        # Save document metadata
        metadata_path = path / "metadata.json"
        metadata = {
            "total_documents": len(self.documents),
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Vector store saved to {path}")

    def load(self, path: str) -> None:
        """Load FAISS index from disk."""
        path = Path(path)
        faiss_path = path / "faiss_index"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        self.vector_store = FAISS.load_local(
            str(faiss_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Load metadata if available
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded vector store with {metadata.get('total_documents', 'unknown')} documents")

        logger.info(f"Vector store loaded from {path}")


class ChromaVectorStore(VectorStoreManager):
    """ChromaDB-based vector store implementation."""

    def __init__(self):
        try:
            import chromadb
            from langchain_community.vectorstores import Chroma

            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
            self.vector_store: Optional[Chroma] = None
            self.collection_name = "document_collection"

        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Chroma vector store."""
        from langchain_community.vectorstores import Chroma

        logger.info(f"Adding {len(documents)} documents to Chroma store")

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                collection_name=self.collection_name
            )
        else:
            self.vector_store.add_documents(documents)

        logger.info("Documents added to Chroma store")

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents using Chroma."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Add documents first.")

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} similar documents for query")
        return results

    def save(self, path: str) -> None:
        """Save Chroma collection."""
        # Chroma automatically persists data
        logger.info("Chroma vector store automatically persisted")

    def load(self, path: str) -> None:
        """Load Chroma collection."""
        from langchain_community.vectorstores import Chroma

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        logger.info("Chroma vector store loaded")


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create_vector_store(store_type: str = None) -> VectorStoreManager:
        """Create a vector store instance based on configuration."""
        store_type = store_type or settings.vector_db_type

        if store_type.lower() == "faiss":
            return FAISSVectorStore()
        elif store_type.lower() == "chroma":
            return ChromaVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


class DocumentIndex:
    """High-level document indexing and search interface."""

    def __init__(self, index_name: str = "default"):
        import re
        from os.path import normpath, realpath
        
        # Validate index_name with a regex pattern
        if not re.match(r'^[a-zA-Z0-9_-]+$', index_name):
            raise ValueError(f"Invalid index name: {index_name}. Only alphanumeric characters, dashes, and underscores are allowed.")
        
        self.index_name = index_name
        self.vector_store = VectorStoreFactory.create_vector_store()
        
        # Construct and validate index path
        normalized_index_name = normpath(index_name)
        self.index_path = Path(realpath(Path(settings.index_dir) / normalized_index_name))
        
        if not str(self.index_path).startswith(str(realpath(settings.index_dir))):
            raise ValueError(f"Invalid index name: {index_name} leads to unsafe path: {self.index_path}")

        # Try to load existing index
        if self.index_path.exists():
            try:
                self.load_index()
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        self.vector_store.add_documents(documents)
        self.save_index()

    def search(self, query: str, k: int = None) -> List[Document]:
        """Search the document index."""
        return self.vector_store.similarity_search(query, k)

    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """Search with similarity scores if supported."""
        if hasattr(self.vector_store, 'similarity_search_with_score'):
            return self.vector_store.similarity_search_with_score(query, k)
        else:
            # Fallback to regular search
            docs = self.search(query, k)
            return [(doc, 1.0) for doc in docs]

    def save_index(self) -> None:
        """Save the index to disk."""
        self.vector_store.save(str(self.index_path))

    def load_index(self) -> None:
        """Load the index from disk."""
        self.vector_store.load(str(self.index_path))
