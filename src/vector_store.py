"""
Vector database management for document embeddings.

Provides abstraction layer for multiple vector database backends including
FAISS, ChromaDB, and Pinecone with local and OpenAI embedding support.

Written by DJ Leamen (2025-2026)
"""

import os
import json
from typing import List, Optional, cast
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import settings

NOT_INITIALIZED = "Vector store not initialized. Add documents first."

class LocalEmbeddings(Embeddings):
    '''
    Local embeddings using sentence-transformers.
    
    Provides embedding generation without external API calls
    using sentence-transformers models.
    '''

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        '''
        Initialize local embeddings.
        
        :param model_name: Name of the sentence-transformers model to use
        '''
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        '''
        Embed a list of documents.
        
        :param texts: List of text strings to embed
        :return: List of embedding vectors
        '''
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Convert numpy array to list of lists
        embeddings_array = cast(np.ndarray, embeddings)
        return embeddings_array.tolist()

    def embed_query(self, text: str) -> List[float]:
        '''
        Embed a single query.
        
        :param text: Query text to embed
        :return: Embedding vector
        '''
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class VectorStoreManager(ABC):
    '''
    Abstract base class for vector store implementations.
    
    Defines interface for vector database operations including
    document storage, retrieval, and persistence.
    '''

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        '''
        Add documents to the vector store.
        
        :param documents: List of documents to add
        '''
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        '''
        Search for similar documents.
        
        :param query: Query text
        :param k: Number of results to return
        :return: List of similar documents
        '''
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        '''
        Save the vector store to disk.
        
        :param path: Path to save the vector store
        '''
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        '''
        Load the vector store from disk.
        
        :param path: Path to load the vector store from
        '''
        pass


class FAISSVectorStore(VectorStoreManager):
    '''
    FAISS-based vector store implementation.
    
    Uses Facebook AI Similarity Search for efficient vector
    similarity operations with local or OpenAI embeddings.
    '''

    def __init__(self):
        # Choose embeddings based on configuration
        use_local = os.getenv("USE_LOCAL_EMBEDDINGS",
                              "false").lower() == "true"

        if use_local or not settings.openai_api_key:
            logger.info("Using local embeddings (sentence-transformers)")
            self.embeddings = LocalEmbeddings("all-MiniLM-L6-v2")
        else:
            logger.info("Using OpenAI embeddings")
            # Note: OpenAIEmbeddings will use OPENAI_API_KEY environment variable
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model
            )

        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        '''
        Add documents to FAISS vector store.
        
        :param documents: List of documents to add to the index
        '''
        logger.info(f"Adding {len(documents)} documents to FAISS store")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        self.documents.extend(documents)
        logger.info(f"Total documents in store: {len(self.documents)}")

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        '''
        Search for similar documents using FAISS.
        
        :param query: Query text
        :param k: Number of results to return (None uses default)
        :return: List of most similar documents
        :raises ValueError: If vector store not initialized
        '''
        if self.vector_store is None:
            raise ValueError(
                NOT_INITIALIZED)

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} similar documents for query")
        return results

    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[tuple]:
        '''
        Search with similarity scores.
        
        :param query: Query text
        :param k: Number of results to return (None uses default)
        :return: List of (document, score) tuples
        :raises ValueError: If vector store not initialized
        '''
        if self.vector_store is None:
            raise ValueError(
                NOT_INITIALIZED)

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search_with_score(query, k=k)

        logger.info(f"Found {len(results)} similar documents with scores")
        return results

    def save(self, path: str) -> None:
        '''
        Save FAISS index and metadata to disk.
        
        :param path: Directory path to save the index
        :raises ValueError: If no vector store to save
        '''
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = path_obj / "faiss_index"
        self.vector_store.save_local(str(faiss_path))

        # Save document metadata
        metadata_path = path_obj / "metadata.json"
        metadata = {
            "total_documents": len(self.documents),
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Vector store saved to {path_obj}")

    def load(self, path: str) -> None:
        '''
        Load FAISS index from disk.
        
        :param path: Directory path to load the index from
        :raises FileNotFoundError: If FAISS index not found
        '''
        path_obj = Path(path)
        faiss_path = path_obj / "faiss_index"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        self.vector_store = FAISS.load_local(
            str(faiss_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Load metadata if available
        metadata_path = path_obj / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(
                f"Loaded vector store with {metadata.get('total_documents', 'unknown')} documents")

        logger.info(f"Vector store loaded from {path_obj}")


class ChromaVectorStore(VectorStoreManager):
    '''
    ChromaDB-based vector store implementation.
    
    Uses ChromaDB for vector storage and retrieval with
    built-in persistence support.
    '''

    def __init__(self):
        try:
            from langchain_community.vectorstores import Chroma

            # Note: OpenAIEmbeddings will use OPENAI_API_KEY environment variable
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model
            )
            self.vector_store: Optional[Chroma] = None
            self.collection_name = "document_collection"

        except ImportError as exc:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb") from exc

    def add_documents(self, documents: List[Document]) -> None:
        '''
        Add documents to Chroma vector store.
        
        :param documents: List of documents to add to the collection
        '''
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

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        '''
        Search for similar documents using Chroma.
        
        :param query: Query text
        :param k: Number of results to return (None uses default)
        :return: List of most similar documents
        :raises ValueError: If vector store not initialized
        '''
        if self.vector_store is None:
            raise ValueError(
                NOT_INITIALIZED)

        k = k or settings.top_k_results
        results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} similar documents for query")
        return results

    def save(self, path: str) -> None:
        '''
        Save Chroma collection.
        
        :param path: Path parameter (unused, Chroma auto-persists)
        '''
        # Chroma automatically persists data
        logger.info("Chroma vector store automatically persisted")

    def load(self, path: str) -> None:
        '''
        Load Chroma collection.
        
        :param path: Path parameter (unused, loads from default location)
        '''
        from langchain_community.vectorstores import Chroma

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        logger.info("Chroma vector store loaded")


class VectorStoreFactory:
    '''
    Factory for creating vector store instances.
    
    Instantiates appropriate vector store implementation based
    on configuration settings.
    '''

    @staticmethod
    def create_vector_store(store_type: Optional[str] = None) -> VectorStoreManager:
        '''
        Create a vector store instance based on configuration.
        
        :param store_type: Type of vector store ('faiss' or 'chroma', None uses config)
        :return: Initialized VectorStoreManager instance
        :raises ValueError: If unsupported store type specified
        '''
        store_type = store_type or settings.vector_db_type

        if store_type.lower() == "faiss":
            return FAISSVectorStore()
        elif store_type.lower() == "chroma":
            return ChromaVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


class DocumentIndex:
    '''
    High-level document indexing and search interface.
    
    Provides simplified API for document management with automatic
    persistence and vector store abstraction.
    '''

    def __init__(self, index_name: str = "default"):
        '''
        Initialize document index.
        
        :param index_name: Name of the index (alphanumeric, dashes, underscores only)
        :raises ValueError: If index name is invalid or leads to unsafe path
        '''
        import re
        from os.path import normpath, realpath

        # Validate index_name with a regex pattern
        if not re.match(r'^[a-zA-Z0-9_-]+$', index_name):
            raise ValueError(
                f"Invalid index name: {index_name}. Only alphanumeric characters, dashes, and underscores are allowed.")

        self.index_name = index_name
        self.vector_store = VectorStoreFactory.create_vector_store()

        # Construct and validate index path
        normalized_index_name = normpath(index_name)
        self.index_path = Path(
            realpath(Path(settings.index_dir) / normalized_index_name))

        if not str(self.index_path).startswith(str(realpath(settings.index_dir))):
            raise ValueError(
                f"Invalid index name: {index_name} leads to unsafe path: {self.index_path}")

        # Try to load existing index
        if self.index_path.exists():
            try:
                self.load_index()
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        '''
        Add documents to the index.
        
        :param documents: List of documents to add and persist
        '''
        self.vector_store.add_documents(documents)
        self.save_index()

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        '''
        Search the document index.
        
        :param query: Query text
        :param k: Number of results to return (None uses default)
        :return: List of most similar documents
        '''
        return self.vector_store.similarity_search(query, k)

    def search_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        '''
        Search with similarity scores if supported.
        
        :param query: Query text
        :param k: Number of results to return (None uses default)
        :return: List of (document, score) tuples
        '''
        if isinstance(self.vector_store, FAISSVectorStore):
            return self.vector_store.similarity_search_with_score(query, k)
        else:
            # Fallback to regular search for stores without score support
            docs = self.search(query, k)
            return [(doc, 1.0) for doc in docs]

    def save_index(self) -> None:
        '''
        Save the index to disk.
        
        Persists the vector store to the configured index path.
        '''
        self.vector_store.save(str(self.index_path))

    def load_index(self) -> None:
        '''
        Load the index from disk.
        
        Restores the vector store from the configured index path.
        '''
        self.vector_store.load(str(self.index_path))

    def clear_index(self) -> None:
        '''
        Clear all documents from the index and delete the index files.
        
        Resets the vector store and removes all persisted data from disk.
        '''
        import shutil
        # Reset the vector store
        self.vector_store = VectorStoreFactory.create_vector_store()
        # Delete the index directory if it exists
        if self.index_path.exists():
            try:
                shutil.rmtree(self.index_path)
                logger.info(f"Cleared index at {self.index_path}")
            except Exception as e:
                logger.error(f"Error clearing index directory: {e}")
                raise
