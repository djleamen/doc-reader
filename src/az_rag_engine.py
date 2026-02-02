"""
Experimental Azure RAG Engine with advanced features.
Combines Azure OpenAI, AI Search, and Document Intelligence.

Written by DJ Leamen (2025-2026)
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from langchain_core.documents import Document
from loguru import logger

from src.az_config import azure_settings
from src.az_openai_service import get_azure_openai_service
from src.az_vector_store import AzureSearchVectorStore
from src.az_document_processor import get_azure_document_processor


@dataclass
class AzureQueryResult:
    '''
    Result of a query to the Azure RAG system.
    Contains the answer, source documents, query details, and metadata.
    '''
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    elapsed_time: float
    model: str
    search_method: str
    cache_hit: bool = False
    metadata: Optional[Dict[str, Any]] = None


class QueryCache:
    """Simple in-memory cache for query results."""

    def __init__(self, ttl: int = 3600):
        '''
        Initialize query cache with time-to-live (TTL).
        
        :param self: QueryCache instance
        :param ttl: Time to live in seconds for cached results
        :type ttl: int
        '''
        self.cache: Dict[str, tuple[AzureQueryResult, datetime]] = {}
        self.ttl = ttl

    def _generate_key(self, query: str, index_name: str, k: int) -> str:
        '''
        Generate a unique cache key based on query parameters.
        
        :param self: QueryCache instance
        :param query: Query string
        :type query: str
        :param index_name: Name of the search index
        :type index_name: str
        :param k: Number of results to retrieve
        :type k: int
        :return: Unique cache key as SHA256 hash string
        :rtype: str
        '''
        key_string = f"{query}:{index_name}:{k}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, query: str, index_name: str, k: int) -> Optional[AzureQueryResult]:
        '''
        Retrieve cached query result if valid.
        
        :param self: QueryCache instance
        :param query: Query string
        :type query: str
        :param index_name: Name of the search index
        :type index_name: str
        :param k: Number of results to retrieve
        :type k: int
        :return: Cached AzureQueryResult if found and valid, None otherwise
        :rtype: AzureQueryResult | None
        '''
        key = self._generate_key(query, index_name, k)

        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                logger.info("Cache hit for query")
                result.cache_hit = True
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]

        return None

    def set(self, query: str, index_name: str, k: int, result: AzureQueryResult) -> None:
        '''
        Cache query result.
        
        :param self: QueryCache instance
        :param query: Query string
        :type query: str
        :param index_name: Name of the search index
        :type index_name: str
        :param k: Number of results to retrieve
        :type k: int
        :param result: AzureQueryResult to cache
        :type result: AzureQueryResult
        '''
        key = self._generate_key(query, index_name, k)
        self.cache[key] = (result, datetime.now())

    def clear(self) -> None:
        '''
        Clear all cached results.
        
        :param self: QueryCache instance
        '''
        self.cache.clear()
        logger.info("Query cache cleared")


class AzureRAGEngine:
    """
    Experimental RAG engine using Azure services with:
    - Azure OpenAI for embeddings and chat
    - Azure AI Search for vector + hybrid search
    - Azure Document Intelligence for document processing
    - Query result caching
    - Performance monitoring
    - Error handling and retry logic
    """

    def __init__(self, index_name: Optional[str] = None):
        '''
        Initialize Azure RAG engine.
        
        :param self: AzureRAGEngine instance
        :param index_name: Name of the search index, uses default from settings if None
        :type index_name: Optional[str]
        '''
        self.settings = azure_settings
        self.index_name = index_name or self.settings.search_index_name

        # Initialize Azure services
        self.openai_service = get_azure_openai_service()
        self.vector_store = AzureSearchVectorStore(index_name=self.index_name)
        self.document_processor = get_azure_document_processor()

        # Initialize cache
        self.cache = QueryCache(
            ttl=self.settings.cache_ttl) if self.settings.enable_caching else None

        logger.info(
            f"Azure RAG Engine initialized with index: {self.index_name}")

    def add_document(self, file_path: str) -> Dict[str, Any]:
        '''
        Add and process a document to the RAG system.
        
        :param self: AzureRAGEngine instance
        :param file_path: Path to document file to process and add
        :type file_path: str
        :return: Dictionary with status, document info, and processing metrics
        :rtype: Dict[str, Any]
        '''
        try:
            logger.info(f"Adding document: {file_path}")
            start_time = time.time()

            # Process document using Azure Document Intelligence
            documents = self.document_processor.process_document(file_path)

            # Add to vector store
            result = self.vector_store.add_documents(documents)

            elapsed_time = time.time() - start_time

            return {
                "status": "success",
                "file_path": file_path,
                "chunks": len(documents),
                "indexed_count": result["count"],
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {
                "status": "error",
                "error": "An internal error occurred while processing this file.",
                "file_path": file_path,
            }

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_semantic: Optional[bool] = None,
        temperature: Optional[float] = None,
        include_sources: bool = True,
    ) -> AzureQueryResult:
        '''
        Process a query using the RAG engine.
        
        :param self: AzureRAGEngine instance
        :param question: User question to answer
        :type question: str
        :param k: Number of documents to retrieve, uses default if None
        :type k: Optional[int]
        :param use_hybrid: Use hybrid search (vector + keyword), uses default if None
        :type use_hybrid: Optional[bool]
        :param use_semantic: Use semantic ranking, uses default if None
        :type use_semantic: Optional[bool]
        :param temperature: LLM temperature for answer generation
        :type temperature: Optional[float]
        :param include_sources: Include source documents in result
        :type include_sources: bool
        :return: Description
        :rtype: AzureQueryResult
        '''
        try:
            logger.info(f"Processing query: '{question}'")
            start_time = time.time()

            k = k or self.settings.top_k_results
            use_hybrid = use_hybrid if use_hybrid is not None else self.settings.enable_hybrid_search
            use_semantic = use_semantic if use_semantic is not None else self.settings.enable_semantic_search

            # Check cache
            if self.cache:
                cached_result = self.cache.get(question, self.index_name, k)
                if cached_result:
                    return cached_result

            # Retrieve relevant documents
            documents = self.vector_store.similarity_search(
                query=question,
                k=k,
                use_hybrid=use_hybrid,
                use_semantic=use_semantic,
            )

            # Build context from retrieved documents
            context = self._build_context(documents)

            # Generate answer using Azure OpenAI
            answer = self.openai_service.chat_completion_with_context(
                query=question,
                context=context,
                temperature=temperature,
            )

            # Prepare sources
            sources = []
            if include_sources:
                for doc in documents:
                    source = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score", 0.0),
                    }
                    sources.append(source)

            elapsed_time = time.time() - start_time

            # Create result
            search_method = []
            if use_hybrid:
                search_method.append("hybrid")
            else:
                search_method.append("vector")
            if use_semantic:
                search_method.append("semantic")

            result = AzureQueryResult(
                answer=answer,
                sources=sources,
                query=question,
                elapsed_time=elapsed_time,
                model=self.settings.openai_chat_deployment,
                search_method="+".join(search_method),
                cache_hit=False,
                metadata={
                    "k": k,
                    "documents_retrieved": len(documents),
                    "index_name": self.index_name,
                }
            )

            # Cache result
            if self.cache:
                self.cache.set(question, self.index_name, k, result)

            logger.info(f"Query completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def _build_context(self, documents: List[Document]) -> str:
        '''
        Build context string from retrieved documents.
        
        :param self: AzureRAGEngine instance
        :param documents: List of retrieved LangChain documents
        :type documents: List[Document]
        :return: Formatted context string with document content and metadata
        :rtype: str
        '''
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            score = doc.metadata.get("score", 0.0)

            context_parts.append(
                f"Document {i} (Source: {source}, Score: {score:.3f}):\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def clear_cache(self) -> None:
        '''
        Clear the query result cache.
        
        :param self: AzureRAGEngine instance
        '''
        if self.cache:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        '''
        Get statistics about the RAG engine and vector store.
        
        :param self: AzureRAGEngine instance
        :return: Dictionary with engine stats including document count, cache info, and model details
        :rtype: Dict[str, Any]
        '''
        try:
            vector_store_stats = self.vector_store.get_stats()

            stats = {
                "index_name": self.index_name,
                "document_count": vector_store_stats.get("document_count", 0),
                "cache_enabled": self.settings.enable_caching,
                "cache_size": len(self.cache.cache) if self.cache else 0,
                "hybrid_search_enabled": self.settings.enable_hybrid_search,
                "semantic_search_enabled": self.settings.enable_semantic_search,
                "openai_model": self.settings.openai_chat_deployment,
                "embedding_model": self.settings.openai_embedding_deployment,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise

    def validate_configuration(self) -> Dict[str, Any]:
        '''
        Validate the RAG engine configuration and connections.
        
        :param self: AzureRAGEngine instance
        :return: Dictionary with validation results for each service and missing config items
        :rtype: Dict[str, Any]
        '''
        results = {
            "configuration_valid": False,
            "openai_connection": False,
            "search_connection": False,
            "document_intelligence_connection": False,
            "missing_config": [],
        }

        try:
            # Check configuration
            is_valid, missing = self.settings.validate_configuration()
            results["configuration_valid"] = is_valid
            results["missing_config"] = missing

            # Test OpenAI connection
            try:
                results["openai_connection"] = self.openai_service.validate_connection()
            except Exception as e:
                logger.error(f"OpenAI validation failed: {e}")

            # Test Search connection
            try:
                self.vector_store.get_stats()
                results["search_connection"] = True
            except Exception as e:
                logger.error(f"Search validation failed: {e}")

            # Test Document Intelligence connection
            try:
                results["document_intelligence_connection"] = self.document_processor.validate_connection()
            except Exception as e:
                logger.error(f"Document Intelligence validation failed: {e}")

        except Exception as e:
            logger.error(f"Validation error: {e}")

        return results


class ConversationalAzureRAG(AzureRAGEngine):
    """
    Conversational RAG engine with conversation history.
    Extends AzureRAGEngine with multi-turn conversation support.
    """

    def __init__(self, index_name: Optional[str] = None):
        '''
        Docstring for ConversationalAzureRAG initializer.
       
        :param self: ConversationalAzureRAG instance
        :param index_name: Name of the search index, uses default from settings if None
        :type index_name: Optional[str]
        '''
        super().__init__(index_name=index_name)
        self.conversation_history: List[Dict[str, str]] = []
        logger.info("Conversational Azure RAG Engine initialized")

    def query_with_history(
        self,
        question: str,
        k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_semantic: Optional[bool] = None,
        temperature: Optional[float] = None,
        include_sources: bool = True,
    ) -> AzureQueryResult:
        '''
        Process a conversational query with history.
        
        :param self: ConversationalAzureRAG instance
        :param question: User question to answer
        :type question: str
        :param k: Number of documents to retrieve
        :type k: Optional[int]
        :param use_hybrid: Use hybrid search (vector + keyword)
        :type use_hybrid: Optional[bool]
        :param use_semantic: Use semantic ranking
        :type use_semantic: Optional[bool]
        :param temperature: LLM temperature for answer generation
        :type temperature: Optional[float]
        :param include_sources: Include source documents in result
        :type include_sources: bool
        :return: AzureQueryResult with answer considering conversation history
        :rtype: AzureQueryResult
        '''
        try:
            # Get base result
            result = self.query(
                question=question,
                k=k,
                use_hybrid=use_hybrid,
                use_semantic=use_semantic,
                temperature=temperature,
                include_sources=include_sources,
            )

            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": question,
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": result.answer,
            })

            # Keep only recent history (last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return result

        except Exception as e:
            logger.error(f"Conversational query failed: {e}")
            raise

    def clear_history(self) -> None:
        '''
        Clear the conversation history.
        
        :param self: ConversationalAzureRAG instance
        '''
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        '''
        Get the current conversation history.
        
        :param self: ConversationalAzureRAG instance
        :return: Current conversation history as a list of messages
        :rtype: List[Dict[str, str]]
        '''
        return self.conversation_history.copy()


# Global RAG engine instances
_azure_rag_engines: Dict[str, AzureRAGEngine] = {}


def get_azure_rag_engine(index_name: Optional[str] = None, conversational: bool = False) -> AzureRAGEngine:
    '''
    Get or create a cached Azure RAG engine instance.
    
    :param index_name: Name of the search index to use
    :type index_name: Optional[str]
    :param conversational: Whether to use conversational RAG with history
    :type conversational: bool
    :return: AzureRAGEngine or ConversationalAzureRAG instance
    :rtype: AzureRAGEngine
    '''
    index_name = index_name or azure_settings.search_index_name
    cache_key = f"{index_name}:{'conv' if conversational else 'std'}"

    if cache_key not in _azure_rag_engines:
        if conversational:
            _azure_rag_engines[cache_key] = ConversationalAzureRAG(
                index_name=index_name)
        else:
            _azure_rag_engines[cache_key] = AzureRAGEngine(
                index_name=index_name)

    return _azure_rag_engines[cache_key]
