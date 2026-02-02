"""
Azure AI Search vector store for experimental RAG.
Supports hybrid search (vector + keyword) and semantic ranking.

Written by DJ Leamen (2025-2026)
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from langchain_core.documents import Document
from loguru import logger

from src.az_config import azure_settings
from src.az_openai_service import get_azure_openai_service

# Constants
_SEARCH_CLIENT_NOT_INITIALIZED = "Search client not initialized"


class AzureSearchVectorStore:
    """
    Experimental vector store using Azure AI Search with:
    - Hybrid search (vector + keyword search)
    - Semantic ranking for better relevance
    - Managed Identity authentication
    - Automatic index creation
    - Performance optimization
    """

    def __init__(self, index_name: Optional[str] = None):
        '''
        Initialize Azure Search vector store.
        
        :param self: AzureSearchVectorStore instance
        :param index_name: Name of the search index (uses default from settings if None)
        :type index_name: Optional[str]
        '''
        self.settings = azure_settings
        self.index_name = index_name or self.settings.search_index_name
        self.openai_service = get_azure_openai_service()

        # Initialize clients
        self.index_client: Optional[SearchIndexClient] = None
        self.search_client: Optional[SearchClient] = None
        self._initialize_clients()

        # Ensure index exists
        self._ensure_index_exists()

    def _initialize_clients(self) -> None:
        '''
        Initialize Azure Search index and search clients.

        :param self: AzureSearchVectorStore instance
        '''
        try:
            endpoint = self.settings.search_endpoint

            if self.settings.use_managed_identity:
                # Use Managed Identity
                logger.info(
                    "Initializing Azure AI Search with Managed Identity")
                credential = self.settings.get_credential()

                self.index_client = SearchIndexClient(
                    endpoint=endpoint,
                    credential=credential
                )
                self.search_client = SearchClient(
                    endpoint=endpoint,
                    index_name=self.index_name,
                    credential=credential
                )
            elif self.settings.search_api_key:
                # Use API Key
                logger.info("Initializing Azure AI Search with API Key")
                credential = AzureKeyCredential(self.settings.search_api_key)

                self.index_client = SearchIndexClient(
                    endpoint=endpoint,
                    credential=credential
                )
                self.search_client = SearchClient(
                    endpoint=endpoint,
                    index_name=self.index_name,
                    credential=credential
                )
            else:
                raise ValueError(
                    "Azure AI Search requires either Managed Identity "
                    "or API Key authentication"
                )

            logger.info("Azure AI Search clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure AI Search clients: {e}")
            raise

    def _create_index_schema(self) -> SearchIndex:
        '''
        Create the search index schema.
        
        :param self: AzureSearchVectorStore instance
        :return: SearchIndex definition with fields, vector search, and semantic config
        :rtype: SearchIndex
        '''
        # Define fields
        fields = [
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada-002 embedding size
                vector_search_profile_name="vector-profile",
            ),
            SearchField(
                name="source",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=False,
            ),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
        ]
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algorithm",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine",
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-algorithm",
                )
            ],
        )

        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="source")],
            ),
        )

        semantic_search = SemanticSearch(
            configurations=[semantic_config]
        )

        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        return index

    def _ensure_index_exists(self) -> None:
        '''
        Ensure the search index exists, create if not.
        
        :param self: AzureSearchVectorStore instance
        '''
        if not self.index_client:
            raise RuntimeError("Index client not initialized")

        try:
            # Check if index exists
            try:
                self.index_client.get_index(self.index_name)
                logger.info(f"Index '{self.index_name}' already exists")
            except ResourceNotFoundError:
                # Create index
                logger.info(f"Creating index '{self.index_name}'")
                index = self._create_index_schema()
                self.index_client.create_index(index)
                logger.info(f"Index '{self.index_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        '''
        Add documents to the search index.
        
        :param self: AzureSearchVectorStore instance
        :param documents: List of LangChain documents to add
        :type documents: List[Document]
        :return: Dictionary with status, count of added documents, and elapsed time
        :rtype: Dict[str, Any]
        '''
        if not documents:
            return {"status": "success", "count": 0}

        if not self.search_client:
            raise RuntimeError(_SEARCH_CLIENT_NOT_INITIALIZED)

        try:
            logger.info(
                f"Adding {len(documents)} documents to index '{self.index_name}'")
            start_time = time.time()

            # Extract texts for embedding
            texts = [doc.page_content for doc in documents]

            # Generate embeddings in batches (using configured batch size)
            embedding_batch_size = self.settings.embedding_batch_size
            logger.info(
                f"Generating embeddings in batches of {embedding_batch_size}")
            embeddings = self.openai_service.get_embeddings(
                texts, batch_size=embedding_batch_size)

            # Prepare documents for upload
            logger.info("Preparing documents for upload")
            search_documents = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                search_doc = {
                    "id": f"{doc.metadata.get('source', 'unknown')}_{i}_{int(time.time())}",
                    "content": doc.page_content,
                    "embedding": embedding,
                    "source": doc.metadata.get("source", "unknown"),
                    "metadata": json.dumps(doc.metadata),
                    "chunk_id": doc.metadata.get("chunk_id", i),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                search_documents.append(search_doc)

            # Upload documents in batches
            upload_batch_size = 100
            uploaded_count = 0
            total_upload_batches = (
                len(search_documents) + upload_batch_size - 1) // upload_batch_size

            logger.info(
                f"Uploading {len(search_documents)} documents in batches of {upload_batch_size}")
            for i in range(0, len(search_documents), upload_batch_size):
                batch = search_documents[i:i + upload_batch_size]
                batch_num = (i // upload_batch_size) + 1
                logger.info(
                    f"Uploading batch {batch_num}/{total_upload_batches}")
                result = self.search_client.upload_documents(documents=batch)
                uploaded_count += len([r for r in result if r.succeeded])

            elapsed_time = time.time() - start_time
            logger.info(
                f"Added {uploaded_count} documents to index in {elapsed_time:.2f}s "
                f"({uploaded_count/elapsed_time:.2f} docs/s)"
            )

            return {
                "status": "success",
                "count": uploaded_count,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        use_hybrid: Optional[bool] = None,
        use_semantic: Optional[bool] = None,
    ) -> List[Document]:
        '''
        Search for similar documents.
        
        :param self: AzureSearchVectorStore instance
        :param query: Search query string
        :type query: str
        :param k: Number of results to return
        :type k: int
        :param use_hybrid: Whether to use hybrid search (vector + keyword)
        :type use_hybrid: Optional[bool]
        :param use_semantic: Whether to use semantic ranking
        :type use_semantic: Optional[bool]
        :return: List of matching LangChain documents with metadata
        :rtype: List[Document]
        '''
        if not self.search_client:
            raise RuntimeError(_SEARCH_CLIENT_NOT_INITIALIZED)

        try:
            logger.info(f"Searching for: '{query}' (k={k})")
            start_time = time.time()

            use_hybrid = use_hybrid if use_hybrid is not None else self.settings.enable_hybrid_search
            use_semantic = use_semantic if use_semantic is not None else self.settings.enable_semantic_search

            # Generate query embedding
            query_embedding = self.openai_service.get_embedding(query)

            # Build search query
            search_kwargs = {
                "top": k,
                "vector_queries": [{
                    "vector": query_embedding,
                    "k_nearest_neighbors": k,
                    "fields": "embedding",
                }],
            }

            # Add hybrid search (keyword search)
            if use_hybrid:
                search_kwargs["search_text"] = query

            # Add semantic ranking
            if use_semantic:
                search_kwargs["query_type"] = "semantic"
                search_kwargs["semantic_configuration_name"] = "semantic-config"

            # Execute search
            results = self.search_client.search(**search_kwargs)

            # Convert to LangChain documents
            documents = []
            for result in results:
                metadata = json.loads(result.get("metadata", "{}"))
                metadata["score"] = result.get("@search.score", 0.0)
                metadata["source"] = result.get("source", "unknown")

                doc = Document(
                    page_content=result["content"],
                    metadata=metadata,
                )
                documents.append(doc)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Found {len(documents)} documents in {elapsed_time:.2f}s "
                f"(hybrid={use_hybrid}, semantic={use_semantic})"
            )

            return documents

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise

    def delete_documents(self, filter_query: Optional[str] = None) -> Dict[str, Any]:
        '''
        Delete documents from the search index.
        
        :param self: AzureSearchVectorStore instance
        :param filter_query: OData filter query (e.g., "source eq 'document.pdf'"), deletes all if None
        :type filter_query: Optional[str]
        :return: Dictionary with deletion status and count
        :rtype: Dict[str, Any]
        '''
        if not self.search_client:
            raise RuntimeError(_SEARCH_CLIENT_NOT_INITIALIZED)

        try:
            logger.info("Starting document deletion")
            if filter_query:
                # Search for documents matching filter
                results = self.search_client.search(
                    search_text="*",
                    filter=filter_query,
                    select=["id"],
                )

                # Delete matching documents
                doc_ids = [{"id": result["id"]} for result in results]
                if doc_ids:
                    self.search_client.delete_documents(documents=doc_ids)
                    logger.info(f"Deleted {len(doc_ids)} documents")
                    return {"status": "success", "count": len(doc_ids)}
                else:
                    logger.info("No documents found matching filter")
                    return {"status": "success", "count": 0}
            else:
                # Delete all documents (recreate index)
                logger.warning("Deleting all documents by recreating index")
                if not self.index_client:
                    raise RuntimeError("Index client not initialized")
                self.index_client.delete_index(self.index_name)
                self._ensure_index_exists()
                return {"status": "success", "count": "all"}

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        '''
        Get index statistics.
        
        :param self: AzureSearchVectorStore instance
        :return: Dictionary with index name, document count, and endpoint
        :rtype: Dict[str, Any]
        '''
        if not self.search_client:
            raise RuntimeError(_SEARCH_CLIENT_NOT_INITIALIZED)

        try:
            # Get document count
            results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0,
            )

            stats = {
                "index_name": self.index_name,
                "document_count": results.get_count(),
                "endpoint": self.settings.search_endpoint,
            }

            logger.info(f"Index stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise


def get_azure_search_vector_store(index_name: Optional[str] = None) -> AzureSearchVectorStore:
    '''
    Get an Azure Search vector store instance.
    
    :param index_name: Name of the search index to use
    :type index_name: Optional[str]
    :return: AzureSearchVectorStore instance
    :rtype: AzureSearchVectorStore
    '''
    return AzureSearchVectorStore(index_name=index_name)
