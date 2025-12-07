"""  
Django views and API endpoints for Azure RAG pipeline.
"""

import sys
from pathlib import Path
from typing import Any, Dict, cast

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from loguru import logger
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_app.serializers import (DocumentUploadSerializer,
                                 QueryRequestSerializer)
from src.az_rag_engine import get_azure_rag_engine, ConversationalAzureRAG

sys.path.append(str(Path(__file__).parent.parent / 'src'))


# Constants
INTERNAL_ERROR_MESSAGE = 'An internal error occurred.'

@method_decorator(csrf_exempt, name='dispatch')
class AzureDocumentUploadView(APIView):
    """
    API endpoint for uploading documents to Azure RAG pipeline.
    Uses Azure Document Intelligence for processing.
    """

    def post(self, request):
        """
        Upload and process documents.

        Request body:
        - files: List of file uploads
        - index_name: Optional index name (default from settings)

        Returns:
        - status: success/error
        - results: List of processing results for each file
        """
        try:
            serializer = DocumentUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )

            files = request.FILES.getlist('files')
            # Cast validated_data to dict since it's guaranteed after is_valid()
            validated_data = cast(Dict[str, Any], serializer.validated_data)
            index_name = validated_data.get('index_name')

            if not files:
                return Response(
                    {'error': 'No files provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get Azure RAG engine
            rag_engine = get_azure_rag_engine(index_name=index_name)

            results = []
            for file in files:
                try:
                    # Save file temporarily
                    file_path = default_storage.save(
                        f"temp/azure_{file.name}",
                        ContentFile(file.read())
                    )
                    full_path = default_storage.path(file_path)

                    # Process document
                    result = rag_engine.add_document(full_path)
                    results.append(result)

                    # Cleanup
                    default_storage.delete(file_path)

                except Exception as e:
                    logger.error(f"Failed to process file {file.name}: {e}")
                    results.append({
                        'status': 'error',
                        'file': file.name,
                        'error': 'An internal error occurred while processing this file.'
                    })

            return Response({
                'status': 'success',
                'files_processed': len(results),
                'results': results,
                'pipeline': 'azure',
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return Response(
                {'error': INTERNAL_ERROR_MESSAGE},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class AzureQueryView(APIView):
    """
    API endpoint for querying Azure RAG pipeline.
    Uses Azure AI Search with hybrid search and Azure OpenAI for generation.
    """

    def post(self, request):
        """
        Query the Azure RAG system.

        Request body:
        - question: User question
        - index_name: Optional index name
        - k: Number of documents to retrieve (default: 5)
        - use_hybrid: Use hybrid search (default: true)
        - use_semantic: Use semantic ranking (default: true)
        - include_sources: Include source documents (default: true)
        - include_scores: Include relevance scores (default: true)

        Returns:
        - answer: Generated answer
        - sources: Source documents (if requested)
        - metadata: Query metadata
        """
        try:
            serializer = QueryRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Cast validated_data to dict since it's guaranteed after is_valid()
            validated_data = cast(Dict[str, Any], serializer.validated_data)
            question = validated_data.get('question')

            if not question:
                return Response(
                    {'error': 'Question is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            index_name = validated_data.get('index_name')
            k = validated_data.get('k', 5)
            include_sources = validated_data.get('include_sources', True)
            use_hybrid = request.data.get('use_hybrid', True)
            use_semantic = request.data.get('use_semantic', True)

            # Get Azure RAG engine
            rag_engine = get_azure_rag_engine(index_name=index_name)

            # Execute query
            result = rag_engine.query(
                question=question,
                k=k,
                use_hybrid=use_hybrid,
                use_semantic=use_semantic,
                include_sources=include_sources,
            )

            return Response({
                'answer': result.answer,
                'sources': result.sources if include_sources else [],
                'metadata': {
                    'query': result.query,
                    'elapsed_time': result.elapsed_time,
                    'model': result.model,
                    'search_method': result.search_method,
                    'cache_hit': result.cache_hit,
                    'pipeline': 'azure',
                    **(result.metadata or {}),
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Azure query failed: {e}")
            return Response(
                {'error': INTERNAL_ERROR_MESSAGE},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class AzureConversationalQueryView(APIView):
    """
    API endpoint for conversational queries with Azure RAG pipeline.
    Maintains conversation history across requests.
    """

    def post(self, request):
        """
        Query with conversation history.

        Request body:
        - question: User question
        - index_name: Optional index name
        - k: Number of documents to retrieve
        - use_hybrid: Use hybrid search
        - use_semantic: Use semantic ranking
        - include_sources: Include source documents

        Returns:
        - answer: Generated answer
        - sources: Source documents
        - conversation_history: Recent conversation
        - metadata: Query metadata
        """
        try:
            serializer = QueryRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Cast validated_data to dict since it's guaranteed after is_valid()
            validated_data = cast(Dict[str, Any], serializer.validated_data)
            question = validated_data.get('question')

            if not question:
                return Response(
                    {'error': 'Question is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            index_name = validated_data.get('index_name')
            k = validated_data.get('k', 5)
            include_sources = validated_data.get('include_sources', True)
            use_hybrid = request.data.get('use_hybrid', True)
            use_semantic = request.data.get('use_semantic', True)

            # Get conversational Azure RAG engine
            rag_engine = cast(
                ConversationalAzureRAG,
                get_azure_rag_engine(
                    index_name=index_name,
                    conversational=True
                )
            )

            # Execute query with history
            result = rag_engine.query_with_history(
                question=question,
                k=k,
                use_hybrid=use_hybrid,
                use_semantic=use_semantic,
                include_sources=include_sources,
            )

            return Response({
                'answer': result.answer,
                'sources': result.sources if include_sources else [],
                'conversation_history': rag_engine.get_history(),
                'metadata': {
                    'query': result.query,
                    'elapsed_time': result.elapsed_time,
                    'model': result.model,
                    'search_method': result.search_method,
                    'cache_hit': result.cache_hit,
                    'pipeline': 'azure',
                    **(result.metadata or {}),
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Azure conversational query failed: {e}")
            return Response(
                {'error': INTERNAL_ERROR_MESSAGE},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def azure_index_stats(request):
    """
    Get Azure RAG index statistics.

    Query params:
    - index_name: Optional index name

    Returns:
    - index statistics and configuration
    """
    try:
        index_name = request.GET.get('index_name')

        # Get Azure RAG engine
        rag_engine = get_azure_rag_engine(index_name=index_name)

        # Get stats
        stats = rag_engine.get_stats()

        return JsonResponse({
            'status': 'success',
            'stats': stats,
            'pipeline': 'azure',
        })

    except Exception as e:
        logger.error(f"Failed to get Azure stats: {e}")
        return JsonResponse(
            {'error': INTERNAL_ERROR_MESSAGE},
            status=500
        )


@api_view(['POST'])
@csrf_exempt
def azure_clear_cache(request):
    """
    Clear Azure RAG query cache.

    Request body:
    - index_name: Optional index name

    Returns:
    - status message
    """
    try:
        index_name = request.data.get('index_name')

        # Get Azure RAG engine
        rag_engine = get_azure_rag_engine(index_name=index_name)

        # Clear cache
        rag_engine.clear_cache()

        return JsonResponse({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'pipeline': 'azure',
        })

    except Exception as e:
        logger.error(f"Failed to clear Azure cache: {e}")
        return JsonResponse(
            {'error': INTERNAL_ERROR_MESSAGE},
            status=500
        )


@api_view(['POST'])
@csrf_exempt
def azure_clear_conversation(request):
    """
    Clear conversation history.

    Request body:
    - index_name: Optional index name

    Returns:
    - status message
    """
    try:
        index_name = request.data.get('index_name')

        # Get conversational Azure RAG engine
        rag_engine = cast(
            ConversationalAzureRAG,
            get_azure_rag_engine(
                index_name=index_name,
                conversational=True
            )
        )

        # Clear history
        rag_engine.clear_history()

        return JsonResponse({
            'status': 'success',
            'message': 'Conversation history cleared successfully',
            'pipeline': 'azure',
        })

    except Exception as e:
        logger.error(f"Failed to clear Azure conversation: {e}")
        return JsonResponse(
            {'error': INTERNAL_ERROR_MESSAGE},
            status=500
        )


@api_view(['GET'])
def azure_health_check(request):
    """
    Health check endpoint for Azure RAG pipeline.
    Validates Azure services connectivity.

    Returns:
    - configuration status
    - service connectivity status
    """
    try:
        # Get Azure RAG engine
        rag_engine = get_azure_rag_engine()

        # Validate configuration
        validation = rag_engine.validate_configuration()

        # Determine overall health
        all_connected = (
            validation['openai_connection'] and
            validation['search_connection'] and
            validation['document_intelligence_connection']
        )

        health_status = 'healthy' if all_connected else 'degraded'
        http_status = 200 if all_connected else 503

        return JsonResponse({
            'status': health_status,
            'pipeline': 'azure',
            'services': {
                'openai': validation['openai_connection'],
                'search': validation['search_connection'],
                'document_intelligence': validation['document_intelligence_connection'],
            },
            'configuration_valid': validation['configuration_valid'],
            'missing_config': validation['missing_config'],
        }, status=http_status)

    except Exception as _e:
        logger.exception("Azure health check failed")
        return JsonResponse({
            'status': 'unhealthy',
            'error': INTERNAL_ERROR_MESSAGE,
            'pipeline': 'azure',
        }, status=503)
