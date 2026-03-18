"""
Django views and API endpoints for Azure RAG pipeline.
Uses Azure Document Intelligence for document processing,
Azure AI Search for retrieval, and Azure OpenAI for answer generation.

Written by DJ Leamen (2025-2026)
"""

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
from src.az_rag_engine import (
    ConversationalAzureRAG,
    _azure_rag_engines,
    get_azure_rag_engine,
)


# Constants
INTERNAL_ERROR_MESSAGE = 'An internal error occurred.'


def _normalize_azure_sources(result) -> tuple[list[dict[str, Any]], list[float]]:
    """Match the classic API response shape expected by the shared UI."""
    source_documents = []
    confidence_scores = []

    for source in result.sources:
        metadata = dict(source.get('metadata') or {})
        score = float(source.get('score', metadata.get('score', 0.0)) or 0.0)
        source_documents.append({
            'content': source.get('content', ''),
            'metadata': metadata,
        })
        confidence_scores.append(score)

    return source_documents, confidence_scores


def _build_azure_metadata(result, retrieval_count: int) -> dict[str, Any]:
    """Provide stable metadata keys across classic and Azure pipelines."""
    extra_metadata = result.metadata or {}
    return {
        'retrieval_count': retrieval_count,
        'context_length': 0,
        'model_used': result.model,
        'prompt_length': 0,
        'elapsed_time': result.elapsed_time,
        'search_method': result.search_method,
        'cache_hit': result.cache_hit,
        'pipeline': 'azure',
        **extra_metadata,
    }

@method_decorator(csrf_exempt, name='dispatch')
class AzureDocumentUploadView(APIView):
    """
    API endpoint for uploading documents to Azure RAG pipeline.
    Uses Azure Document Intelligence for processing.
    """

    def post(self, request):
        '''
        Upload documents to Azure RAG system.
        
        :param request: HTTP request with uploaded files
        :return: JSON response with upload status and results
        '''
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

            processed_files = []
            errors = []
            total_chunks_added = 0
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
                    if result.get('status') == 'success':
                        processed_files.append(file.name)
                        total_chunks_added += int(
                            result.get('indexed_count') or result.get('chunks') or 0
                        )
                    else:
                        errors.append(
                            f"{file.name}: {result.get('error', INTERNAL_ERROR_MESSAGE)}"
                        )

                    # Cleanup
                    default_storage.delete(file_path)

                except Exception as e:
                    logger.error(f"Failed to process file {file.name}: {e}")
                    errors.append(
                        f"{file.name}: An internal error occurred while processing this file."
                    )

            return Response({
                'message': f'Processed {len(processed_files)} files successfully',
                'files_processed': processed_files,
                'total_chunks_added': total_chunks_added,
                'errors': errors,
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
        '''
        Execute a query against the Azure RAG system.
        
        :param request: HTTP request with query parameters
        :return: JSON response with answer and source documents
        '''
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

            source_documents, confidence_scores = _normalize_azure_sources(result)

            return Response({
                'query': question,
                'answer': result.answer,
                'source_documents': source_documents if include_sources else [],
                'confidence_scores': confidence_scores if include_sources else [],
                'metadata': _build_azure_metadata(result, len(source_documents)),
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
        '''
        Execute a conversational query against the Azure RAG system.
        
        :param request: HTTP request with query and conversation history
        :return: JSON response with answer and source documents
        '''
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

            source_documents, confidence_scores = _normalize_azure_sources(result)

            return Response({
                'query': question,
                'answer': result.answer,
                'source_documents': source_documents if include_sources else [],
                'confidence_scores': confidence_scores if include_sources else [],
                'conversation_history': rag_engine.get_history(),
                'metadata': _build_azure_metadata(result, len(source_documents)),
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Azure conversational query failed: {e}")
            return Response(
                {'error': INTERNAL_ERROR_MESSAGE},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def azure_index_stats(request):
    '''
    Get statistics for a specific Azure RAG index.
    
    :param request: HTTP request object
    :return: HTTP response with index statistics
    '''
    try:
        index_name = request.GET.get('index_name')

        # Get Azure RAG engine
        rag_engine = get_azure_rag_engine(index_name=index_name)

        # Get stats
        stats = rag_engine.get_stats()

        return JsonResponse({
            'index_name': index_name or stats.get('index_name'),
            'stats': stats,
            'pipeline': 'azure',
        })

    except Exception as e:
        logger.error(f"Failed to get Azure stats: {e}")
        return JsonResponse(
            {'error': INTERNAL_ERROR_MESSAGE},
            status=500
        )


@api_view(['POST', 'DELETE'])
def azure_clear_cache(request):
    '''
    Clear the Azure RAG engine cache.
    
    :param request: HTTP request object
    :return: HTTP response with cache clear status
    '''
    try:
        index_name = request.data.get('index_name') or request.query_params.get('index_name')

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


@api_view(['POST', 'DELETE'])
def azure_clear_conversation(request):
    '''
    Clear conversation history.
    
    :param request: HTTP request object
    :return: HTTP response with conversation clear status
    '''
    try:
        index_name = request.data.get('index_name') or request.query_params.get('index_name')

        if index_name:
            rag_engine = cast(
                ConversationalAzureRAG,
                get_azure_rag_engine(
                    index_name=index_name,
                    conversational=True
                )
            )
            rag_engine.clear_history()
        else:
            for cache_key, rag_engine in _azure_rag_engines.items():
                if cache_key.endswith(':conv') and isinstance(rag_engine, ConversationalAzureRAG):
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
    '''
    Health check endpoint for Azure RAG pipeline.
    
    :param request: HTTP request object
    :return: HTTP response with health status
    '''
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
