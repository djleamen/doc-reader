"""
Views for the RAG Document Q&A system.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List
from pathlib import Path

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DocumentIndex, Document, QuerySession, Query
from .serializers import QueryRequestSerializer, DocumentUploadSerializer

# Lazy import to avoid Django startup issues
def get_rag_modules():
    """Lazy import of RAG modules."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from src.rag_engine import RAGEngine, ConversationalRAG
    from src.document_processor import DocumentProcessor
    return RAGEngine, ConversationalRAG, DocumentProcessor


# Global RAG engine instances (similar to FastAPI implementation)
_rag_engines = {}
_conversational_rags = {}


def get_rag_engine(index_name: str = "default"):
    """Get or create RAG engine instance for given index."""
    RAGEngine, _, _ = get_rag_modules()
    if index_name not in _rag_engines:
        _rag_engines[index_name] = RAGEngine(index_name=index_name)
    return _rag_engines[index_name]


def get_conversational_rag(index_name: str = "default"):
    """Get or create conversational RAG engine instance for given index."""
    _, ConversationalRAG, _ = get_rag_modules()
    if index_name not in _conversational_rags:
        _conversational_rags[index_name] = ConversationalRAG(index_name=index_name)
    return _conversational_rags[index_name]


class HomeView(TemplateView):
    """Main web interface view."""
    template_name = 'rag_app/index.html'

    def get_context_data(self, **kwargs):
        # Import here to avoid circular imports
        from src.config import settings as rag_settings
        context = super().get_context_data(**kwargs)
        context.update({
            'indexes': DocumentIndex.objects.all(),
            'recent_documents': Document.objects.filter(processed=True)[:10],
            'supported_formats': rag_settings.supported_formats_list,
            'max_file_size_mb': rag_settings.max_document_size_mb,
        })
        return context


@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    """Handle document uploads via API."""

    def post(self, request):
        try:
            _, _, DocumentProcessor = get_rag_modules()

            # Get or create index
            index_name = request.POST.get('index_name', 'default')
            index, created = DocumentIndex.objects.get_or_create(
                name=index_name,
                defaults={'description': f'Document index: {index_name}'}
            )

            files = request.FILES.getlist('files')
            if not files:
                return Response({
                    'error': 'No files provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process uploaded files
            processed_files = []
            errors = []
            total_chunks = 0

            rag_engine = get_rag_engine(index_name)
            document_processor = DocumentProcessor()

            for file in files:
                try:
                    # Import here to avoid circular imports
                    from src.config import settings as rag_settings

                    # Validate file
                    max_file_size = rag_settings.max_document_size_mb * 1024 * 1024  # Convert to bytes
                    if file.size > max_file_size:
                        errors.append(f"{file.name}: File too large")
                        continue

                    # Save file temporarily
                    temp_path = default_storage.save(
                        f"temp/{file.name}",
                        ContentFile(file.read())
                    )
                    file_path = default_storage.path(temp_path)

                    # Create document record
                    document = Document.objects.create(
                        index=index,
                        filename=file.name,
                        original_filename=file.name,
                        file_path=file_path,
                        file_size=file.size,
                        file_type=file.name.split('.')[-1].lower()
                    )

                    try:
                        # Process document
                        chunks_added = rag_engine.add_documents([file_path])
                        document.chunk_count = chunks_added
                        document.processed = True
                        document.save()

                        processed_files.append(file.name)
                        total_chunks += chunks_added

                    except Exception as e:
                        document.processing_error = str(e)
                        document.save()
                        errors.append(f"{file.name}: {str(e)}")

                    finally:
                        # Clean up temp file
                        if default_storage.exists(temp_path):
                            default_storage.delete(temp_path)

                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")

            # Update index statistics
            index.document_count = Document.objects.filter(index=index, processed=True).count()
            index.chunk_count = sum(doc.chunk_count for doc in Document.objects.filter(index=index, processed=True))
            index.save()

            return Response({
                'message': f'Processed {len(processed_files)} files successfully',
                'files_processed': processed_files,
                'total_chunks_added': total_chunks,
                'errors': errors
            })

        except Exception as e:
            return Response({
                'error': f'Upload failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class QueryView(APIView):
    """Handle document queries."""

    def post(self, request):
        try:
            data = json.loads(request.body) if request.body else request.POST.dict()

            # Import here to avoid circular imports
            from src.config import settings as rag_settings

            question = data.get('question', '').strip()
            if not question:
                return Response({
                    'error': 'Question is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            index_name = data.get('index_name', 'default')
            k = data.get('k', rag_settings.top_k_results)
            include_sources = data.get('include_sources', True)
            include_scores = data.get('include_scores', True)

            # Get index
            try:
                index = DocumentIndex.objects.get(name=index_name)
            except DocumentIndex.DoesNotExist:
                return Response({
                    'error': f'Index "{index_name}" not found'
                }, status=status.HTTP_404_NOT_FOUND)

            # Perform query
            start_time = time.time()
            rag_engine = get_rag_engine(index_name)

            try:
                result = rag_engine.query(
                    question=question,
                    k=k,
                    include_sources=include_sources,
                    include_scores=include_scores
                )
                response_time = time.time() - start_time

                # Save query record
                Query.objects.create(
                    index=index,
                    question=question,
                    answer=result.answer,
                    k_results=k,
                    include_sources=include_sources,
                    include_scores=include_scores,
                    response_time=response_time,
                    context_length=result.metadata.get('context_length', 0),
                    retrieval_count=result.metadata.get('retrieval_count', 0),
                    model_used=result.metadata.get('model_used', '')
                )

                # Prepare response
                response_data = {
                    'query': question,
                    'answer': result.answer,
                    'metadata': result.metadata
                }

                if include_sources and result.source_documents:
                    response_data['source_documents'] = [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        } for doc in result.source_documents
                    ]

                if include_scores and hasattr(result, 'confidence_scores'):
                    response_data['confidence_scores'] = result.confidence_scores

                return Response(response_data)

            except Exception as e:
                return Response({
                    'error': f'Query failed: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except json.JSONDecodeError:
            return Response({
                'error': 'Invalid JSON in request body'
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                'error': f'Request failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class ConversationalQueryView(APIView):
    """Handle conversational queries."""

    def post(self, request):
        try:
            data = json.loads(request.body) if request.body else request.POST.dict()

            question = data.get('question', '').strip()
            if not question:
                return Response({
                    'error': 'Question is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            index_name = data.get('index_name', 'default')
            session_key = request.session.session_key or request.session.create()

            # Get or create session
            try:
                index = DocumentIndex.objects.get(name=index_name)
                query_session, created = QuerySession.objects.get_or_create(
                    session_key=session_key,
                    index=index
                )
            except DocumentIndex.DoesNotExist:
                return Response({
                    'error': f'Index "{index_name}" not found'
                }, status=status.HTTP_404_NOT_FOUND)

            # Perform conversational query
            start_time = time.time()
            conversational_rag = get_conversational_rag(index_name)

            try:
                result = conversational_rag.conversational_query(question)
                response_time = time.time() - start_time

                # Save query record
                Query.objects.create(
                    session=query_session,
                    index=index,
                    question=question,
                    answer=result.answer,
                    response_time=response_time,
                    context_length=result.metadata.get('context_length', 0),
                    retrieval_count=result.metadata.get('retrieval_count', 0),
                    model_used=result.metadata.get('model_used', '')
                )

                return Response({
                    'query': question,
                    'answer': result.answer,
                    'metadata': result.metadata,
                    'session_id': str(query_session.id)
                })

            except Exception as e:
                return Response({
                    'error': f'Conversational query failed: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            return Response({
                'error': f'Request failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def index_stats(request):
    """Get index statistics."""
    try:
        index_name = request.GET.get('index_name', 'default')

        try:
            index = DocumentIndex.objects.get(name=index_name)
        except DocumentIndex.DoesNotExist:
            return Response({
                'error': f'Index "{index_name}" not found'
            }, status=status.HTTP_404_NOT_FOUND)

        recent_queries = Query.objects.filter(index=index)[:5]

        stats = {
            'index_name': index.name,
            'stats': {
                'document_count': index.document_count,
                'chunk_count': index.chunk_count,
                'created_at': index.created_at.isoformat(),
                'updated_at': index.updated_at.isoformat(),
                'recent_documents': [
                    {
                        'filename': doc.original_filename,
                        'uploaded_at': doc.uploaded_at.isoformat(),
                        'chunks': doc.chunk_count
                    } for doc in Document.objects.filter(index=index, processed=True)[:5]
                ],
                'recent_queries': [
                    {
                        'question': query.question[:100] + '...' if len(query.question) > 100 else query.question,
                        'created_at': query.created_at.isoformat(),
                        'response_time': query.response_time
                    } for query in recent_queries
                ]
            }
        }

        return Response(stats)

    except Exception as e:
        return Response({
            'error': f'Failed to get stats: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
def clear_conversation(request):
    """Clear conversation history for current session."""
    try:
        session_key = request.session.session_key
        if not session_key:
            return Response({'message': 'No active session found'})

        # Clear conversational RAG memory
        for conv_rag in _conversational_rags.values():
            if hasattr(conv_rag, 'clear_memory'):
                conv_rag.clear_memory()

        # Delete session queries
        QuerySession.objects.filter(session_key=session_key).delete()

        return Response({'message': 'Conversation cleared successfully'})

    except Exception as e:
        return Response({
            'error': f'Failed to clear conversation: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'message': 'RAG Document Q&A API is running',
        'indexes': list(DocumentIndex.objects.values_list('name', flat=True))
    })
