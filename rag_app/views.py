"""
Views for the RAG Document Q&A system.

Written by DJ Leamen (2025-2026)
"""

import json
import sys
import time
import logging
from pathlib import Path

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from src.rag_engine import ConversationalRAG, RAGEngine
from src.config import settings as rag_settings

from .models import Document, DocumentIndex, Query, QuerySession

sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Global RAG engine instances (similar to FastAPI implementation)
_rag_engines = {}
_conversational_rags = {}


def get_rag_engine(index_name: str = "default"):
    '''
    Get or create RAG engine instance for given index.
    
    :param index_name: Name of the document index
    :return: RAGEngine instance for the specified index
    '''
    if index_name not in _rag_engines:
        _rag_engines[index_name] = RAGEngine(index_name=index_name)
    return _rag_engines[index_name]


def get_conversational_rag(index_name: str = "default"):
    '''
    Get or create conversational RAG engine instance for given index.
    
    :param index_name: Name of the document index
    :return: ConversationalRAG instance for the specified index
    '''
    if index_name not in _conversational_rags:
        _conversational_rags[index_name] = ConversationalRAG(
            index_name=index_name)
    return _conversational_rags[index_name]


class IndexView(TemplateView):
    '''
    Main web interface view.
    
    Renders the RAG Document Q&A web interface with document indexes,
    recent documents, and supported file formats.
    '''
    template_name = 'rag_app/index.html'

    def get_context_data(self, **kwargs):
        '''
        Get context data for template rendering.
        
        :param kwargs: Additional context arguments
        :return: Template context dictionary
        '''
        context = super().get_context_data(**kwargs)
        context.update({
            'indexes': DocumentIndex.objects.all(),
            'recent_documents': Document.objects.filter(processed=True)[:10],
            'supported_formats': ['pdf', 'docx', 'txt', 'md'],
            'max_file_size_mb': 100,
        })
        return context


class HomeView(IndexView):
    '''
    Alias for IndexView to match URLs pattern.
    
    Inherits all functionality from IndexView.
    '''
    # Inherits all functionality from IndexView


class TestView(TemplateView):
    '''
    Simple test view to verify Django is working.
    
    Renders a basic test template for debugging purposes.
    '''
    template_name = 'simple_test.html'


@method_decorator(csrf_exempt, name='dispatch')
class DocumentUploadView(APIView):
    '''
    Handle document uploads via API.
    
    Accepts file uploads, processes documents using RAG engine,
    and stores metadata in the database.
    '''

    def post(self, request):
        '''
        Handle document upload via API.
        
        :param request: HTTP request with uploaded files
        :return: JSON response with upload status and results
        '''
        try:
            # Get or create index
            files = request.FILES.getlist('files')
            if not files:
                return Response({
                    'error': 'No files provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            index_name = request.POST.get('index_name', 'default')

            # Get or create DocumentIndex
            index, _ = DocumentIndex.objects.get_or_create(name=index_name)

            # Process uploaded files
            processed_files = []
            errors = []
            total_chunks = 0

            rag_engine = get_rag_engine(index_name)

            for file in files:
                try:
                    # Validate file
                    max_file_size = rag_settings.max_document_size_mb * 1024 * 1024
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
                        file_type=file.name.split('.')[-1].lower(),
                        chunk_count=0
                    )

                    try:
                        # Process document
                        chunks_added = rag_engine.add_documents([file_path])
                        document.chunk_count = chunks_added or 0
                        document.processed = True
                        document.save()

                        processed_files.append(file.name)
                        total_chunks += (chunks_added or 0)

                    except (OSError, IOError, ValueError, RuntimeError) as e:
                        document.processing_error = str(e)
                        document.save()
                        errors.append(f"{file.name}: {str(e)}")

                    finally:
                        # Clean up temp file
                        if default_storage.exists(temp_path):
                            default_storage.delete(temp_path)

                except (OSError, IOError, ValueError, RuntimeError) as e:
                    errors.append(f"{file.name}: {str(e)}")

            # Update index statistics
            index.document_count = Document.objects.filter(
                index=index, processed=True).count()
            index.chunk_count = sum(doc.chunk_count or 0 for doc in Document.objects.filter(
                index=index, processed=True))
            index.save()

            return Response({
                'message': f'Processed {len(processed_files)} files successfully',
                'files_processed': processed_files,
                'total_chunks_added': total_chunks,
                'errors': errors
            })

        except (OSError, IOError, ValueError, RuntimeError) as e:
            return Response({
                'error': f'Upload failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class QueryView(APIView):
    '''
    Handle document queries.
    
    Processes natural language questions against indexed documents
    and returns relevant answers with source citations.
    '''

    def post(self, request):
        '''Handle document queries.'''
        try:
            data = json.loads(
                request.body) if request.body else request.POST.dict()

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

            except (OSError, IOError, ValueError, RuntimeError) as e:
                return Response({
                    'error': f'Query failed: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except json.JSONDecodeError:
            return Response({
                'error': 'Invalid JSON in request body'
            }, status=status.HTTP_400_BAD_REQUEST)
        except (OSError, IOError, ValueError, RuntimeError) as e:
            return Response({
                'error': f'Request failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class ConversationalQueryView(APIView):
    '''
    Handle conversational queries.
    
    Processes queries while maintaining conversation history
    for multi-turn dialogues.
    '''

    def post(self, request):
        '''Handle document queries.'''
        try:
            data = json.loads(
                request.body) if request.body else request.POST.dict()

            question = data.get('question', '').strip()
            if not question:
                return Response({
                    'error': 'Question is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            index_name = data.get('index_name', 'default')

            # Ensure session exists and has a key
            if not request.session.session_key:
                request.session.create()
            session_key = request.session.session_key

            # Get or create session
            try:
                index = DocumentIndex.objects.get(name=index_name)
                query_session, _ = QuerySession.objects.get_or_create(
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

            except (OSError, IOError, ValueError, RuntimeError):
                logger = logging.getLogger(__name__)
                logger.error("Conversational query failed", exc_info=True)
                return Response({
                    'error': 'An internal server error occurred during the conversational query.'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except (OSError, IOError, ValueError, RuntimeError):
            logger = logging.getLogger(__name__)
            logger.error("Request processing failed", exc_info=True)
            return Response({
                'error': 'An internal server error occurred while processing the request.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def index_stats(request):
    '''
    Get index statistics.
    
    :param request: HTTP request with optional index_name parameter
    :return: JSON response with index statistics and metadata
    '''
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
                        'question': (query.question[:100] + '...'
                                     if len(query.question) > 100
                                     else query.question),
                        'created_at': query.created_at.isoformat(),
                        'response_time': query.response_time
                    } for query in recent_queries
                ]
            }
        }

        return Response(stats)

    except (OSError, IOError, ValueError, RuntimeError):
        logger = logging.getLogger(__name__)
        logger.error("Failed to get stats", exc_info=True)
        return Response({
            'error': 'An internal server error occurred while retrieving stats.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_protect
@api_view(['DELETE'])
def clear_conversation(request):
    '''
    Clear conversation history for current session.
    
    :param request: HTTP request with session information
    :return: JSON response with success or error status
    '''
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

    except (OSError, IOError, ValueError, RuntimeError):
        logger = logging.getLogger(__name__)
        logger.error("Failed to clear conversation", exc_info=True)
        return Response({
            'error': 'An internal server error occurred.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_protect
@api_view(['DELETE'])
def clear_documents(request):
    '''
    Clear all documents from an index.
    
    :param request: HTTP request with index_name parameter
    :return: JSON response with deletion status
    '''
    logger_instance = logging.getLogger(__name__)
    try:
        index_name = request.query_params.get(
            'index_name') or request.data.get('index_name', 'default')

        # Get the index
        try:
            index = DocumentIndex.objects.get(name=index_name)
        except DocumentIndex.DoesNotExist:
            return Response({
                'error': f'Index "{index_name}" not found'
            }, status=status.HTTP_404_NOT_FOUND)

        # Clear the RAG engine and vector store
        rag_engine = get_rag_engine(index_name)
        rag_engine.clear_index()

        # Clear RAG engine cache
        if index_name in _rag_engines:
            del _rag_engines[index_name]
        if index_name in _conversational_rags:
            del _conversational_rags[index_name]

        # Delete all documents from database
        Document.objects.filter(index=index).delete()

        # Update index stats
        index.document_count = 0
        index.chunk_count = 0
        index.save()

        logger_instance.info(
            "Cleared all documents from index '%s'", index_name)

        return Response({
            'message': f'Successfully cleared all documents from index "{index_name}"',
            'index_name': index_name
        })

    except (OSError, IOError, ValueError, RuntimeError):
        logger_instance.error("Failed to clear documents", exc_info=True)
        return Response({
            'error': 'Failed to clear documents due to a server error.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    '''
    Health check endpoint.
    
    :param request: HTTP request (unused)
    :return: JSON response with service health status
    '''
    # pylint: disable=unused-argument
    indexes = [index.name for index in DocumentIndex.objects.only('name')]

    return Response({
        'status': 'healthy',
        'message': 'RAG Document Q&A API is running',
        'indexes': indexes
    })
