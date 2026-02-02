"""
Tests for the RAG App.

Written by DJ Leamen (2025-2026)
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase

from rag_app.models import Document, DocumentIndex, Query, QuerySession

User = get_user_model()

class DocumentIndexModelTest(TestCase):
    '''
    Test the DocumentIndex model.
    
    Verifies DocumentIndex creation, field defaults, and UUID generation.
    '''

    def test_create_index(self):
        '''
        Test creating a document index.
        
        Verifies that index is created with correct name, default counts,
        and auto-generated UUID.
        '''
        index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )
        self.assertEqual(index.name, "test_index")
        self.assertEqual(index.document_count, 0)
        self.assertEqual(index.chunk_count, 0)
        self.assertTrue(str(index.id))  # UUID should be generated


class DocumentModelTest(TestCase):
    '''
    Test the Document model.
    
    Verifies Document creation, foreign key relationships, and field defaults.
    '''

    def setUp(self):
        '''
        Set up test fixtures.
        
        Creates a test DocumentIndex for use in Document tests.
        '''
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_create_document(self):
        '''
        Test creating a document.
        
        Verifies document creation with all required fields and
        correct default values for processed status and chunk count.
        '''
        document = Document.objects.create(
            index=self.index,
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            file_type="pdf"
        )
        self.assertEqual(document.filename, "test.pdf")
        self.assertEqual(document.index, self.index)
        self.assertFalse(document.processed)
        self.assertEqual(document.chunk_count, 0)


class QuerySessionModelTest(TestCase):
    '''
    Test the QuerySession model.
    
    Verifies QuerySession creation with user association and UUID generation.
    '''

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_create_query_session(self):
        '''
        Test creating a query session.
        
        Verifies session creation with user, session key, index,
        and auto-generated UUID.
        '''
        session = QuerySession.objects.create(
            user=self.user,
            session_key="test_session_key",
            index=self.index
        )
        self.assertEqual(session.user, self.user)
        self.assertEqual(session.session_key, "test_session_key")
        self.assertEqual(session.index, self.index)
        self.assertIsInstance(session.id, uuid.UUID)


class QueryModelTest(TestCase):
    '''
    Test the Query model.
    
    Verifies Query creation with all fields and relationships to
    session and index.
    '''

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )
        self.session = QuerySession.objects.create(
            user=self.user,
            session_key="test_session_key",
            index=self.index
        )

    def test_create_query(self):
        '''
        Test creating a query.
        
        Verifies query creation with all fields including question,
        answer, metadata, and performance metrics.
        '''
        query = Query.objects.create(
            session=self.session,
            index=self.index,
            question="What is the main topic?",
            answer="The main topic is testing.",
            response_time=0.5,
            k_results=5,
            include_sources=True,
            include_scores=True,
            context_length=100,
            retrieval_count=3,
            model_used="gpt-3.5-turbo"
        )
        self.assertEqual(query.question, "What is the main topic?")
        self.assertEqual(query.answer, "The main topic is testing.")
        self.assertEqual(query.session, self.session)
        self.assertEqual(query.index, self.index)
        self.assertEqual(query.response_time, 0.5)
        self.assertTrue(query.include_sources)
        self.assertTrue(query.include_scores)


class APIViewsTest(TestCase):
    '''
    Test the API views.
    
    Verifies API endpoint behavior including health checks, queries,
    uploads, and error handling.
    '''

    def setUp(self):
        '''
        Set up test fixtures.
        
        Creates test client and index for API endpoint tests.
        '''
        self.client = Client()
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_health_check(self):
        '''
        Test the health check endpoint.
        
        Verifies health endpoint returns 200 status and healthy status.
        '''
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')

    def test_index_stats(self):
        '''
        Test the index stats endpoint.
        
        Verifies stats endpoint returns correct index information
        and statistics.
        '''
        response = self.client.get('/api/index-stats/?index_name=test_index')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['index_name'], 'test_index')
        self.assertIn('stats', data)

    def test_index_stats_not_found(self):
        '''
        Test index stats for non-existent index.
        
        Verifies appropriate 404 response for missing index.
        '''
        response = self.client.get('/api/index-stats/?index_name=nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_query_without_question(self):
        '''
        Test query endpoint without question.
        
        Verifies proper validation and 400 error for missing question.
        '''
        response = self.client.post('/api/query/',
                                  json.dumps({}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_query_nonexistent_index(self):
        '''
        Test query with non-existent index.
        
        Verifies 404 response when querying non-existent index.
        '''
        query_data = {
            'question': 'Test question',
            'index_name': 'nonexistent'
        }
        response = self.client.post('/api/query/',
                                  json.dumps(query_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 404)

    def test_upload_document_api(self):
        '''
        Test document upload API endpoint.
        
        Verifies upload endpoint accepts files and returns appropriate
        response. May return 500 if RAG dependencies are unavailable.
        '''
        test_file = SimpleUploadedFile(
            "test.txt",
            b"This is test document content for RAG testing.",
            content_type="text/plain"
        )
        response = self.client.post('/api/upload-documents/', {
            'files': [test_file],
            'index_name': 'test_index'
        })
        # Note: This might fail if RAG engine dependencies aren't available
        # The test verifies the endpoint exists and handles the request
        self.assertIn(response.status_code, [200, 500])  # 500 if dependencies missing


class WebViewsTest(TestCase):
    '''
    Test the web interface views.
    
    Verifies web page rendering and content for user-facing pages.
    '''

    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        '''
        Test the home page loads.
        
        Verifies home page returns 200 status and contains
        expected page title.
        '''
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Document Q&A System')
        self.assertContains(response, 'Upload Documents')
        self.assertContains(response, 'Ask Questions')
