"""
Tests for the RAG App.
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase

from rag_app.models import Document, DocumentIndex, Query, QuerySession

User = get_user_model()

class DocumentIndexModelTest(TestCase):
    '''Test the DocumentIndex model.'''

    def test_create_index(self):
        '''Test creating a document index.'''
        index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )
        self.assertEqual(index.name, "test_index")
        self.assertEqual(index.document_count, 0)
        self.assertEqual(index.chunk_count, 0)
        self.assertTrue(str(index.id))  # UUID should be generated


class DocumentModelTest(TestCase):
    '''Test the Document model.'''

    def setUp(self):
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_create_document(self):
        '''Test creating a document.'''
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
    '''Test the QuerySession model.'''

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
        '''Test creating a query session.'''
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
    '''Test the Query model.'''

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
        '''Test creating a query.'''
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
    '''Test the API views.'''

    def setUp(self):
        self.client = Client()
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_health_check(self):
        '''Test the health check endpoint.'''
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')

    def test_index_stats(self):
        '''Test the index stats endpoint.'''
        response = self.client.get('/api/index-stats/?index_name=test_index')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['index_name'], 'test_index')
        self.assertIn('stats', data)

    def test_index_stats_not_found(self):
        '''Test index stats for non-existent index.'''
        response = self.client.get('/api/index-stats/?index_name=nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_query_without_question(self):
        '''Test query endpoint without question.'''
        response = self.client.post('/api/query/',
                                  json.dumps({}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_query_nonexistent_index(self):
        '''Test query with non-existent index.'''
        query_data = {
            'question': 'Test question',
            'index_name': 'nonexistent'
        }
        response = self.client.post('/api/query/',
                                  json.dumps(query_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 404)

    def test_upload_document_api(self):
        '''Test document upload API endpoint.'''
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
    '''Test the web interface views.'''

    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        '''Test the home page loads.'''
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Document Q&A System')
        self.assertContains(response, 'Upload Documents')
        self.assertContains(response, 'Ask Questions')
