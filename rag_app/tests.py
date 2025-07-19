"""
Tests for the RAG App.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from .models import DocumentIndex, Document, QuerySession, Query
import json


class DocumentIndexModelTest(TestCase):
    """Test the DocumentIndex model."""

    def test_create_index(self):
        """Test creating a document index."""
        index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )
        self.assertEqual(index.name, "test_index")
        self.assertEqual(index.document_count, 0)
        self.assertEqual(index.chunk_count, 0)
        self.assertTrue(str(index.id))  # UUID should be generated


class DocumentModelTest(TestCase):
    """Test the Document model."""

    def setUp(self):
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_create_document(self):
        """Test creating a document."""
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


class APIViewsTest(TestCase):
    """Test the API views."""

    def setUp(self):
        self.client = Client()
        self.index = DocumentIndex.objects.create(
            name="test_index",
            description="Test index"
        )

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')

    def test_index_stats(self):
        """Test the index stats endpoint."""
        response = self.client.get('/api/index-stats/?index_name=test_index')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['index_name'], 'test_index')
        self.assertIn('stats', data)

    def test_index_stats_not_found(self):
        """Test index stats for non-existent index."""
        response = self.client.get('/api/index-stats/?index_name=nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_query_without_question(self):
        """Test query endpoint without question."""
        response = self.client.post('/api/query/',
                                  json.dumps({}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_query_nonexistent_index(self):
        """Test query with non-existent index."""
        query_data = {
            'question': 'Test question',
            'index_name': 'nonexistent'
        }
        response = self.client.post('/api/query/',
                                  json.dumps(query_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 404)


class WebViewsTest(TestCase):
    """Test the web interface views."""

    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        """Test the home page loads."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Document Q&A System')
        self.assertContains(response, 'Upload Documents')
        self.assertContains(response, 'Ask Questions')
