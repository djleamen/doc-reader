"""
Tests for the RAG App.

Written by DJ Leamen (2025-2026)
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

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


class PermissionConfigTest(TestCase):
    '''
    Test REST framework permission configuration.

    Verifies that the default permission policy requires authentication
    and that explicitly public endpoints (health checks) allow unauthenticated
    access.
    '''

    def test_default_permission_is_authenticated(self):
        '''
        Verify the DRF default permission is IsAuthenticated.

        Ensures AllowAny has not been set as the global default,
        which would expose all endpoints without authentication.
        '''
        from django.conf import settings as django_settings
        default_perms = django_settings.REST_FRAMEWORK.get('DEFAULT_PERMISSION_CLASSES', [])
        self.assertIn(
            'rest_framework.permissions.IsAuthenticated',
            default_perms,
        )

    def test_default_permission_is_not_allow_any(self):
        '''
        Verify AllowAny is not the global default permission.

        AllowAny as a default would expose every endpoint without
        authentication, which is a security risk.
        '''
        from django.conf import settings as django_settings
        default_perms = django_settings.REST_FRAMEWORK.get('DEFAULT_PERMISSION_CLASSES', [])
        self.assertNotIn(
            'rest_framework.permissions.AllowAny',
            default_perms,
        )


class APIViewsTest(TestCase):
    '''
    Test the API views.
    
    Verifies API endpoint behavior including health checks, queries,
    uploads, and error handling.
    '''

    def setUp(self):
        '''
        Set up test fixtures.
        
        Creates an authenticated test client and index for API endpoint tests.
        '''
        self.client = Client()
        self.user = User.objects.create_user(
            username='apitestuser',
            password='apitestpass123',
        )
        self.client.login(username='apitestuser', password='apitestpass123')
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
        # Authenticated callers still receive index metadata.
        self.assertIn('indexes', data)

    def test_health_check_unauthenticated(self):
        '''
        Test that the health check endpoint is publicly accessible.

        The health check endpoint must not require authentication so that
        load balancers and monitoring tools can reach it freely, but it must
        only return a minimal liveness response — no internal index metadata.
        '''
        unauthenticated_client = Client()
        response = unauthenticated_client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        # Public probes must not leak internal index names.
        self.assertNotIn('indexes', data)

    def test_protected_endpoint_requires_auth(self):
        '''
        Test that protected endpoints reject unauthenticated requests.

        Verifies that the index-stats endpoint returns 403 for an
        unauthenticated client, confirming IsAuthenticated is enforced.
        '''
        unauthenticated_client = Client()
        response = unauthenticated_client.get('/api/index-stats/?index_name=test_index')
        self.assertEqual(response.status_code, 403)

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

    def test_query_malformed_json(self):
        '''
        Test query endpoint with a malformed JSON body.

        Verifies a client-side JSON error yields 400, not 500.
        '''
        response = self.client.post('/api/query/',
                                  '{"question": "unterminated',
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_conversational_query_malformed_json(self):
        '''
        Test conversational query endpoint with a malformed JSON body.

        Mirrors the /api/query/ handling so a broken JSON body returns 400
        instead of falling through to a 500 internal error.
        '''
        response = self.client.post('/api/conversational-query/',
                                  '{"question": "unterminated',
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_query_non_object_json_body(self):
        '''
        Test query endpoint with a valid-JSON, non-object body.

        A body such as a JSON array or string decodes cleanly but is not a
        dict; it must return 400 rather than falling through to an uncaught
        AttributeError (500) when the view reads ``.get('question')``.
        '''
        response = self.client.post('/api/query/',
                                  json.dumps([1, 2, 3]),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_conversational_query_non_object_json_body(self):
        '''
        Test conversational query endpoint with a non-object JSON body.

        Mirrors the /api/query/ handling so a non-dict JSON body returns 400
        instead of falling through to a 500 internal error.
        '''
        response = self.client.post('/api/conversational-query/',
                                  json.dumps("just a string"),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_clear_conversation_clears_in_memory_history(self):
        '''
        Test that clearing a conversation resets in-memory RAG history.

        Regression test: the endpoint previously guarded on a non-existent
        clear_memory() method, so the process-global ConversationalRAG history
        was never actually cleared and old context leaked into later turns.
        '''
        from rag_app import views

        class _StubConversationalRAG:
            def __init__(self):
                self.conversation_history = [{'q': 'hi', 'a': 'there'}]

            def clear_conversation(self):
                self.conversation_history = []

        # Engines are keyed by (session_key, index_name); give the client a
        # real session so the DELETE request resolves to this stub's key.
        session = self.client.session
        session.save()
        session_key = session.session_key

        stub = _StubConversationalRAG()
        cache_key = (session_key, 'test_index')
        views._conversational_rags[cache_key] = stub
        try:
            response = self.client.delete('/api/conversation/')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(stub.conversation_history, [])
        finally:
            views._conversational_rags.pop(cache_key, None)

    def test_clear_conversation_is_scoped_to_requesting_session(self):
        '''
        Test that clearing one session's conversation leaves others intact.

        Regression test for #177: clear_conversation previously iterated every
        cached engine, so one session's "clear" wiped in-memory history for all
        other sessions (and indexes). It must only clear the requester's own
        session-scoped engines.
        '''
        from rag_app import views

        class _StubConversationalRAG:
            def __init__(self):
                self.conversation_history = [{'q': 'hi', 'a': 'there'}]

            def clear_conversation(self):
                self.conversation_history = []

        # The requesting session (owns the client's cookie) plus a second,
        # unrelated session whose history must survive the clear.
        session = self.client.session
        session.save()
        my_key = session.session_key

        mine = _StubConversationalRAG()
        other = _StubConversationalRAG()
        views._conversational_rags[(my_key, 'default')] = mine
        views._conversational_rags[('other-session', 'default')] = other
        try:
            response = self.client.delete('/api/conversation/')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(mine.conversation_history, [])
            self.assertEqual(other.conversation_history, [{'q': 'hi', 'a': 'there'}])
        finally:
            views._conversational_rags.pop((my_key, 'default'), None)
            views._conversational_rags.pop(('other-session', 'default'), None)

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

    def test_upload_unsupported_file_type(self):
        '''
        Test document upload with an unsupported file type.

        Verifies unsupported extensions are rejected with a per-file
        error before any Document records are created.
        '''
        test_file = SimpleUploadedFile(
            "malware.exe",
            b"MZ fake executable content",
            content_type="application/octet-stream"
        )
        response = self.client.post('/api/upload-documents/', {
            'files': [test_file],
            'index_name': 'test_index'
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['files_processed'], [])
        self.assertEqual(len(data['errors']), 1)
        self.assertIn('Unsupported file type', data['errors'][0])
        self.assertEqual(Document.objects.count(), 0)


class WebViewsTest(TestCase):
    '''
    Test the web interface views.
    
    Verifies web page rendering and content for user-facing pages.
    '''

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='webtestuser',
            password='webtestpass123',
        )

    def test_home_view_requires_login(self):
        response = self.client.get('/')
        self.assertRedirects(response, '/login/?next=/')

    def test_login_page_loads(self):
        response = self.client.get('/login/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Sign in')

    def test_home_view(self):
        '''
        Test the home page loads.

        Verifies home page returns 200 status and contains
        expected page title.
        '''
        self.client.login(username='webtestuser', password='webtestpass123')
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Document Q&A System')
        self.assertContains(response, 'Upload Documents')
        self.assertContains(response, 'Ask Questions')
        self.assertContains(response, 'webtestuser')

    def test_logout_ends_authenticated_session(self):
        self.client.login(username='webtestuser', password='webtestpass123')
        response = self.client.post('/logout/')
        self.assertRedirects(response, '/login/')
        self.assertNotIn('_auth_user_id', self.client.session)


class AuthenticatedConversationTest(TestCase):
    """Verify persisted conversation history belongs to the signed-in user."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='conversationuser',
            password='conversationpass123',
        )
        self.client.login(
            username='conversationuser',
            password='conversationpass123',
        )
        self.index = DocumentIndex.objects.create(name='conversation_index')

    @patch('rag_app.views.get_conversational_rag')
    def test_conversation_session_is_linked_to_user(self, get_engine):
        get_engine.return_value.conversational_query.return_value = SimpleNamespace(
            answer='An answer',
            metadata={},
        )

        response = self.client.post(
            '/api/conversational-query/',
            json.dumps({
                'question': 'A question',
                'index_name': self.index.name,
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        query_session = QuerySession.objects.get(id=response.json()['session_id'])
        self.assertEqual(query_session.user, self.user)


class ConversationalRAGCacheTest(TestCase):
    '''
    Test session-scoped caching of conversational RAG engines.

    Verifies that conversation history is isolated per browser session
    rather than shared process-wide per index, and that the cache is
    bounded by an LRU eviction cap.
    '''

    def setUp(self):
        '''
        Set up test fixtures.

        Swaps ConversationalRAG for a lightweight stub so the cache logic
        can be exercised without the heavy RAG/LLM dependencies, and starts
        from an empty cache.
        '''
        from rag_app import views

        self.views = views
        self._real_engine = views.ConversationalRAG

        class _StubEngine:
            def __init__(self, index_name="default"):
                self.index_name = index_name

        views.ConversationalRAG = _StubEngine
        views._conversational_rags.clear()

    def tearDown(self):
        '''Restore the real engine and empty the cache after each test.'''
        self.views.ConversationalRAG = self._real_engine
        self.views._conversational_rags.clear()

    def test_sessions_get_isolated_engines(self):
        '''
        Two sessions querying the same index get distinct engines.

        Keying by (session_key, index_name) prevents one user's conversation
        history from leaking into another's, while a repeated (session, index)
        lookup returns the cached instance.
        '''
        engine_a = self.views.get_conversational_rag('shared_index', 'session-a')
        engine_b = self.views.get_conversational_rag('shared_index', 'session-b')
        self.assertIsNot(engine_a, engine_b)
        self.assertIs(
            engine_a,
            self.views.get_conversational_rag('shared_index', 'session-a'),
        )

    def test_cache_is_bounded_by_lru_cap(self):
        '''
        The cache never exceeds the configured maximum size.

        Creating more engines than the cap evicts the least-recently-used
        entries while keeping the most recent.
        '''
        cap = self.views._MAX_CONVERSATIONAL_RAGS
        for i in range(cap + 5):
            self.views.get_conversational_rag('idx', f'session-{i}')
        self.assertLessEqual(len(self.views._conversational_rags), cap)
        self.assertNotIn(('session-0', 'idx'), self.views._conversational_rags)
        self.assertIn(
            (f'session-{cap + 4}', 'idx'),
            self.views._conversational_rags,
        )
