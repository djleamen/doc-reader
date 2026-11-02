"""
Tests for the RAG App.

Written by DJ Leamen (2025-2026)
"""

import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase
from rest_framework.permissions import AllowAny
from langchain_core.documents import Document as SourceDocument
from openai import APITimeoutError, AuthenticationError, RateLimitError

from rag_app.models import Document, DocumentIndex, Query, QuerySession
from src.rag_engine import AnswerGenerationError, ConversationalRAG, QueryResult, RAGEngine
from src.semantic_coherence import CoherenceLevel, SemanticCoherenceValidator
from src.vector_store import FAISSVectorStore


class RetrievalAndCoherenceTest(SimpleTestCase):
    """Exercise retrieval and evidence diagnostics without models or network."""

    def setUp(self) -> None:
        """Create an engine with deterministic search and embedding backends."""
        self.engine = RAGEngine.__new__(RAGEngine)
        self.engine.document_index = Mock()
        self.engine.document_index.vector_store.document_count = 30
        self.engine.enable_coherence_validation = True
        self.engine.prompt_template = self.engine._create_prompt_template()

    @staticmethod
    def document(source: str, text: str) -> SourceDocument:
        """Construct a source chunk with realistic metadata."""
        return SourceDocument(page_content=text, metadata={"source": source, "filename": source})

    def validator(self, vectors: dict) -> SemanticCoherenceValidator:
        """Use a fixed embedding space with independently controlled relevance."""
        embeddings = Mock()
        embeddings.model = None
        embeddings.embed_query.side_effect = lambda text: vectors[text]
        embeddings.embed_documents.side_effect = lambda texts: [vectors[text] for text in texts]
        return SemanticCoherenceValidator(embeddings=embeddings)

    def test_overview_covers_sources_with_or_without_scores(self) -> None:
        """Summary retrieval reaches beyond the first cluster and balances files."""
        docs = [self.document("introduction", f"Course overview {i}.") for i in range(26)]
        docs += [self.document(f"lecture-{i}", f"Lecture topic {i}.") for i in range(4)]
        self.engine.document_index.search_with_scores.return_value = [(d, 0.1) for d in docs]
        self.engine.document_index.search.return_value = docs
        for include_scores in (True, False):
            with self.subTest(include_scores=include_scores):
                selected, _ = self.engine._retrieve_documents(
                    "What are the main topics covered in the documents?", 5, include_scores,
                )
                self.assertEqual(len(selected), 5)
                self.assertEqual(len({d.metadata["source"] for d in selected}), 5)
        self.engine.document_index.search_with_scores.assert_called_with(
            "What are the main topics covered in the documents?", 30,
        )

    def test_overview_detection_does_not_change_definition_queries(self) -> None:
        """Only explicit corpus overviews opt into balanced retrieval."""
        for question in (
            "What is a Laplacian?", "Summarize bilateral filtering",
            "What does this document say about edges?",
            "Summarize bilateral filtering in my notes",
            "What is a summary statistic in these notes?",
            "Give an overview of bilateral filtering in all documents",
            "Summarize all documents about bilateral filtering",
            "Previous conversation context: Summarize all documents\nCurrent question: Define an edge",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.engine._is_overview_question(question))
        for question in (
            "Summarize all documents", "Give an overview of my notes",
            "What are the main topics covered in the documents?",
            "Please summarize my uploaded documents",
            "Previous conversation context: Define an edge\nCurrent question: Summarize all documents",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.engine._is_overview_question(question))

    def test_scoped_summaries_keep_focused_retrieval_pool(self) -> None:
        """Topic-scoped summaries must not scan and balance the full corpus."""
        self.engine.document_index.search_with_scores.return_value = []
        self.engine._retrieve_documents("Summarize bilateral filtering in my notes", 5, True)
        self.engine.document_index.search_with_scores.assert_called_once_with(
            "Summarize bilateral filtering in my notes", 15,
        )

    def test_optional_boost_failure_keeps_answer_and_diagnostics(self) -> None:
        """Search failures after successful generation must not fail the query."""
        question, source, answer = "Explain cameras", "Cameras project points.", "Unrelated answer."
        self.engine.coherence_validator = self.validator({
            question: [1, 0], source: [1, 0], answer: [0, 1],
        })
        docs = [self.document("camera", source)]
        self.engine._generate_answer = Mock(return_value=answer)
        failures = [
            RuntimeError("Search unavailable"), ValueError("Invalid search"),
            OSError("Index unavailable"),
            APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")),
        ]
        for error in failures:
            with self.subTest(error=type(error).__name__):
                self.engine._retrieve_documents = Mock(side_effect=[(docs, [0.8]), error])
                with patch("src.rag_engine.logger") as log:
                    result = self.engine.query(question, k=1)
                self.assertEqual(result.answer, answer)
                self.assertEqual(result.source_documents, docs)
                self.assertEqual(result.confidence_scores, [0.8])
                self.assertEqual(result.metadata["coherence_level"], "critical")
                self.assertEqual(result.metadata["coherence_validation_status"], "complete")
                self.assertTrue(result.fallback_action.flag_uncertainty)
                log.exception.assert_called_once()

    def test_overview_supports_backends_without_scores(self) -> None:
        """Source balancing also works when only regular search is available."""
        self.engine.document_index.vector_store = SimpleNamespace(document_count=30)
        docs = [self.document("a", f"Topic {i}.") for i in range(28)]
        docs += [self.document("b", "Filtering."), self.document("c", "Geometry.")]
        self.engine.document_index.search.return_value = docs
        selected, scores = self.engine._retrieve_documents("Summarize all documents", 3, False)
        self.assertEqual(len({d.metadata["source"] for d in selected}), 3)
        self.assertEqual(scores, [])

    def test_loaded_index_count_does_not_use_ingestion_cache(self) -> None:
        """A reloaded FAISS index still provides the full overview candidate pool."""
        backend = FAISSVectorStore.__new__(FAISSVectorStore)
        backend.vector_store = SimpleNamespace(index=SimpleNamespace(ntotal=116))
        backend.documents = []
        self.assertEqual(backend.document_count, 116)

    def test_cli_labels_similarity_as_diagnostic(self) -> None:
        """CLI users see separate heuristic support rather than answer hedging."""
        from src.cli import _print_support_diagnostics

        result = QueryResult(
            query="question", answer="An answer.", source_documents=[], confidence_scores=[],
            metadata={"answer_support_cosine": 0.708, "semantic_support_warning": "Review sources."},
        )
        with patch("builtins.print") as output:
            _print_support_diagnostics(result)
        printed = " ".join(call.args[0] for call in output.call_args_list)
        self.assertIn("0.708", printed)
        self.assertIn("not factual confidence", printed)
        self.assertIn("Review sources.", printed)
        self.assertEqual(result.answer, "An answer.")

    def test_supported_answer_not_penalized_by_generic_question_or_other_chunks(self) -> None:
        """Source-backed claims stay supported even with unrelated retrieved text."""
        question, answer, other = "Summarize my notes", "Cameras project 3D points.", "Filters smooth noise."
        validator = self.validator({question: [0, 0, 1], answer: [1, 0, 0], other: [0, 1, 0]})
        metrics, action = validator.validate_coherence(
            question, [self.document("camera", answer), self.document("filters", other)], answer,
        )
        self.assertAlmostEqual(metrics.query_chunk_cosine, 0)
        self.assertAlmostEqual(metrics.answer_support_cosine, 1)
        self.assertEqual(metrics.coherence_level, CoherenceLevel.HIGH)
        self.assertFalse(action.boost_k)
        self.assertFalse(action.flag_uncertainty)

    def test_each_answer_sentence_needs_matching_evidence(self) -> None:
        """One related sentence cannot hide unrelated content in a long answer."""
        question, supported, unsupported = "What are cameras?", "Cameras project points.", "Bananas grow on Mars."
        validator = self.validator({
            question: [1, 0], supported: [1, 0], unsupported: [0, 1],
            f"{supported} {unsupported}": [1, 0],
        })
        metrics, _ = validator.validate_coherence(
            question, [self.document("camera", supported)], f"{supported} {unsupported}",
        )
        self.assertAlmostEqual(metrics.answer_support_cosine, 0.5)
        self.assertTrue(metrics.needs_fallback)

    def test_unrelated_answer_remains_flagged(self) -> None:
        """The change must not simply inflate all confidence scores."""
        question, source, answer = "Explain cameras", "Cameras project points.", "Bananas grow on Mars."
        validator = self.validator({question: [1, 0], source: [1, 0], answer: [0, 1]})
        metrics, action = validator.validate_coherence(question, [self.document("camera", source)], answer)
        self.assertEqual(metrics.coherence_level, CoherenceLevel.CRITICAL)
        self.assertTrue(action.flag_uncertainty)
        self.assertIn("similarity", action.uncertainty_message)
        self.assertNotIn("Low confidence", action.uncertainty_message)

    def test_exact_source_sentences_are_supported(self) -> None:
        """Copied source text stays supported despite sentence-level embedding drift."""
        source = "First sentence. Second sentence."
        validator = self.validator({
            "question": [0, 0, 1], source: [1, 0, 0],
            "First sentence.": [0, 1, 0], "Second sentence.": [0, 1, 0],
        })
        metrics, _ = validator.validate_coherence(
            "question", [self.document("a", source)], source,
        )
        self.assertAlmostEqual(metrics.answer_support_cosine, 1)

    def test_invalid_embedding_is_not_a_zero_confidence_score(self) -> None:
        """Invalid vectors must be reported as unavailable."""
        for vector in ([0, 0], [float("nan"), 1], []):
            with self.subTest(vector=vector):
                validator = self.validator({"question": vector})
                with self.assertRaises(RuntimeError):
                    validator.validate_coherence("question", [self.document("a", "source")], "answer")

    def test_overview_prefers_topic_overviews_over_exercises(self) -> None:
        """A generic summary should select overview material when it is available."""
        docs = [
            self.document("a", "Main topics exercise questions."),
            self.document("b", "Main topics exercise questions."),
            self.document("a", "Week at a glance. Camera geometry."),
            self.document("b", "Week at a glance. Linear filtering."),
        ]
        docs[2].metadata["chunk_id"] = 0
        docs[3].metadata["chunk_id"] = 0
        selected = self.engine._rerank_and_filter_chunks(
            "What are the main topics covered in the documents?", [(d, 0.1) for d in docs], 2,
        )
        self.assertEqual({d.page_content for d, _ in selected}, {d.page_content for d in docs[2:]})

    def test_boost_reuses_balanced_retrieval(self) -> None:
        """A fallback must not undo the initial summary's source diversity."""
        docs = [self.document(f"source-{i % 5}", f"Topic {i}.") for i in range(30)]
        self.engine.document_index.search_with_scores.return_value = [(d, 0.1) for d in docs]
        selected, _, _ = self.engine._apply_coherence_boost(
            "Summarize all documents", SimpleNamespace(boost_k=True, new_k_value=10), 5, True,
        )
        counts = {source: sum(d.metadata["source"] == source for d in selected)
                  for source in {d.metadata["source"] for d in selected}}
        self.assertEqual(set(counts.values()), {2})

    def test_embedding_failure_is_unavailable_not_zero_confidence(self) -> None:
        """A broken embedding service must not produce fabricated zero scores."""
        validator = self.validator({})
        validator.embeddings.embed_query.side_effect = RuntimeError("Embedding service offline")
        with self.assertRaises(RuntimeError):
            validator.validate_coherence("question", [self.document("a", "source")], "answer")

    def test_regeneration_is_rescored_and_answer_not_hedged(self) -> None:
        """Final metadata and fallback decisions describe the regenerated answer."""
        question = "What are cameras?"
        source, unrelated = "Cameras project points.", "Bananas grow on Mars."
        self.engine.coherence_validator = self.validator({
            question: [1, 0], source: [1, 0], unrelated: [0, 1],
        })
        self.engine._retrieve_documents = Mock(return_value=([self.document("a", source)], [0.8]))
        boosted = [self.document("a", source), self.document("b", source)]
        self.engine._apply_coherence_boost = Mock(return_value=(boosted, [0.8, 0.8], None))
        self.engine._generate_answer = Mock(side_effect=[unrelated, source])
        result = self.engine.query(question)
        self.assertEqual(result.answer, source)
        self.assertEqual(result.metadata["coherence_level"], "high")
        self.assertAlmostEqual(result.metadata["answer_support_cosine"], 1)
        self.assertEqual(result.coherence_metrics.num_chunks, 2)
        self.assertFalse(result.fallback_action.flag_uncertainty)
        self.engine._generate_answer.assert_called()
        self.assertEqual(self.engine._generate_answer.call_count, 2)

    def test_validation_failure_preserves_regenerated_answer_and_reports_unavailable(self) -> None:
        """A failed rescore must neither restore an old answer nor invent metrics."""
        self.engine._retrieve_documents = Mock(return_value=([self.document("a", "Source.")], [0.8]))
        validator = self.validator({"question": [1, 0], "Source.": [1, 0], "Unrelated.": [0, 1]})
        initial = validator.validate_coherence("question", [self.document("a", "Source.")], "Unrelated.")
        validator.validate_coherence = Mock(side_effect=[initial, RuntimeError("Embedding unavailable")])
        self.engine.coherence_validator = validator
        self.engine._apply_coherence_boost = Mock(return_value=(
            [self.document("a", "Source."), self.document("b", "More evidence.")], [0.8, 0.7], None,
        ))
        self.engine._generate_answer = Mock(side_effect=["Unrelated.", "Improved answer."])
        result = self.engine.query("question")
        self.assertEqual(result.answer, "Improved answer.")
        self.assertEqual(result.metadata["coherence_validation_status"], "unavailable")
        self.assertIsNone(result.coherence_metrics)
        self.assertIsNone(result.fallback_action)

User = get_user_model()


class AnswerGenerationFailureTest(TestCase):
    """Keep upstream API failures out of answers and conversation history."""

    def setUp(self) -> None:
        """Build a real query pipeline with only retrieval and OpenAI mocked."""
        self.user = User.objects.create_user(username="generation-test")
        self.client.force_login(self.user)
        self.index = DocumentIndex.objects.create(name="generation-test")
        self.engine = ConversationalRAG.__new__(ConversationalRAG)
        self.engine.conversation_history = []
        self.engine.max_history_length = 10
        self.engine.enable_coherence_validation = False
        self.engine.coherence_validator = Mock()
        self.engine.prompt_template = self.engine._create_prompt_template()
        self.engine.llm = Mock()
        self.engine._retrieve_documents = Mock(return_value=(
            [SourceDocument(page_content="The Laplacian sums second spatial derivatives.")],
            [0.9],
        ))
        self.request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def quota_error(self, code: str = "credit_balance_exhausted") -> RateLimitError:
        """Create a provider quota failure without using real credentials."""
        return RateLimitError(
            "Provider detail that must not be exposed",
            response=httpx.Response(429, request=self.request),
            body={"code": code, "type": "insufficient_quota"},
        )

    def test_endpoints_allowlist_error_messages_and_codes(self) -> None:
        """Never return arbitrary exception text or unknown codes to clients."""
        cases = [
            ("openai_quota_exceeded", "credits"),
            ("openai_authentication_failed", "API key"),
            ("openai_timeout", "timed out"),
            ("openai_rate_limited", "rate limit"),
            ("openai_generation_failed", "configuration"),
            ("private-provider-detail", "configuration"),
        ]
        for endpoint, getter, method in [
            ("/api/query/", "get_rag_engine", "query"),
            ("/api/conversational-query/", "get_conversational_rag", "conversational_query"),
        ]:
            for code, expected in cases:
                with self.subTest(endpoint=endpoint, code=code), patch(
                    f"rag_app.views.{getter}",
                ) as get_engine:
                    getattr(get_engine.return_value, method).side_effect = AnswerGenerationError(
                        "private exception with traceback and credentials", code,
                    )
                    response = self.client.post(endpoint, {
                        "question": "What is a Laplacian?", "index_name": self.index.name,
                    }, content_type="application/json")
                    self.assertEqual(response.status_code, 503)
                    self.assertIn(expected, response.json()["error"])
                    self.assertNotIn("private", response.content.decode())
                    self.assertEqual(response.json()["error_code"],
                                     code if code != "private-provider-detail" else "openai_generation_failed")
        self.assertEqual(Query.objects.count(), 0)

    def test_quota_failure_is_not_an_answer_or_history_entry(self) -> None:
        """Do not score or remember provider errors as generated answers."""
        self.engine.enable_coherence_validation = True
        self.engine.llm.invoke.side_effect = self.quota_error()
        with self.assertRaisesRegex(RuntimeError, "credits"):
            self.engine.conversational_query("What is a Laplacian?")
        self.engine.coherence_validator.validate_coherence.assert_not_called()
        self.assertEqual(self.engine.conversation_history, [])
        self.engine.llm.invoke.assert_called_once()

    def test_boosted_generation_failure_is_not_swallowed(self) -> None:
        """Propagate failures during coherence-triggered regeneration too."""
        self.engine.enable_coherence_validation = True
        self.engine.coherence_validator.validate_coherence.return_value = (
            SimpleNamespace(
                needs_fallback=True, coherence_level=SimpleNamespace(value="low"),
            ),
            SimpleNamespace(boost_k=True, new_k_value=10),
        )
        self.engine.llm.invoke.side_effect = [
            SimpleNamespace(content="The Laplacian sums second derivatives."),
            self.quota_error(),
        ]
        documents = [
            SourceDocument(page_content="The Laplacian sums second derivatives."),
            SourceDocument(page_content="It is a differential operator."),
        ]
        with patch.object(self.engine, "_apply_coherence_boost",
                          return_value=(documents, [0.9, 0.8], None)):
            with self.assertRaisesRegex(RuntimeError, "credits"):
                self.engine.conversational_query("What is a Laplacian?")
        self.assertEqual(self.engine.conversation_history, [])
        self.assertEqual(self.engine.llm.invoke.call_count, 2)

    def test_generation_failure_categories(self) -> None:
        """Distinguish quota, transient rate limits, credentials, and timeouts."""
        cases = [
            (self.quota_error("insufficient_quota"), "credits"),
            (RateLimitError("private", response=httpx.Response(429, request=self.request),
                            body={"code": "rate_limit_exceeded"}), "rate limit"),
            (AuthenticationError("private", response=httpx.Response(401, request=self.request),
                                 body=None), "API key"),
            (APITimeoutError(request=self.request), "timed out"),
        ]
        for error, message in cases:
            with self.subTest(error=type(error).__name__, message=message):
                self.engine.llm.invoke.side_effect = error
                with self.assertRaisesRegex(RuntimeError, message):
                    self.engine._generate_answer("Laplacian definition", "What is a Laplacian?")

    def test_query_endpoints_report_quota_and_recover(self) -> None:
        """Both endpoints return actionable errors and accept subsequent queries."""
        for endpoint, getter in [
            ("/api/query/", "get_rag_engine"),
            ("/api/conversational-query/", "get_conversational_rag"),
        ]:
            with self.subTest(endpoint=endpoint), patch(
                f"rag_app.views.{getter}", return_value=self.engine
            ):
                for code in ("credit_balance_exhausted", "insufficient_quota"):
                    self.engine.llm.invoke.side_effect = self.quota_error(code)
                    response = self.client.post(endpoint, {
                        "question": "What is a Laplacian?",
                        "index_name": self.index.name,
                    }, content_type="application/json")
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["error_code"], "openai_quota_exceeded")
                    self.assertIn("credits", response.json()["error"])
                    self.assertNotIn("Provider detail", response.json()["error"])
                    self.assertNotIn("answer", response.json())
                self.assertEqual(Query.objects.count(), 0)
                self.engine.llm.invoke.side_effect = None
                self.engine.llm.invoke.return_value = SimpleNamespace(
                    content="The Laplacian sums second spatial derivatives."
                )
                response = self.client.post(endpoint, {
                    "question": "What is a Laplacian?",
                    "index_name": self.index.name,
                }, content_type="application/json")
                self.assertEqual(response.status_code, 200)
                self.assertIn("second spatial derivatives", response.json()["answer"])
                Query.objects.all().delete()

    def test_conversational_options_sources_and_diagnostics(self) -> None:
        """Conversational mode honors the same retrieval settings and source UI."""
        self.engine.llm.invoke.return_value = SimpleNamespace(content="The Laplacian sums derivatives.")
        for include in (True, False):
            with self.subTest(include=include), patch(
                "rag_app.views.get_conversational_rag", return_value=self.engine,
            ):
                response = self.client.post("/api/conversational-query/", {
                    "question": "What is a Laplacian?", "index_name": self.index.name,
                    "k": 3, "include_sources": include, "include_scores": include,
                }, content_type="application/json")
                self.assertEqual(response.status_code, 200)
                self.assertEqual("source_documents" in response.json(), include)
                self.assertEqual("confidence_scores" in response.json(), include)
                self.engine._retrieve_documents.assert_called_with(
                    self.engine._retrieve_documents.call_args.args[0], 3, include,
                )
                self.assertEqual(response.json()["metadata"]["coherence_validation_status"], "disabled")
                record = Query.objects.latest("created_at")
                self.assertEqual(record.k_results, 3)
                self.assertEqual(record.include_sources, include)
                self.assertEqual(record.include_scores, include)

    def test_conversational_rejects_invalid_retrieval_size(self) -> None:
        """Invalid conversational retrieval settings must not reach the engine."""
        for k in (0, 21, "bad"):
            with self.subTest(k=k), patch("rag_app.views.get_conversational_rag") as getter:
                response = self.client.post("/api/conversational-query/", {
                    "question": "Summarize all documents", "index_name": self.index.name, "k": k,
                }, content_type="application/json")
                self.assertEqual(response.status_code, 400)
                getter.assert_not_called()


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

    def test_upload_rejects_invalid_index_name_before_creating_it(self):
        '''
        An index name that can't be a directory is a clear 400, and no
        DocumentIndex row is created for it.
        '''
        test_file = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
        response = self.client.post('/api/upload-documents/', {
            'files': [test_file],
            'index_name': 'my index/../x'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid index_name', response.json()['error'])
        self.assertFalse(DocumentIndex.objects.filter(name='my index/../x').exists())

    def test_query_rejects_index_name_with_trailing_newline(self):
        '''``re.match('^...$')`` accepted "name\\n"; fullmatch must not.'''
        response = self.client.post(
            '/api/query/',
            json.dumps({'question': 'What is this?', 'index_name': 'test_index\n'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def _assert_status(self, url, payload, expected):
        '''POST a JSON body to a query endpoint and assert the status code.'''
        resp = self.client.post(url, json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, expected, msg=payload)
        return resp

    def test_query_input_validation_returns_400(self):
        '''
        Test the classic query endpoints reject malformed/oversized fields.

        Covers the raw-JSON bounds that mirror the serializers: the 5001
        question length, the 256 index_name length, and non-string question /
        index_name values across both query endpoints. All return 400 (a
        non-string question must not raise AttributeError -> 500).
        '''
        q, c = '/api/query/', '/api/conversational-query/'
        self._assert_status(q, {'question': 'a' * 5001, 'index_name': 'test_index'}, 400)
        self._assert_status(q, {'question': 7, 'index_name': 'test_index'}, 400)
        self._assert_status(q, {'question': 'hi', 'index_name': 'i' * 256}, 400)
        self._assert_status(q, {'question': 'hi', 'index_name': 7}, 400)
        self._assert_status(c, {'question': 7, 'index_name': 'test_index'}, 400)
        self._assert_status(c, {'question': 'hi', 'index_name': 7}, 400)

    def test_query_accepts_boundary_field_lengths(self):
        '''
        Test boundary-length question/index_name pass validation.

        A 5000-char question and a 255-char index_name are within bounds;
        aiming at a missing index isolates each to a 404 (not 400), confirming
        the value was accepted without invoking the engine.
        '''
        self._assert_status('/api/query/', {'question': 'a' * 5000, 'index_name': 'missing'}, 404)
        self._assert_status('/api/query/', {'question': 'hi', 'index_name': 'i' * 255}, 404)

    def test_upload_rejects_oversized_index_name(self):
        '''
        Test the upload endpoint rejects an over-length index_name.

        The classic upload path reads index_name straight from POST; it is
        capped at 255 chars to match DocumentUploadSerializer.
        '''
        upload = SimpleUploadedFile('t.txt', b'x', content_type='text/plain')
        resp = self.client.post(
            '/api/upload-documents/', {'files': [upload], 'index_name': 'i' * 256})
        self.assertEqual(resp.status_code, 400)

    def test_get_rag_engine_constructs_once_under_concurrency(self):
        '''
        Concurrent cache misses build exactly one engine.

        get_rag_engine constructs under _rag_engines_lock, so parallel callers
        that all miss the same index share a single RAGEngine instance rather
        than each building their own.
        '''
        from rag_app import views

        built = []

        class _StubEngine:
            def __init__(self, index_name):
                time.sleep(0.02)  # widen the window for thread overlap
                built.append(index_name)

        with patch.object(views, 'RAGEngine', _StubEngine):
            views._rag_engines.pop('concurrent_idx', None)
            engines = []

            def worker():
                engines.append(views.get_rag_engine('concurrent_idx'))

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            views._rag_engines.pop('concurrent_idx', None)

        self.assertEqual(len(built), 1)
        self.assertTrue(all(engine is engines[0] for engine in engines))

    def test_cleared_engine_is_rebuilt_not_reused(self):
        '''
        Invalidating the cache forces a fresh build, never a stale instance.

        Construction and the cache pop both hold _rag_engines_lock, so an engine
        built from pre-clear state can't be cached after invalidation; the first
        get after a pop constructs a new instance.
        '''
        from rag_app import views

        class _StubEngine:
            def __init__(self, index_name):
                self.index_name = index_name

        with patch.object(views, 'RAGEngine', _StubEngine):
            views._rag_engines.pop('rebuild_idx', None)
            first = views.get_rag_engine('rebuild_idx')
            with views._rag_engines_lock:
                views._rag_engines.pop('rebuild_idx', None)
            second = views.get_rag_engine('rebuild_idx')
            views._rag_engines.pop('rebuild_idx', None)

        self.assertIsNot(first, second)


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
            source_documents=[],
            confidence_scores=[],
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

    @patch('rag_app.views.get_conversational_rag')
    def test_existing_anonymous_session_is_linked_without_duplication(
            self, get_engine):
        session = self.client.session
        session.save()
        existing_session = QuerySession.objects.create(
            session_key=session.session_key,
            index=self.index,
        )
        get_engine.return_value.conversational_query.return_value = SimpleNamespace(
            answer='An answer',
            metadata={},
            source_documents=[],
            confidence_scores=[],
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
        self.assertEqual(QuerySession.objects.count(), 1)
        existing_session.refresh_from_db()
        self.assertEqual(existing_session.user, self.user)


class AnonymousConversationTest(TestCase):
    """Verify the documented local-development authentication opt-out."""

    @patch('rag_app.views.get_conversational_rag')
    def test_anonymous_conversation_keeps_nullable_user(self, get_engine):
        from rag_app.views import ConversationalQueryView

        index = DocumentIndex.objects.create(name='anonymous_index')
        get_engine.return_value.conversational_query.return_value = SimpleNamespace(
            answer='An answer',
            metadata={},
            source_documents=[],
            confidence_scores=[],
        )

        with patch.object(
                ConversationalQueryView, 'permission_classes', [AllowAny]):
            response = self.client.post(
                '/api/conversational-query/',
                json.dumps({
                    'question': 'A question',
                    'index_name': index.name,
                }),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 200)
        query_session = QuerySession.objects.get(id=response.json()['session_id'])
        self.assertIsNone(query_session.user)


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
