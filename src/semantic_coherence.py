"""
Semantic coherence validation for RAG pipeline.

Tracks coherence across query→chunk, chunk→generation, and query→generation
embeddings for retrieval diagnostics and sentence-to-passage semantic support.
These heuristic similarities are not calibrated factual-confidence estimates.

Written by DJ Leamen (2025-2026)
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger
from openai import OpenAIError

from src.config import settings
from src.vector_store import LocalEmbeddings


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Calculate cosine similarity between two vectors using numpy.
    
    :param a: First vector
    :param b: Second vector
    :return: Cosine similarity score (-1 to 1)
    '''
    # Flatten arrays to ensure they're 1D
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Calculate cosine similarity
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

class CoherenceLevel(Enum):
    '''
    Coherence validation levels.
    
    Defines severity levels for coherence drops in the RAG pipeline.
    '''
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

@dataclass
class CoherenceMetrics:
    '''
    Container for coherence validation metrics.
    
    Stores similarity scores, deltas, and coherence assessment
    across all RAG pipeline stages.
    '''
    query_chunk_cosine: float
    chunk_generation_cosine: float
    query_generation_cosine: float
    coherence_delta: float
    coherence_level: CoherenceLevel
    needs_fallback: bool

    # Individual deltas for detailed analysis
    query_chunk_delta: float
    chunk_generation_delta: float
    query_generation_delta: float

    # Metadata
    num_chunks: int
    avg_chunk_similarity: float
    min_chunk_similarity: float
    max_chunk_similarity: float
    answer_support_cosine: float = 0.0


@dataclass
class FallbackAction:
    '''
    Defines fallback actions when coherence drops.
    
    Specifies retrieval boosting and diagnostic warnings. The legacy
    hedge_output field is retained for compatibility; answers are not rewritten.
    '''
    boost_k: bool = False
    hedge_output: bool = False
    flag_uncertainty: bool = False
    new_k_value: Optional[int] = None
    uncertainty_message: Optional[str] = None
    confidence_threshold: float = 0.5


class SemanticCoherenceValidator:
    '''
    Validates semantic coherence across RAG pipeline stages.
    
    Measures embedding similarity between query, retrieved chunks,
    and generated answer to detect information loss.
    '''

    def __init__(
        self,
        coherence_thresholds: Optional[Dict[str, float]] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Embeddings] = None
    ):
        '''
        Initialize the semantic coherence validator.

        :param coherence_thresholds: Custom threshold values for coherence levels
        :param fallback_config: Custom configuration for fallback actions
        :param embeddings: Optional embedding backend to reuse. The RAG engine
            passes its vector store's embeddings so a second copy of the local
            model isn't loaded per cached engine.
        '''
        # Share the retriever's backend and avoid loading another local model.
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
            if use_local or not settings.openai_api_key:
                logger.info("SemanticCoherenceValidator using local embeddings (sentence-transformers)")
                self.embeddings = LocalEmbeddings("all-MiniLM-L6-v2")
            else:
                self.embeddings = OpenAIEmbeddings(
                    model=settings.embedding_model,
                    timeout=settings.request_timeout
                )
        # Default coherence thresholds from config
        self.thresholds = coherence_thresholds or {
            "high_threshold": settings.coherence_high_threshold,
            "medium_threshold": settings.coherence_medium_threshold,
            "low_threshold": settings.coherence_low_threshold,
            "critical_threshold": settings.coherence_critical_threshold
        }

        # Default fallback configuration from config
        self.fallback_config = fallback_config or {
            "boost_k_on_low": True,
            "boost_k_multiplier": settings.boost_k_multiplier,
            "max_k_boost": settings.max_k_boost
        }

        logger.info("SemanticCoherenceValidator initialized")

    def validate_coherence(
        self,
        query: str,
        retrieved_chunks: List[Document],
        generated_answer: str
    ) -> Tuple[CoherenceMetrics, FallbackAction]:
        '''
        Validate semantic coherence across the RAG pipeline.
        
        :param query: User's query text
        :param retrieved_chunks: List of retrieved document chunks
        :param generated_answer: Generated answer text
        :return: Tuple of (CoherenceMetrics, FallbackAction)
        '''
        logger.debug(f"Validating coherence for query: {query[:50]}...")
        if not query.strip() or not generated_answer.strip() or not retrieved_chunks:
            raise ValueError("Semantic support requires a question, answer, and source chunks")

        # Get embeddings
        query_embedding = self._get_embedding(query)
        chunk_embeddings = self._get_chunk_embeddings(retrieved_chunks)
        answer_embedding = self._get_embedding(generated_answer)

        # Calculate pairwise cosine similarities
        query_chunk_similarities = self._calculate_similarities(
            query_embedding, chunk_embeddings
        )
        chunk_answer_similarities = self._calculate_chunk_answer_similarities(
            chunk_embeddings, answer_embedding
        )
        query_answer_similarity = self._calculate_similarities(
            query_embedding, [answer_embedding]
        )[0]

        # Calculate coherence metrics
        metrics = self._calculate_coherence_metrics(
            query_chunk_similarities,
            chunk_answer_similarities,
            query_answer_similarity,
            len(retrieved_chunks),
            self._calculate_answer_support(retrieved_chunks, generated_answer),
        )

        # Determine fallback actions
        fallback_action = self._determine_fallback_action(metrics)

        logger.info(f"Coherence validation complete. Level: {metrics.coherence_level.value}")

        return metrics, fallback_action

    @staticmethod
    def _validated_vector(embedding: List[float]) -> np.ndarray:
        """Reject failed or invalid embeddings instead of inventing low support."""
        vector = np.asarray(embedding, dtype=float).reshape(1, -1)
        if not vector.size or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Semantic support received an invalid embedding")
        return vector

    def _get_embedding(self, text: str) -> np.ndarray:
        '''
        Get embedding vector for text.

        :param text: Text to embed
        :return: Numpy array containing embedding vector
        '''
        try:
            embedding = self.embeddings.embed_query(text)
            return self._validated_vector(embedding)
        except (OpenAIError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Semantic embedding failed: {}", type(e).__name__)
            raise RuntimeError("Semantic support embedding unavailable") from e

    def _get_chunk_embeddings(self, chunks: List[Document]) -> List[np.ndarray]:
        '''
        Get embeddings for all chunks.
        
        :param chunks: List of document chunks
        :return: List of embedding vectors for each chunk
        '''
        return self._embed_texts([chunk.page_content for chunk in chunks])

    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Batch and validate diagnostic embeddings, surfacing backend failures."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            if len(embeddings) != len(texts):
                raise ValueError("Embedding response length does not match the input")
            return [self._validated_vector(vector) for vector in embeddings]
        except (OpenAIError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Semantic batch embedding failed: {}", type(e).__name__)
            raise RuntimeError("Semantic support embedding unavailable") from e

    @staticmethod
    def _text_windows(text: str) -> List[str]:
        """Keep passages short enough for local models, with overlapping context."""
        words = text.split()
        if len(words) <= 60:
            return [" ".join(words)]
        return [" ".join(words[start:start + 60]) for start in range(0, len(words), 30)]

    def _statement_windows(self, text: str) -> List[str]:
        """Use identical sentence boundaries for source and answer comparisons."""
        normalized = " ".join(text.split())
        return [
            window
            for sentence in re.split(r'(?<=[.!?])\s+', normalized)
            if sentence.strip()
            for window in self._text_windows(sentence)
        ]

    def _calculate_answer_support(self, chunks: List[Document], answer: str) -> float:
        """Match each answer sentence to its best source passage, not every chunk."""
        passages = list(dict.fromkeys(
            passage
            for chunk in chunks
            for passage in [
                " ".join(chunk.page_content.split()),
                *self._text_windows(chunk.page_content),
                *self._statement_windows(chunk.page_content),
            ]
        ))
        statements = self._statement_windows(answer)
        vectors = self._embed_texts(passages + statements)
        passage_vectors = vectors[:len(passages)]
        support = [
            max(self._calculate_similarities(statement, passage_vectors))
            for statement in vectors[len(passages):]
        ]
        return float(np.mean(support))

    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray]
    ) -> List[float]:
        '''
        Calculate cosine similarities between query and chunks.
        
        :param query_embedding: Query embedding vector
        :param chunk_embeddings: List of chunk embedding vectors
        :return: List of cosine similarity scores (-1 to 1)
        '''
        return [
            float(cosine_similarity_numpy(query_embedding, chunk_embedding))
            for chunk_embedding in chunk_embeddings
        ]

    def _calculate_chunk_answer_similarities(
        self,
        chunk_embeddings: List[np.ndarray],
        answer_embedding: np.ndarray
    ) -> List[float]:
        '''
        Calculate cosine similarities between chunks and generated answer.
        
        :param chunk_embeddings: List of chunk embedding vectors
        :param answer_embedding: Generated answer embedding vector
        :return: List of cosine similarity scores between chunks and answer
        '''
        return self._calculate_similarities(answer_embedding, chunk_embeddings)

    def _calculate_coherence_metrics(
        self,
        query_chunk_similarities: List[float],
        chunk_answer_similarities: List[float],
        query_answer_similarity: float,
        num_chunks: int,
        answer_support_cosine: float,
    ) -> CoherenceMetrics:
        '''
        Calculate comprehensive coherence metrics.
        
        :param query_chunk_similarities: Similarity scores between query and chunks
        :param chunk_answer_similarities: Similarity scores between chunks and answer
        :param query_answer_similarity: Similarity score between query and answer
        :param num_chunks: Number of retrieved chunks
        :param answer_support_cosine: Mean best-passage similarity per answer sentence
        :return: CoherenceMetrics object with all calculated metrics
        '''

        # Average similarities
        avg_query_chunk = np.mean(query_chunk_similarities) if query_chunk_similarities else 0.0
        avg_chunk_answer = np.mean(chunk_answer_similarities) if chunk_answer_similarities else 0.0

        # Calculate deltas (gaps between pipeline stages)
        query_chunk_delta = 1.0 - avg_query_chunk
        chunk_generation_delta = abs(avg_query_chunk - avg_chunk_answer)
        query_generation_delta = abs(avg_query_chunk - query_answer_similarity)

        # Query similarity is a retrieval diagnostic, not a measure of answer support.
        coherence_delta = 1.0 - answer_support_cosine

        # Determine coherence level
        coherence_level = self._determine_coherence_level(float(coherence_delta))

        # Determine if fallback is needed
        needs_fallback = coherence_level in [CoherenceLevel.LOW, CoherenceLevel.CRITICAL]

        return CoherenceMetrics(
            query_chunk_cosine=float(avg_query_chunk),
            chunk_generation_cosine=float(avg_chunk_answer),
            query_generation_cosine=query_answer_similarity,
            coherence_delta=float(coherence_delta),
            coherence_level=coherence_level,
            needs_fallback=needs_fallback,
            query_chunk_delta=float(query_chunk_delta),
            chunk_generation_delta=float(chunk_generation_delta),
            query_generation_delta=float(query_generation_delta),
            num_chunks=num_chunks,
            avg_chunk_similarity=float(avg_query_chunk),
            min_chunk_similarity=float(min(query_chunk_similarities)) if query_chunk_similarities else 0.0,
            max_chunk_similarity=float(max(query_chunk_similarities)) if query_chunk_similarities else 0.0,
            answer_support_cosine=answer_support_cosine,
        )

    def _determine_coherence_level(self, coherence_delta: float) -> CoherenceLevel:
        '''
        Determine coherence level based on delta.
        
        :param coherence_delta: Calculated coherence gap (0-1)
        :return: CoherenceLevel enum value (HIGH, MEDIUM, LOW, or CRITICAL)
        '''

        # Invert the logic since delta represents gap (higher = worse)
        if coherence_delta <= (1 - self.thresholds["high_threshold"]):
            return CoherenceLevel.HIGH
        elif coherence_delta <= (1 - self.thresholds["medium_threshold"]):
            return CoherenceLevel.MEDIUM
        elif coherence_delta <= (1 - self.thresholds["low_threshold"]):
            return CoherenceLevel.LOW
        else:
            return CoherenceLevel.CRITICAL

    def _determine_fallback_action(self, metrics: CoherenceMetrics) -> FallbackAction:
        '''
        Determine appropriate fallback actions based on coherence metrics.
        
        :param metrics: CoherenceMetrics object with validation results
        :return: FallbackAction object specifying remediation strategies
        '''

        action = FallbackAction()

        if metrics.coherence_level == CoherenceLevel.CRITICAL:
            # Critical coherence - apply all fallbacks
            action.boost_k = self.fallback_config.get("boost_k_on_low", True)
            action.flag_uncertainty = True
            action.uncertainty_message = (
                "Some answer passages have limited similarity to the retrieved sources. "
                "Review the source chunks. This heuristic does not determine factual correctness."
            )

            if action.boost_k:
                current_k = metrics.num_chunks
                max_k = self.fallback_config.get("max_k_boost", 20)
                multiplier = self.fallback_config.get("boost_k_multiplier", 2.0)
                action.new_k_value = min(int(current_k * multiplier), max_k)

        elif metrics.coherence_level == CoherenceLevel.LOW:
            # Low semantic support warrants another retrieval attempt.
            action.boost_k = self.fallback_config.get("boost_k_on_low", True)
            action.flag_uncertainty = True
            action.uncertainty_message = (
                "Some answer passages have limited similarity to the retrieved sources. "
                "Review the source chunks. This heuristic does not determine factual correctness."
            )

            if action.boost_k:
                current_k = metrics.num_chunks
                max_k = self.fallback_config.get("max_k_boost", 20)
                multiplier = self.fallback_config.get("boost_k_multiplier", 1.5)
                action.new_k_value = min(int(current_k * multiplier), max_k)

        # High coherence requires no fallbacks

        return action

    def get_hedged_response(self, original_answer: str, metrics: CoherenceMetrics) -> str:
        """Preserve answers; similarity diagnostics no longer rewrite their certainty."""
        return original_answer
