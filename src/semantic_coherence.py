"""
Semantic coherence validation for RAG pipeline.

Tracks coherence across query→chunk, chunk→generation, and query→generation
embeddings to identify and mitigate information loss during retrieval and generation.

Written by DJ Leamen (2025-2026)
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from loguru import logger

from src.config import settings


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Calculate cosine similarity between two vectors using numpy.
    
    :param a: First vector
    :param b: Second vector
    :return: Cosine similarity score (0 to 1)
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


@dataclass
class FallbackAction:
    '''
    Defines fallback actions when coherence drops.
    
    Specifies remediation strategies including k-boosting,
    hedging, and uncertainty flagging.
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
        fallback_config: Optional[Dict[str, Any]] = None
    ):
        '''
        Initialize the semantic coherence validator.
        
        :param coherence_thresholds: Custom threshold values for coherence levels
        :param fallback_config: Custom configuration for fallback actions
        '''
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
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
            "hedge_on_medium": True,
            "flag_uncertainty_on_critical": True,
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
            len(retrieved_chunks)
        )

        # Determine fallback actions
        fallback_action = self._determine_fallback_action(metrics)

        logger.info(f"Coherence validation complete. Level: {metrics.coherence_level.value}")

        return metrics, fallback_action

    def _get_embedding(self, text: str) -> np.ndarray:
        '''
        Get embedding vector for text.
        
        :param text: Text to embed
        :return: Numpy array containing embedding vector
        '''
        try:
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros((1, 1536))  # Default OpenAI embedding dimension

    def _get_chunk_embeddings(self, chunks: List[Document]) -> List[np.ndarray]:
        '''
        Get embeddings for all chunks.
        
        :param chunks: List of document chunks
        :return: List of embedding vectors for each chunk
        '''
        embeddings = []

        for chunk in chunks:
            try:
                embedding = self.embeddings.embed_query(chunk.page_content)
                embeddings.append(np.array(embedding).reshape(1, -1))
            except Exception as e:
                logger.error(f"Error getting chunk embedding: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros((1, 1536)))

        return embeddings

    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray]
    ) -> List[float]:
        '''
        Calculate cosine similarities between query and chunks.
        
        :param query_embedding: Query embedding vector
        :param chunk_embeddings: List of chunk embedding vectors
        :return: List of similarity scores (0-1)
        '''
        similarities = []

        for chunk_embedding in chunk_embeddings:
            try:
                similarity = cosine_similarity_numpy(query_embedding, chunk_embedding)
                similarities.append(float(similarity))
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                similarities.append(0.0)

        return similarities

    def _calculate_chunk_answer_similarities(
        self,
        chunk_embeddings: List[np.ndarray],
        answer_embedding: np.ndarray
    ) -> List[float]:
        '''
        Calculate cosine similarities between chunks and generated answer.
        
        :param chunk_embeddings: List of chunk embedding vectors
        :param answer_embedding: Generated answer embedding vector
        :return: List of similarity scores between chunks and answer (0-1)
        '''
        similarities = []

        for chunk_embedding in chunk_embeddings:
            try:
                similarity = cosine_similarity_numpy(chunk_embedding, answer_embedding)
                similarities.append(float(similarity))
            except Exception as e:
                logger.error(f"Error calculating chunk-answer similarity: {e}")
                similarities.append(0.0)

        return similarities

    def _calculate_coherence_metrics(
        self,
        query_chunk_similarities: List[float],
        chunk_answer_similarities: List[float],
        query_answer_similarity: float,
        num_chunks: int
    ) -> CoherenceMetrics:
        '''
        Calculate comprehensive coherence metrics.
        
        :param query_chunk_similarities: Similarity scores between query and chunks
        :param chunk_answer_similarities: Similarity scores between chunks and answer
        :param query_answer_similarity: Similarity score between query and answer
        :param num_chunks: Number of retrieved chunks
        :return: CoherenceMetrics object with all calculated metrics
        '''

        # Average similarities
        avg_query_chunk = np.mean(query_chunk_similarities) if query_chunk_similarities else 0.0
        avg_chunk_answer = np.mean(chunk_answer_similarities) if chunk_answer_similarities else 0.0

        # Calculate deltas (gaps between pipeline stages)
        query_chunk_delta = 1.0 - avg_query_chunk
        chunk_generation_delta = abs(avg_query_chunk - avg_chunk_answer)
        query_generation_delta = abs(avg_query_chunk - query_answer_similarity)

        # Overall coherence delta (maximum gap)
        coherence_delta = max(query_chunk_delta, chunk_generation_delta, query_generation_delta)

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
            max_chunk_similarity=float(max(query_chunk_similarities)) if query_chunk_similarities else 0.0
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
            action.hedge_output = True
            action.flag_uncertainty = True
            action.uncertainty_message = (
                "⚠️ Low confidence in answer due to semantic inconsistency. "
                "Consider rephrasing your question or providing more context."
            )

            if action.boost_k:
                current_k = metrics.num_chunks
                max_k = self.fallback_config.get("max_k_boost", 20)
                multiplier = self.fallback_config.get("boost_k_multiplier", 2.0)
                action.new_k_value = min(int(current_k * multiplier), max_k)

        elif metrics.coherence_level == CoherenceLevel.LOW:
            # Low coherence - boost retrieval and hedge
            action.boost_k = self.fallback_config.get("boost_k_on_low", True)
            action.hedge_output = True
            action.flag_uncertainty = False

            if action.boost_k:
                current_k = metrics.num_chunks
                max_k = self.fallback_config.get("max_k_boost", 20)
                multiplier = self.fallback_config.get("boost_k_multiplier", 1.5)
                action.new_k_value = min(int(current_k * multiplier), max_k)

        elif metrics.coherence_level == CoherenceLevel.MEDIUM:
            # Medium coherence - light hedging only
            action.hedge_output = self.fallback_config.get("hedge_on_medium", True)

        # High coherence requires no fallbacks

        return action

    def get_hedged_response(self, original_answer: str, metrics: CoherenceMetrics) -> str:
        '''
        Add hedging language to response when coherence is questionable.
        
        :param original_answer: Original generated answer text
        :param metrics: CoherenceMetrics object with validation results
        :return: Answer with appropriate hedging language based on coherence level
        '''
        if metrics.coherence_level == CoherenceLevel.HIGH:
            return original_answer

        hedging_phrases = {
            CoherenceLevel.MEDIUM: [
                "Based on the available information, ",
                "According to the documents, ",
                "From what I can determine, "
            ],
            CoherenceLevel.LOW: [
                "Based on limited information in the documents, ",
                "While the available context suggests ",
                "From the relevant sections I found, "
            ],
            CoherenceLevel.CRITICAL: [
                "⚠️ With limited confidence, ",
                "⚠️ Based on potentially incomplete information, ",
                "⚠️ The available context suggests, though with uncertainty, "
            ]
        }

        hedge = random.choice(hedging_phrases.get(metrics.coherence_level, [""]))

        hedged_answer = hedge + original_answer.lstrip()

        # Add uncertainty footer for critical cases
        if metrics.coherence_level == CoherenceLevel.CRITICAL:
            hedged_answer += (
                "\n\n*Note: This answer has low semantic coherence. "
                "Consider rephrasing your question for better results.*"
            )

        return hedged_answer
