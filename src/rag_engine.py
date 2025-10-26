"""
RAG (Retrieval-Augmented Generation) engine for document Q&A.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from loguru import logger

from src.config import settings
from src.vector_store import DocumentIndex
from src.document_processor import DocumentProcessor
from src.semantic_coherence import SemanticCoherenceValidator, CoherenceMetrics, FallbackAction


@dataclass
class QueryResult:
    '''Result of a Q&A query.'''
    query: str
    answer: str
    source_documents: List[Document]
    confidence_scores: List[float]
    metadata: Dict[str, Any]
    coherence_metrics: Optional[CoherenceMetrics] = None
    fallback_action: Optional[FallbackAction] = None


class RAGEngine:
    '''Main RAG engine for document Q&A.'''

    def __init__(self, index_name: str = "default", enable_coherence_validation: bool = True):
        self.index_name = index_name
        self.document_index = DocumentIndex(index_name)
        self.document_processor = DocumentProcessor()

        # Initialize semantic coherence validator
        self.enable_coherence_validation = enable_coherence_validation
        if enable_coherence_validation:
            self.coherence_validator = SemanticCoherenceValidator()
            logger.info("Semantic coherence validation enabled")
        else:
            self.coherence_validator = None
            logger.info("Semantic coherence validation disabled")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        # Create prompt template
        self.prompt_template = self._create_prompt_template()

        logger.info(f"RAG Engine initialized with index: {index_name}")

    def _create_prompt_template(self) -> PromptTemplate:
        '''Create the prompt template for Q&A.'''
        template = '''You are a helpful AI assistant that answers questions based on the provided context chunks from documents.
Use only the information from the context to answer questions. If the context doesn't contain enough information
to answer the question, say so clearly.

Note: The context below consists of relevant text chunks (segments) extracted from larger documents.

Context chunks:
{context}

Question: {question}

Instructions:
1. Answer based only on the provided context chunks
2. Be specific and cite relevant parts of the context
3. If the context is insufficient, state this clearly
4. Provide a comprehensive answer when possible
5. If there are multiple relevant chunks, synthesize them
6. Respond in paragraph format, avoid using markdown

Answer:'''

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def add_documents(self, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> int:
        '''Add documents to the RAG system.'''
        all_documents = []

        for file_path in file_paths:
            try:
                logger.info(f"Processing document: {file_path}")
                documents = self.document_processor.process_document(
                    file_path, metadata)
                all_documents.extend(documents)
                logger.info(f"Added {len(documents)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if all_documents:
            self.document_index.add_documents(all_documents)
            logger.info(
                f"Successfully added {len(all_documents)} total document chunks")
            return len(all_documents)
        else:
            logger.warning("No documents were successfully processed")
            return 0

    def _retrieve_documents(self, question: str, k: int, include_scores: bool):
        '''Retrieve documents with optional scores.'''
        initial_k = min(k * 3, 20)

        if include_scores and hasattr(self.document_index.vector_store, 'similarity_search_with_score'):
            doc_score_pairs = self.document_index.search_with_scores(
                question, initial_k)
            filtered_pairs = self._rerank_and_filter_chunks(
                question, doc_score_pairs, target_k=k)
            source_documents = [doc for doc, score in filtered_pairs]
            confidence_scores = [float(score) for doc, score in filtered_pairs]
        else:
            source_documents = self.document_index.search(question, initial_k)
            source_documents = self._filter_chunks_by_relevance(
                question, source_documents, target_k=k)
            confidence_scores = [1.0] * len(source_documents)

        return source_documents, confidence_scores

    def _generate_answer(self, context: str, question: str) -> str:
        '''Generate answer from context and question.'''
        prompt = self.prompt_template.format(
            context=context, question=question)
        try:
            answer = self.llm.predict(prompt)
            logger.info("Successfully generated answer")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."

    def _apply_coherence_boost(self, question: str, fallback_action, original_k: int,
                               include_scores: bool):
        '''Apply k-boost fallback and regenerate answer.'''
        if not fallback_action.boost_k or not fallback_action.new_k_value:
            return None, None, None

        if fallback_action.new_k_value <= original_k:
            return None, None, None

        logger.info(
            f"Boosting k from {original_k} to {fallback_action.new_k_value}")

        if include_scores and hasattr(self.document_index.vector_store, 'similarity_search_with_score'):
            boosted_doc_score_pairs = self.document_index.search_with_scores(
                question, fallback_action.new_k_value)
            boosted_source_documents = [
                doc for doc, score in boosted_doc_score_pairs]
            boosted_confidence_scores = [
                float(score) for doc, score in boosted_doc_score_pairs]
        else:
            boosted_source_documents = self.document_index.search(
                question, fallback_action.new_k_value)
            boosted_confidence_scores = [1.0] * len(boosted_source_documents)

        return boosted_source_documents, boosted_confidence_scores, None

    def _apply_coherence_fallbacks(self, question: str, answer: str, source_documents,
                                   confidence_scores, coherence_metrics, fallback_action,
                                   original_k: int, include_scores: bool):
        '''Apply coherence fallback actions.'''
        if not coherence_metrics.needs_fallback:
            return answer, source_documents, confidence_scores

        logger.info(
            f"Applying coherence fallbacks. Level: {coherence_metrics.coherence_level.value}")

        # 1. Boost k if needed
        boosted_docs, boosted_scores, _ = self._apply_coherence_boost(
            question, fallback_action, original_k, include_scores)

        if boosted_docs and len(boosted_docs) > len(source_documents):
            boosted_context = self._prepare_context(boosted_docs)
            try:
                boosted_answer = self._generate_answer(
                    boosted_context, question)
                logger.info(
                    "Successfully regenerated answer with boosted retrieval")
                source_documents = boosted_docs
                confidence_scores = boosted_scores
                answer = boosted_answer
            except Exception as e:
                logger.error(f"Error regenerating answer with boosted k: {e}")

        # 2. Apply hedging
        final_answer = answer
        if fallback_action.hedge_output and self.coherence_validator:
            final_answer = self.coherence_validator.get_hedged_response(
                answer, coherence_metrics)
            logger.debug("Applied hedging to response")

        # 3. Add uncertainty warning
        if fallback_action.flag_uncertainty and fallback_action.uncertainty_message:
            final_answer = fallback_action.uncertainty_message + "\n\n" + final_answer
            logger.debug("Added uncertainty flag to response")

        return final_answer, source_documents, confidence_scores

    def _validate_and_apply_coherence(self, question: str, source_documents, answer: str,
                                      confidence_scores, original_k: int, include_scores: bool,
                                      enable_coherence_fallback: bool):
        '''Validate coherence and apply fallbacks if needed.'''
        if not (self.enable_coherence_validation and self.coherence_validator and enable_coherence_fallback):
            return answer, source_documents, confidence_scores, None, None

        try:
            coherence_metrics, fallback_action = self.coherence_validator.validate_coherence(
                query=question,
                retrieved_chunks=source_documents,
                generated_answer=answer
            )

            final_answer, source_documents, confidence_scores = self._apply_coherence_fallbacks(
                question, answer, source_documents, confidence_scores,
                coherence_metrics, fallback_action, original_k, include_scores
            )

            return final_answer, source_documents, confidence_scores, coherence_metrics, fallback_action

        except Exception as e:
            logger.error(f"Error during coherence validation: {e}")
            return answer, source_documents, confidence_scores, None, None

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        include_sources: bool = True,
        include_scores: bool = True,
        enable_coherence_fallback: bool = True
    ) -> QueryResult:
        '''Query the RAG system with a question.'''
        logger.info(f"Processing query: {question}")

        k = k or settings.top_k_results
        original_k = k

        # Retrieve documents
        source_documents, confidence_scores = self._retrieve_documents(
            question, k, include_scores)

        if not source_documents:
            logger.warning("No relevant chunks found for query")
            return QueryResult(
                query=question,
                answer="I couldn't find any relevant information in the document chunks to answer your question.",
                source_documents=[],
                confidence_scores=[],
                metadata={"retrieval_count": 0}
            )

        # Generate answer
        context = self._prepare_context(source_documents)
        answer = self._generate_answer(context, question)

        # Validate coherence and apply fallbacks
        final_answer, source_documents, confidence_scores, coherence_metrics, fallback_action = \
            self._validate_and_apply_coherence(
                question, source_documents, answer, confidence_scores,
                original_k, include_scores, enable_coherence_fallback
            )

        # Prepare metadata
        metadata = {
            "retrieval_count": len(source_documents),
            "context_length": len(context),
            "prompt_length": len(self.prompt_template.format(context=context, question=question)),
            "model_used": settings.chat_model,
            "original_k": original_k,
            "final_k": len(source_documents),
            "coherence_validation_enabled": self.enable_coherence_validation
        }

        if coherence_metrics:
            metadata.update({
                "coherence_level": coherence_metrics.coherence_level.value,
                "coherence_delta": coherence_metrics.coherence_delta,
                "query_chunk_cosine": coherence_metrics.query_chunk_cosine,
                "chunk_generation_cosine": coherence_metrics.chunk_generation_cosine,
                "query_generation_cosine": coherence_metrics.query_generation_cosine
            })

        return QueryResult(
            query=question,
            answer=final_answer,
            source_documents=source_documents if include_sources else [],
            confidence_scores=confidence_scores if include_scores and confidence_scores is not None else [],
            metadata=metadata,
            coherence_metrics=coherence_metrics,
            fallback_action=fallback_action
        )

    def _extract_question_keywords(self, question: str) -> set:
        '''Extract meaningful keywords from question by removing stop words.'''
        question_words = set(question.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were',
                      'what', 'when', 'where', 'who', 'why', 'how', 'which'}
        return question_words - stop_words

    def _calculate_chunk_score(self, doc: Document, similarity_score: float,
                               question_keywords: set) -> tuple:
        '''Calculate combined relevance score for a document chunk.'''
        content_lower = doc.page_content.lower()
        sim_score = float(similarity_score)

        # Keyword overlap bonus
        content_words = set(content_lower.split())
        keyword_matches = len(question_keywords & content_words)
        keyword_score = min(
            keyword_matches / max(len(question_keywords), 1), 1.0)

        # Content length penalty
        length_penalty = self._get_length_penalty(len(doc.page_content))

        # Combined score
        combined_score = (
            sim_score * 0.6 +
            keyword_score * 0.3 +
            length_penalty * 0.1
        )

        return (doc, combined_score, doc.metadata.get('source', 'unknown'))

    def _get_length_penalty(self, content_length: int) -> float:
        '''Calculate penalty based on content length.'''
        if content_length < 100:
            return 0.8  # Short chunks are less likely to be comprehensive
        elif content_length > 2000:
            return 0.9  # Very long chunks might be too general
        else:
            return 1.0

    def _select_diverse_chunks(self, scored_chunks: List[tuple], target_k: int) -> List[tuple]:
        '''Select chunks with source diversity preference.'''
        selected_chunks = []
        source_counts = {}

        # First pass: Select top chunks with source diversity
        for doc, score, source in scored_chunks:
            if len(selected_chunks) >= target_k:
                break

            # Prefer variety, but allow up to 3 chunks from same source
            if source_counts.get(source, 0) < 3 or len(selected_chunks) < target_k // 2:
                selected_chunks.append((doc, score))
                source_counts[source] = source_counts.get(source, 0) + 1

        # Second pass: Fill remaining slots if needed
        if len(selected_chunks) < target_k:
            selected_chunks = self._fill_remaining_slots(
                scored_chunks, selected_chunks, target_k
            )

        return selected_chunks

    def _fill_remaining_slots(self, scored_chunks: List[tuple],
                              selected_chunks: List[tuple], target_k: int) -> List[tuple]:
        '''Fill remaining slots with high-scoring chunks avoiding duplicates.'''
        for doc, score, _ in scored_chunks:
            if len(selected_chunks) >= target_k:
                break
            if not any(d.page_content == doc.page_content for d, _ in selected_chunks):
                selected_chunks.append((doc, score))
        return selected_chunks

    def _rerank_and_filter_chunks(
        self,
        question: str,
        doc_score_pairs: List[tuple],
        target_k: int
    ) -> List[tuple]:
        '''Rerank and filter chunks using multiple criteria.'''
        if not doc_score_pairs:
            return []

        # Extract keywords for relevance scoring
        question_keywords = self._extract_question_keywords(question)

        # Calculate enhanced scores for each chunk
        scored_chunks = [
            self._calculate_chunk_score(
                doc, similarity_score, question_keywords)
            for doc, similarity_score in doc_score_pairs
        ]

        # Sort by combined score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity selection
        selected_chunks = self._select_diverse_chunks(scored_chunks, target_k)

        logger.info(
            f"Reranked {len(doc_score_pairs)} chunks to {len(selected_chunks)} using enhanced scoring")

        return selected_chunks[:target_k]

    def _filter_chunks_by_relevance(
        self,
        question: str,
        documents: List[Document],
        target_k: int
    ) -> List[Document]:
        '''Filter chunks by relevance when scores aren't available.'''
        if not documents:
            return []

        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Calculate relevance scores
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()

            # Count keyword matches
            content_words = set(content_lower.split())
            matches = len(question_words & content_words)

            # Normalize by question length
            relevance_score = matches / max(len(question_words), 1)

            scored_docs.append((doc, relevance_score))

        # Sort by relevance
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        filtered = [doc for doc, _ in scored_docs[:target_k]]

        logger.info(
            f"Filtered {len(documents)} chunks to {len(filtered)} based on relevance")

        return filtered

    def _prepare_context(self, documents: List[Document]) -> str:
        '''Prepare context string from retrieved documents.'''
        context_parts = []

        for i, doc in enumerate(documents, 1):
            # Extract source info
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', i)

            # Format document chunk
            context_part = f"Document {i} (Source: {source}, Chunk: {chunk_id}):\n{doc.page_content}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def batch_query(self, questions: List[str], **kwargs) -> List[QueryResult]:
        '''Process multiple queries in batch.'''
        results = []

        for question in questions:
            try:
                result = self.query(question, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                # Add error result
                error_result = QueryResult(
                    query=question,
                    answer=f"Error processing question: {str(e)}",
                    source_documents=[],
                    confidence_scores=[],
                    metadata={"error": str(e)}
                )
                results.append(error_result)

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        '''Get statistics about the document index.'''
        # This would need to be implemented based on the vector store
        # For now, return basic info
        return {
            "index_name": self.index_name,
            "vector_store_type": settings.vector_db_type,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        }

    def clear_index(self) -> None:
        '''Clear the document index and delete all documents/chunks.'''
        self.document_index.clear_index()
        logger.info(f"Document index '{self.index_name}' cleared completely")


class ConversationalRAG(RAGEngine):
    '''Extended RAG engine with conversation memory.'''

    def __init__(self, index_name: str = "default", enable_coherence_validation: bool = True):
        super().__init__(index_name, enable_coherence_validation)
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10

    def conversational_query(self, question: str, **kwargs) -> QueryResult:
        '''Query with conversation context.'''
        # Add conversation context to the question if history exists
        if self.conversation_history:
            context_question = self._add_conversation_context(question)
        else:
            context_question = question

        # Get the answer
        result = self.query(context_question, **kwargs)

        # Update conversation history
        self.conversation_history.append({
            "question": question,
            "answer": result.answer
        })

        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

        return result

    def _add_conversation_context(self, question: str) -> str:
        '''Add conversation history to the current question.'''
        if not self.conversation_history:
            return question

        context_parts = ["Previous conversation context:"]

        # Add last few exchanges
        recent_history = self.conversation_history[-3:]  # Last 3 exchanges
        for i, exchange in enumerate(recent_history, 1):
            context_parts.append(f"Q{i}: {exchange['question']}")
            # Truncate long answers
            context_parts.append(f"A{i}: {exchange['answer'][:200]}...")

        context_parts.append(f"\nCurrent question: {question}")

        return "\n".join(context_parts)

    def clear_conversation(self) -> None:
        '''Clear conversation history.'''
        self.conversation_history = []
        logger.info("Conversation history cleared")
