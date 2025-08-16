"""
RAG (Retrieval-Augmented Generation) engine for document Q&A.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
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
            api_key=settings.openai_api_key,
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        # Create prompt template
        self.prompt_template = self._create_prompt_template()

        logger.info(f"RAG Engine initialized with index: {index_name}")

    def _create_prompt_template(self) -> PromptTemplate:
        '''Create the prompt template for Q&A.'''
        template = '''You are a helpful AI assistant that answers questions based on the provided document context.
Use only the information from the context to answer questions. If the context doesn't contain enough information
to answer the question, say so clearly.

Context from documents:
{context}

Question: {question}

Instructions:
1. Answer based only on the provided context
2. Be specific and cite relevant parts of the context
3. If the context is insufficient, state this clearly
4. Provide a comprehensive answer when possible
5. If there are multiple relevant pieces of information, synthesize them
6. Respond in paragraph format, avoid using markdown

Answer:'''

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def add_documents(self, file_paths: List[str], metadata: Dict[str, Any] = None) -> int:
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

    def query(
        self,
        question: str,
        k: int = None,
        include_sources: bool = True,
        include_scores: bool = True,
        enable_coherence_fallback: bool = True
    ) -> QueryResult:
        '''Query the RAG system with a question.'''
        logger.info(f"Processing query: {question}")

        k = k or settings.top_k_results
        original_k = k

        # Retrieve relevant documents
        if include_scores and hasattr(self.document_index.vector_store, 'similarity_search_with_score'):
            doc_score_pairs = self.document_index.search_with_scores(
                question, k)
            source_documents = [doc for doc, score in doc_score_pairs]
            confidence_scores = [float(score)
                                 for doc, score in doc_score_pairs]
        else:
            source_documents = self.document_index.search(question, k)
            confidence_scores = [1.0] * len(source_documents)

        if not source_documents:
            logger.warning("No relevant documents found for query")
            return QueryResult(
                query=question,
                answer="I couldn't find any relevant information in the documents to answer your question.",
                source_documents=[],
                confidence_scores=[],
                metadata={"retrieval_count": 0}
            )

        # Prepare context
        context = self._prepare_context(source_documents)

        # Generate answer
        prompt = self.prompt_template.format(
            context=context, question=question)

        try:
            answer = self.llm.predict(prompt)
            logger.info("Successfully generated answer")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "I encountered an error while generating the answer. Please try again."

        # Validate semantic coherence and apply fallbacks if enabled
        coherence_metrics = None
        fallback_action = None
        final_answer = answer

        if self.enable_coherence_validation and self.coherence_validator and enable_coherence_fallback:
            try:
                coherence_metrics, fallback_action = self.coherence_validator.validate_coherence(
                    query=question,
                    retrieved_chunks=source_documents,
                    generated_answer=answer,
                    chunk_scores=confidence_scores
                )

                # Apply fallback behaviors
                if coherence_metrics.needs_fallback:
                    logger.info(
                        f"Applying coherence fallbacks. Level: {coherence_metrics.coherence_level.value}")

                    # 1. Boost k if needed and re-retrieve
                    if fallback_action.boost_k and fallback_action.new_k_value and fallback_action.new_k_value > original_k:
                        logger.info(
                            f"Boosting k from {original_k} to {fallback_action.new_k_value}")

                        # Re-retrieve with boosted k
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
                            boosted_confidence_scores = [
                                1.0] * len(boosted_source_documents)

                        # Re-generate with expanded context if we got more documents
                        if len(boosted_source_documents) > len(source_documents):
                            boosted_context = self._prepare_context(
                                boosted_source_documents)
                            boosted_prompt = self.prompt_template.format(
                                context=boosted_context, question=question)

                            try:
                                boosted_answer = self.llm.predict(
                                    boosted_prompt)
                                logger.info(
                                    "Successfully regenerated answer with boosted retrieval")

                                # Update variables for final result
                                source_documents = boosted_source_documents
                                confidence_scores = boosted_confidence_scores
                                context = boosted_context
                                answer = boosted_answer

                            except Exception as e:
                                logger.error(
                                    f"Error regenerating answer with boosted k: {e}")
                                # Keep original answer

                    # 2. Apply hedging to output
                    if fallback_action.hedge_output:
                        final_answer = self.coherence_validator.get_hedged_response(
                            answer, coherence_metrics)
                        logger.debug("Applied hedging to response")

                    # 3. Add uncertainty warning if flagged
                    if fallback_action.flag_uncertainty and fallback_action.uncertainty_message:
                        final_answer = fallback_action.uncertainty_message + "\n\n" + final_answer
                        logger.debug("Added uncertainty flag to response")

            except Exception as e:
                logger.error(f"Error during coherence validation: {e}")
                # Continue with original answer if coherence validation fails

        # Prepare metadata
        metadata = {
            "retrieval_count": len(source_documents),
            "context_length": len(context),
            "prompt_length": len(prompt),
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

        result = QueryResult(
            query=question,
            answer=final_answer,
            source_documents=source_documents if include_sources else [],
            confidence_scores=confidence_scores if include_scores else [],
            metadata=metadata,
            coherence_metrics=coherence_metrics,
            fallback_action=fallback_action
        )

        return result

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
        '''Clear the document index.'''
        self.document_index = DocumentIndex(self.index_name)
        logger.info("Document index cleared")


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
