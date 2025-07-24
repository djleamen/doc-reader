"""
RAG (Retrieval-Augmented Generation) engine for document Q&A.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from loguru import logger

from src.config import settings
from src.vector_store import DocumentIndex
from src.document_processor import DocumentProcessor


@dataclass
class QueryResult:
    """Result of a Q&A query."""
    query: str
    answer: str
    source_documents: List[Document]
    confidence_scores: List[float]
    metadata: Dict[str, Any]


class RAGEngine:
    """Main RAG engine for document Q&A."""

    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.document_index = DocumentIndex(index_name)
        self.document_processor = DocumentProcessor()

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
        """Create the prompt template for Q&A."""
        template = """You are a helpful AI assistant that answers questions based on the provided document context.
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

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def add_documents(self, file_paths: List[str], metadata: Dict[str, Any] = None) -> int:
        """Add documents to the RAG system."""
        all_documents = []

        for file_path in file_paths:
            try:
                logger.info(f"Processing document: {file_path}")
                documents = self.document_processor.process_document(file_path, metadata)
                all_documents.extend(documents)
                logger.info(f"Added {len(documents)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if all_documents:
            self.document_index.add_documents(all_documents)
            logger.info(f"Successfully added {len(all_documents)} total document chunks")
            return len(all_documents)
        else:
            logger.warning("No documents were successfully processed")
            return 0

    def query(
        self,
        question: str,
        k: int = None,
        include_sources: bool = True,
        include_scores: bool = True
    ) -> QueryResult:
        """Query the RAG system with a question."""
        logger.info(f"Processing query: {question}")

        k = k or settings.top_k_results

        # Retrieve relevant documents
        if include_scores and hasattr(self.document_index.vector_store, 'similarity_search_with_score'):
            doc_score_pairs = self.document_index.search_with_scores(question, k)
            source_documents = [doc for doc, score in doc_score_pairs]
            confidence_scores = [float(score) for doc, score in doc_score_pairs]
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
        prompt = self.prompt_template.format(context=context, question=question)

        try:
            answer = self.llm.predict(prompt)
            logger.info("Successfully generated answer")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "I encountered an error while generating the answer. Please try again."

        # Prepare metadata
        metadata = {
            "retrieval_count": len(source_documents),
            "context_length": len(context),
            "prompt_length": len(prompt),
            "model_used": settings.chat_model
        }

        result = QueryResult(
            query=question,
            answer=answer,
            source_documents=source_documents if include_sources else [],
            confidence_scores=confidence_scores if include_scores else [],
            metadata=metadata
        )

        return result

    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from retrieved documents."""
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
        """Process multiple queries in batch."""
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
        """Get statistics about the document index."""
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
        """Clear the document index."""
        self.document_index = DocumentIndex(self.index_name)
        logger.info("Document index cleared")


class ConversationalRAG(RAGEngine):
    """Extended RAG engine with conversation memory."""

    def __init__(self, index_name: str = "default"):
        super().__init__(index_name)
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10

    def conversational_query(self, question: str, **kwargs) -> QueryResult:
        """Query with conversation context."""
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
        """Add conversation history to the current question."""
        if not self.conversation_history:
            return question

        context_parts = ["Previous conversation context:"]

        # Add last few exchanges
        recent_history = self.conversation_history[-3:]  # Last 3 exchanges
        for i, exchange in enumerate(recent_history, 1):
            context_parts.append(f"Q{i}: {exchange['question']}")
            context_parts.append(f"A{i}: {exchange['answer'][:200]}...")  # Truncate long answers

        context_parts.append(f"\nCurrent question: {question}")

        return "\n".join(context_parts)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
