"""
Azure OpenAI service for embeddings and chat completions.
Uses Managed Identity for secure authentication.
"""

import time
from typing import List, Optional, Any
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai import RateLimitError, APIError
from azure.core.exceptions import AzureError
from loguru import logger

from src.az_config import azure_settings


class AzureOpenAIService:
    """
    Experimental Azure OpenAI service with:
    - Managed Identity authentication
    - Automatic retry with exponential backoff
    - Connection pooling
    - Error handling
    - Performance monitoring
    """

    def __init__(self):
        """Initialize Azure OpenAI client with proper authentication."""
        self.settings = azure_settings
        self.client: AzureOpenAI
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        Initialize Azure OpenAI client with appropriate authentication.
        Uses Managed Identity when available, falls back to API key.
        """
        try:
            if self.settings.use_managed_identity:
                # Use Managed Identity (recommended for production)
                logger.info("Initializing Azure OpenAI with Managed Identity")
                credential = self.settings.get_credential()
                # Get token for Azure OpenAI
                token = credential.get_token(
                    "https://cognitiveservices.azure.com/.default")

                self.client = AzureOpenAI(
                    azure_endpoint=self.settings.openai_endpoint,
                    api_version=self.settings.openai_api_version,
                    azure_ad_token=token.token,
                )
            elif self.settings.openai_api_key:
                # Use API Key (for development/testing)
                logger.info("Initializing Azure OpenAI with API Key")
                self.client = AzureOpenAI(
                    azure_endpoint=self.settings.openai_endpoint,
                    api_key=self.settings.openai_api_key,
                    api_version=self.settings.openai_api_version,
                )
            else:
                raise ValueError(
                    "Azure OpenAI requires either Managed Identity "
                    "or API Key authentication"
                )

            logger.info("Azure OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        max_retries = self.settings.max_retries

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                self._handle_rate_limit_error(e, attempt, max_retries)
            except (AzureError, APIError) as e:
                self._handle_azure_api_error(e, attempt, max_retries)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    def _handle_rate_limit_error(self, error: RateLimitError, attempt: int, max_retries: int) -> None:
        """Handle OpenAI rate limit errors with retry logic."""
        wait_time = 60
        logger.warning(
            f"Rate limit reached (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Waiting {wait_time}s before retry..."
        )
        time.sleep(wait_time)

        if attempt == max_retries - 1:
            logger.error(f"Max retries reached after rate limiting: {error}")
            raise error

    def _handle_azure_api_error(self, error: Exception, attempt: int, max_retries: int) -> None:
        """Handle Azure API errors with exponential backoff or rate limit handling."""
        error_str = str(error)

        if "429" in error_str or "RateLimitReached" in error_str:
            self._handle_rate_limit_in_error(error, attempt, max_retries)
        else:
            self._handle_retriable_error(error, attempt, max_retries)

    def _handle_rate_limit_in_error(self, error: Exception, attempt: int, max_retries: int) -> None:
        """Handle rate limit detected in error message."""
        wait_time = 60
        logger.warning(
            f"Rate limit detected (attempt {attempt + 1}/{max_retries}). "
            f"Waiting {wait_time}s before retry..."
        )
        time.sleep(wait_time)

        if attempt == max_retries - 1:
            logger.error(f"Max retries reached after {max_retries} attempts")
            raise error

    def _handle_retriable_error(self, error: Exception, attempt: int, max_retries: int) -> None:
        """Handle retriable errors with exponential backoff."""
        if attempt == max_retries - 1:
            logger.error(f"Max retries reached. Last error: {error}")
            raise error

        wait_time = self.settings.retry_delay * (2 ** attempt)
        logger.warning(
            f"Attempt {attempt + 1} failed: {error}. "
            f"Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)

    def get_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        Generate embeddings for text using Azure OpenAI.
        Processes in batches to avoid rate limits.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch (default: 20 for S0 tier)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            logger.info(
                f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
            start_time = time.time()

            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

                def _create_embeddings(batch_texts):
                    response = self.client.embeddings.create(
                        model=self.settings.openai_embedding_deployment,
                        input=batch_texts
                    )
                    return [item.embedding for item in response.data]

                batch_embeddings = self._retry_with_backoff(
                    _create_embeddings, batch)
                all_embeddings.extend(batch_embeddings)

                # Add delay between batches to avoid rate limiting (except for last batch)
                if i + batch_size < len(texts):
                    delay = 3.0  # 3 second delay for S0 tier rate limits
                    logger.info(f"Waiting {delay}s before next batch...")
                    time.sleep(delay)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Generated {len(all_embeddings)} embeddings in {elapsed_time:.2f}s "
                f"({len(texts)/elapsed_time:.2f} texts/s)"
            )

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []

    def chat_completion(
        self,
        messages: List[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate chat completion using Azure OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        try:
            logger.info("Generating chat completion")
            start_time = time.time()

            temperature = temperature or self.settings.temperature
            max_tokens = max_tokens or self.settings.max_tokens

            def _create_completion():
                response = self.client.chat.completions.create(
                    model=self.settings.openai_chat_deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,  # Always use non-streaming for retry compatibility
                )
                return response.choices[0].message.content

            result = self._retry_with_backoff(_create_completion)

            elapsed_time = time.time() - start_time
            logger.info(f"Chat completion generated in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise

    def chat_completion_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate chat completion with context (for RAG).

        Args:
            query: User query
            context: Retrieved context from documents
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        default_system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based on the "
            "provided context. If the answer cannot be found in the context, say so. "
            "Be concise and accurate."
        )
        system_prompt = system_prompt or default_system_prompt

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def validate_connection(self) -> bool:
        """
        Validate Azure OpenAI connection.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            logger.info("Validating Azure OpenAI connection")
            test_embedding = self.get_embedding("test")

            if test_embedding and len(test_embedding) > 0:
                logger.info("Azure OpenAI connection validated successfully")
                return True

            logger.warning("Azure OpenAI connection validation failed")
            return False

        except Exception as e:
            logger.error(f"Azure OpenAI connection validation error: {e}")
            return False


class AzureOpenAIServiceSingleton:
    """Singleton wrapper for AzureOpenAIService."""

    _instance: Optional[AzureOpenAIService] = None

    @classmethod
    def get_instance(cls) -> AzureOpenAIService:
        """
        Get or create Azure OpenAI service instance (singleton pattern).

        Returns:
            AzureOpenAIService instance
        """
        if cls._instance is None:
            cls._instance = AzureOpenAIService()

        return cls._instance


def get_azure_openai_service() -> AzureOpenAIService:
    """
    Get or create Azure OpenAI service instance (singleton pattern).

    Returns:
        AzureOpenAIService instance
    """
    return AzureOpenAIServiceSingleton.get_instance()
