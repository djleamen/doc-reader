"""
Azure OpenAI service for embeddings and chat completions.
Uses Managed Identity for secure authentication.

Written by DJ Leamen (2025-2026)
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
        '''
        Initialize Azure OpenAI service.
        
        :param self: AzureOpenAIService instance
        '''
        self.settings = azure_settings
        self.client: AzureOpenAI
        self._initialize_client()

    def _initialize_client(self) -> None:
        '''
        Initialize Azure OpenAI client with authentication.
        
        :param self: AzureOpenAIService instance
        '''
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
        '''
        Retry function with exponential backoff on rate limit and retriable errors.
        
        :param self: AzureOpenAIService instance
        :param func: Function to execute with retry logic
        :param args: Positional arguments to pass to function
        :param kwargs: Keyword arguments to pass to function
        :return: Result from successful function execution
        :rtype: Any
        '''
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

    def _handle_rate_limit_error(self, error: RateLimitError, 
                                 attempt: int, max_retries: int) -> None:
        '''
        Handle rate limit errors with fixed wait time.
        
        :param self: AzureOpenAIService instance
        :param error: RateLimitError instance
        :type error: RateLimitError
        :param attempt: Current retry attempt number
        :type attempt: int
        :param max_retries: Maximum number of retry attempts
        :type max_retries: int
        '''
        wait_time = 60
        logger.warning(
            f"Rate limit reached (attempt {attempt + 1}/{max_retries}): {error}. "
            f"Waiting {wait_time}s before retry..."
        )
        time.sleep(wait_time)

        if attempt == max_retries - 1:
            logger.error(f"Max retries reached after rate limiting: {error}")
            raise error

    def _handle_azure_api_error(self, error: Exception, attempt: int, 
                                max_retries: int) -> None:
        '''
        Handle Azure API errors with exponential backoff or rate limit handling.
        
        :param self: AzureOpenAIService instance
        :param error: Exception from Azure API
        :type error: Exception
        :param attempt: Current retry attempt number
        :type attempt: int
        :param max_retries: Maximum number of retry attempts
        :type max_retries: int
        '''
        error_str = str(error)

        if "429" in error_str or "RateLimitReached" in error_str:
            self._handle_rate_limit_in_error(error, attempt, max_retries)
        else:
            self._handle_retriable_error(error, attempt, max_retries)

    def _handle_rate_limit_in_error(self, error: Exception, attempt: int, max_retries: int) -> None:
        '''
        Handle rate limit errors detected in generic exceptions.
        
        :param self: AzureOpenAIService instance
        :param error: Exception from Azure API
        :type error: Exception
        :param attempt: Current retry attempt number
        :type attempt: int
        :param max_retries: Maximum number of retry attempts
        :type max_retries: int
        '''
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
        '''
        Handle retriable errors with exponential backoff.
        
        :param self: AzureOpenAIService instance
        :param error: Exception from Azure API
        :type error: Exception
        :param attempt: Current retry attempt number
        :type attempt: int
        :param max_retries: Maximum number of retry attempts
        :type max_retries: int
        '''
        if attempt == max_retries - 1:
            logger.error(f"Max retries reached. Last error: {error}")
            raise error

        wait_time = self.settings.retry_delay * (2 ** attempt)
        logger.warning(
            f"Attempt {attempt + 1} failed: {error}. "
            f"Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)

    def get_embeddings(self, texts: List[str], 
                       batch_size: int = 20) -> List[List[float]]:
        '''
        Generate embeddings for a list of texts.
        
        :param self: AzureOpenAIService instance
        :param texts: List of text strings to embed
        :type texts: List[str]
        :param batch_size: Number of texts to process per batch (default: 20 for S0 tier)
        :type batch_size: int
        :return: List of embedding vectors (1536-dimensional for Ada-002)
        :rtype: List[List[float]]
        '''
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
        '''
        Generate embedding for a single text.
        
        :param self: AzureOpenAIService instance
        :param text: Text string to embed
        :type text: str
        :return: Embedding vector (1536-dimensional for Ada-002)
        :rtype: List[float]
        '''
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []

    def chat_completion(
        self,
        messages: List[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        '''
        Generate chat completion from messages.
        
        :param self: AzureOpenAIService instance
        :param messages: List of message dictionaries with 'role' and 'content'
        :type messages: List[ChatCompletionMessageParam]
        :param temperature: Sampling temperature (0-2), uses config default if None
        :type temperature: Optional[float]
        :param max_tokens: Maximum tokens to generate, uses config default if None
        :type max_tokens: Optional[int]
        :param stream: Description
        :type stream: bool
        :return: Description
        :rtype: str
        '''
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
        '''
        Generate chat completion using provided context.
        
        :param self: AzureOpenAIService instance
        :param query: User query to answer
        :type query: str
        :param context: Retrieved context from documents
        :type context: str
        :param system_prompt: Optional system prompt for LLM behavior
        :type system_prompt: Optional[str]
        :param temperature: Description
        :type temperature: Optional[float]
        :param max_tokens: Description
        :type max_tokens: Optional[int]
        :return: Description
        :rtype: str
        '''
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
        '''
        Validate Azure OpenAI connection by testing embedding generation.
        
        :param self: AzureOpenAIService instance
        :return: True if connection is valid, False otherwise
        :rtype: bool
        '''
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
    '''
    Singleton wrapper for AzureOpenAIService.
    
    Ensures only one instance of AzureOpenAIService is created.
    '''

    _instance: Optional[AzureOpenAIService] = None

    @classmethod
    def get_instance(cls) -> AzureOpenAIService:
        '''
        Get or create AzureOpenAIService instance.
        
        :param cls: AzureOpenAIServiceSingleton class
        :return: AzureOpenAIService instance
        :rtype: AzureOpenAIService
        '''
        if cls._instance is None:
            cls._instance = AzureOpenAIService()

        return cls._instance


def get_azure_openai_service() -> AzureOpenAIService:
    '''
    Get or create AzureOpenAIService instance (singleton pattern).
    
    :return: AzureOpenAIService instance
    :rtype: AzureOpenAIService
    '''
    return AzureOpenAIServiceSingleton.get_instance()
