"""
Configuration management for the RAG Document Q&A system.
"""
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

        # Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4-turbo-preview")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))

        # Vector Database
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", "faiss")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k_results = int(os.getenv("TOP_K_RESULTS", "5"))

        # Document Processing
        self.max_document_size_mb = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "100"))
        self.supported_formats = os.getenv("SUPPORTED_FORMATS", "pdf,docx,txt,md")

        # API Settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "False").lower() == "true"

        # Paths
        self.documents_dir = os.getenv("DOCUMENTS_DIR", "documents")
        self.index_dir = os.getenv("INDEX_DIR", "indexes")
        self.logs_dir = os.getenv("LOGS_DIR", "logs")

    @property
    def supported_formats_list(self) -> List[str]:
        """Get supported formats as a list."""
        return [fmt.strip() for fmt in self.supported_formats.split(',')]


# Global settings instance
settings = Settings()
