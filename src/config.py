"""
Configuration management for the RAG Document Q&A system.
"""
import os
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    
    # Model Configuration
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    chat_model: str = Field("gpt-4-turbo-preview", env="CHAT_MODEL")
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    # Vector Database
    vector_db_type: str = Field("faiss", env="VECTOR_DB_TYPE")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")
    
    # Document Processing
    max_document_size_mb: int = Field(100, env="MAX_DOCUMENT_SIZE_MB")
    supported_formats: str = Field("pdf,docx,txt,md", env="SUPPORTED_FORMATS")
    
    @property
    def supported_formats_list(self) -> List[str]:
        """Get supported formats as a list."""
        if isinstance(self.supported_formats, str):
            return [fmt.strip() for fmt in self.supported_formats.split(',')]
        return self.supported_formats
    
    # API Settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Paths
    documents_dir: str = Field("documents", env="DOCUMENTS_DIR")
    index_dir: str = Field("indexes", env="INDEX_DIR")
    logs_dir: str = Field("logs", env="LOGS_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
