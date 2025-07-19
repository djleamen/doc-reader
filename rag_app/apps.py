"""
RAG App configuration.
"""
from django.apps import AppConfig


class RagAppConfig(AppConfig):
    """Configuration for the RAG app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag_app'
    verbose_name = 'RAG Document Q&A'
