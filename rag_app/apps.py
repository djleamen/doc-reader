"""
RAG App configuration.

Written by DJ Leamen (2025-2026)
"""

from django.apps import AppConfig


class RagAppConfig(AppConfig):
    '''
    Django application configuration for the RAG Document Q&A app.
    
    Defines the default auto field type and application metadata
    for the RAG application.
    '''
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag_app'
    verbose_name = 'RAG Document Q&A'
