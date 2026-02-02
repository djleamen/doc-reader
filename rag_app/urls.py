"""
URL configuration for the RAG Document Q&A system.

Defines URL patterns for both API endpoints and web interface views,
including standard RAG pipeline and Azure RAG pipeline routes.

Written by DJ Leamen (2025-2026)
"""

from django.urls import path
from . import views
from . import azure_views

# API URL patterns for RAG Document Q&A system
api_urlpatterns = [
    path('upload-documents/', views.DocumentUploadView.as_view(),
         name='api-upload-documents'),
    path('query/', views.QueryView.as_view(), name='api-query'),
    path('conversational-query/', views.ConversationalQueryView.as_view(),
         name='api-conversational-query'),
    path('index-stats/', views.index_stats, name='api-index-stats'),
    path('conversation/', views.clear_conversation,
         name='api-clear-conversation'),
    path('clear-documents/', views.clear_documents,
         name='api-clear-documents'),
    path('health/', views.health_check, name='api-health'),

    # Azure RAG Pipeline endpoints
    path('azure/upload-documents/', azure_views.AzureDocumentUploadView.as_view(),
         name='api-azure-upload-documents'),
    path('azure/query/', azure_views.AzureQueryView.as_view(),
         name='api-azure-query'),
    path('azure/conversational-query/', azure_views.AzureConversationalQueryView.as_view(),
         name='api-azure-conversational-query'),
    path('azure/index-stats/', azure_views.azure_index_stats,
         name='api-azure-index-stats'),
    path('azure/clear-cache/', azure_views.azure_clear_cache,
         name='api-azure-clear-cache'),
    path('azure/clear-conversation/', azure_views.azure_clear_conversation,
         name='api-azure-clear-conversation'),
    path('azure/health/', azure_views.azure_health_check,
         name='api-azure-health'),
]

# Web URL patterns for user interface
web_urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('test/', views.TestView.as_view(), name='test'),
]

# Combined URL patterns - API patterns are included directly since
# main urls.py already routes them under /api/
urlpatterns = api_urlpatterns + web_urlpatterns
