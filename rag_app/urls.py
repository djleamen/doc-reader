"""
URL configuration for the RAG app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API URLs
api_urlpatterns = [
    path('upload-documents/', views.DocumentUploadView.as_view(), name='api-upload-documents'),
    path('query/', views.QueryView.as_view(), name='api-query'),
    path('conversational-query/', views.ConversationalQueryView.as_view(), name='api-conversational-query'),
    path('index-stats/', views.index_stats, name='api-index-stats'),
    path('conversation/', views.clear_conversation, name='api-clear-conversation'),
    path('health/', views.health_check, name='api-health'),
]

# Web URLs
web_urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('test/', views.TestView.as_view(), name='test'),
]

# Combined URL patterns - API patterns are included directly since 
# main urls.py already routes them under /api/
urlpatterns = api_urlpatterns + web_urlpatterns
