"""
Admin configuration for RAG app models.
"""
from typing import Union, List, Sequence, Any
from django.contrib import admin
from django.http import HttpRequest
from .models import DocumentIndex, Document, QuerySession, Query


@admin.register(DocumentIndex)
class DocumentIndexAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'document_count', 'chunk_count', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('name', 'description')
    readonly_fields = ('id', 'created_at', 'updated_at', 'document_count', 'chunk_count')
    ordering = ('-updated_at',)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'index', 'file_type', 'file_size', 'processed', 'uploaded_at')
    list_filter = ('processed', 'file_type', 'uploaded_at', 'index')
    search_fields = ('original_filename', 'filename')
    readonly_fields = ('id', 'uploaded_at', 'file_size')
    ordering = ('-uploaded_at',)

    def get_readonly_fields(self, request: HttpRequest, obj=None):
        readonly_fields = ('id', 'uploaded_at', 'file_size')
        if obj:  # editing an existing object
            return readonly_fields + ('file_path', 'chunk_count')
        return readonly_fields


@admin.register(QuerySession)
class QuerySessionAdmin(admin.ModelAdmin):
    list_display = ('session_key', 'index', 'user', 'created_at', 'updated_at')
    list_filter = ('index', 'created_at')
    search_fields = ('session_key',)
    readonly_fields = ('id', 'created_at', 'updated_at')
    ordering = ('-updated_at',)


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ('question_preview', 'index', 'session', 'response_time', 'created_at')
    list_filter = ('index', 'created_at', 'include_sources', 'include_scores')
    search_fields = ('question', 'answer')
    readonly_fields = ('id', 'created_at', 'response_time')
    ordering = ('-created_at',)

    def question_preview(self, obj):
        return obj.question[:50] + '...' if len(obj.question) > 50 else obj.question
    question_preview.short_description = 'Question'  # type: ignore

    fieldsets = (
        ('Query Information', {
            'fields': ('question', 'answer', 'index', 'session')
        }),
        ('Query Parameters', {
            'fields': ('k_results', 'include_sources', 'include_scores')
        }),
        ('Performance Metrics', {
            'fields': ('response_time', 'context_length', 'retrieval_count', 'model_used')
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        })
    )
