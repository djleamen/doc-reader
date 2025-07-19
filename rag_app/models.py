"""
Models for the RAG Document Q&A system.
"""
import uuid

from django.contrib.auth.models import User
from django.db import models


class DocumentIndex(models.Model):
    """Model to track document indexes."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    document_count = models.IntegerField(default=0)
    chunk_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return str(self.name)


class Document(models.Model):
    """Model to track uploaded documents."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    index = models.ForeignKey(DocumentIndex, on_delete=models.CASCADE, related_name='documents')
    filename = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=500)
    file_path = models.CharField(max_length=1000)
    file_size = models.BigIntegerField()
    file_type = models.CharField(max_length=10)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    chunk_count = models.IntegerField(default=0)
    processing_error = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return str(self.original_filename)


class QuerySession(models.Model):
    """Model to track query sessions for conversational mode."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_key = models.CharField(max_length=255, db_index=True)
    index = models.ForeignKey(DocumentIndex, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']


class Query(models.Model):
    """Model to track individual queries."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(QuerySession, on_delete=models.CASCADE, related_name='queries', null=True, blank=True)
    index = models.ForeignKey(DocumentIndex, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    k_results = models.IntegerField(default=5)
    include_sources = models.BooleanField(default=True)
    include_scores = models.BooleanField(default=True)
    response_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata
    context_length = models.IntegerField(default=0)
    retrieval_count = models.IntegerField(default=0)
    model_used = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Query: {str(self.question)[:50]}..."
