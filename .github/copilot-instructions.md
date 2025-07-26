# GitHub Copilot Instructions for RAG Document Q&A System

## Project Overview

This is a Django-based RAG (Retrieval-Augmented Generation) Document Q&A system that enables users to upload documents and query them using AI-powered natural language processing. The system uses vector embeddings, multiple document formats, and conversational AI capabilities.

## Architecture

### Core Components
- **Django Web Framework**: Main web application with REST API
- **RAG Engine**: Core retrieval and generation logic using LangChain
- **Vector Store**: FAISS/ChromaDB/Pinecone for document embeddings
- **Document Processor**: Multi-format document parsing (PDF, DOCX, TXT, MD)
- **CLI Interface**: Command-line tools for batch operations

### Key Technologies
- Django 5.2.4 with DRF (Django REST Framework)
- LangChain for RAG implementation
- OpenAI API for embeddings and chat completion
- FAISS for vector similarity search
- Bootstrap 5 for frontend UI
- SQLite for metadata storage

## Code Style and Conventions

### Python Style
- Follow PEP 8 guidelines
- Use type hints for all functions and methods
- Prefer descriptive variable names over abbreviations
- Use docstrings for all classes and functions
- Import organization: standard library, third-party, local imports

### Django Conventions
- Use Django's built-in features (models, views, serializers)
- Follow Django's naming conventions for models, views, and URLs
- Use Django's authentication and session management
- Implement proper error handling with Django's exception classes

### Error Handling
```python
from django.http import Http404, JsonResponse
from rest_framework import status
from loguru import logger

try:
    # Operation
    pass
except SpecificException as e:
    logger.error(f"Error description: {e}")
    return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
```

### Logging Pattern
```python
from loguru import logger

logger.info("Operation started")
logger.warning("Warning message")
logger.error("Error occurred: {}", error_message)
```

## File Organization

### Core Directories
- `django_app/`: Django project settings and configuration
- `rag_app/`: Main Django application with models, views, APIs
- `src/`: Core RAG logic and document processing
- `templates/`: HTML templates with Bootstrap UI
- `static/`: CSS, JS, and static assets
- `documents/`: Uploaded document storage
- `indexes/`: Vector database storage
- `logs/`: Application logs

### Key Files
- `main.py`: Unified entry point for all operations
- `manage.py`: Django management commands
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Container orchestration
- `.env`: Environment variables (not in repo)

## Common Patterns

### Model Definition
```python
from django.db import models
import uuid

class DocumentIndex(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'document_indexes'
        verbose_name = 'Document Index'
        verbose_name_plural = 'Document Indexes'
    
    def __str__(self):
        return self.name
```

### API View Pattern
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')
class QueryView(APIView):
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if serializer.is_valid():
            # Process query
            return Response(result, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

### RAG Engine Usage
```python
from src.rag_engine import RAGEngine, ConversationalRAG

def get_rag_engine(index_name: str = "default"):
    if index_name not in _rag_engines:
        _rag_engines[index_name] = RAGEngine(index_name)
    return _rag_engines[index_name]
```

## Development Guidelines

### When Adding New Features

1. **Models**: Always add to `rag_app/models.py` with proper relationships
2. **APIs**: Add new endpoints to `rag_app/views.py` and `rag_app/urls.py`
3. **Processing**: Core logic goes in `src/` modules
4. **Templates**: Use Bootstrap classes and the existing glass-card design
5. **Tests**: Add comprehensive tests in `rag_app/tests.py`

### Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Environment Variables
Always use settings from `src/config.py` which loads from `.env`:
```python
from src.config import settings
openai_key = settings.openai_api_key
```

### Vector Store Operations
```python
from src.vector_store import DocumentIndex

# Initialize index
index = DocumentIndex(index_name, vector_store_type="faiss")

# Add documents
chunks = document_processor.process_document(file_path)
index.add_documents(chunks)

# Search
results = index.similarity_search(query, k=5)
```

## Security Considerations

- Never commit API keys or secrets
- Use Django's CSRF protection
- Validate all file uploads (size, type, content)
- Sanitize user inputs before processing
- Use proper authentication for admin endpoints
- Log security events to `logs/security.log`

## Performance Guidelines

- Use pagination for large result sets
- Implement caching for frequently accessed data
- Optimize vector search with appropriate k values
- Use background tasks for document processing
- Monitor memory usage with large documents

## Testing Patterns

```python
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile

class DocumentUploadTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.test_file = SimpleUploadedFile(
            "test.txt", 
            b"test content", 
            content_type="text/plain"
        )
    
    def test_upload_document(self):
        response = self.client.post('/api/upload-documents/', {
            'files': [self.test_file],
            'index_name': 'test'
        })
        self.assertEqual(response.status_code, 200)
```

## Common Issues and Solutions

### Document Processing
- **Large files**: Implement chunking and progress tracking
- **Unsupported formats**: Add format validation before processing
- **Memory issues**: Process documents in batches

### Vector Search
- **Poor results**: Tune chunk size and overlap parameters
- **Slow queries**: Optimize k parameter and use caching
- **Index corruption**: Implement backup and recovery

### Django Integration
- **Static files**: Run `collectstatic` for production
- **Database locks**: Use transactions for batch operations
- **Session management**: Implement proper cleanup

## Helpful Commands

```bash
# Development
python main.py start                    # Quick start with setup
python manage.py runserver             # Django development server
python manage.py shell                 # Django shell

# CLI Operations
python main.py cli add document.pdf    # Add document
python main.py cli query "question"    # Query system
python main.py cli stats               # Show statistics

# Database
python manage.py makemigrations        # Create migrations
python manage.py migrate               # Apply migrations
python manage.py createsuperuser       # Create admin user

# Docker
docker-compose up                      # Start with Docker
docker-compose down                    # Stop containers
```

## Integration Examples

### Adding a New Document Format
1. Update `document_processor.py` with new format handler
2. Add format to `SUPPORTED_FORMATS` in config
3. Update file validation in views
4. Add tests for new format

### Creating a New API Endpoint
1. Add serializer to `serializers.py`
2. Add view to `views.py`
3. Add URL pattern to `urls.py`
4. Update API documentation
5. Add integration tests

### Extending the RAG Engine
1. Add new functionality to `rag_engine.py`
2. Update configuration in `config.py`
3. Integrate with Django views
4. Add CLI commands if needed

## Best Practices for AI-Assisted Development

- Always understand the existing codebase patterns before suggesting changes
- Maintain consistency with Django conventions and project structure
- Consider performance implications of RAG operations
- Implement proper error handling and logging
- Write comprehensive tests for new functionality
- Document complex RAG logic and algorithms
- Follow the existing import patterns and code organization

## Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FAISS Documentation](https://faiss.ai/)
- [Bootstrap 5 Documentation](https://getbootstrap.com/docs/5.1/)

This system is designed to be extensible and maintainable. When contributing, always consider the impact on document processing performance, user experience, and system scalability.
