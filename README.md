# RAG Document Q&A System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/django-5.0%2B-green)](https://www.djangoproject.com/)
[![Azure](https://img.shields.io/badge/azure-enabled-0078D4)](https://azure.microsoft.com/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Django-based document Q&A system using Retrieval-Augmented Generation (RAG) to process and query large documents with AI-powered responses. Features both standard OpenAI integration and **enterprise-ready Azure RAG pipeline**.

## ✨ Features

### Core Features
- **Django Web Interface**: Modern Bootstrap UI with admin panel
- **Large Document Support**: Handle documents up to 800k+ words
- **Multiple Formats**: PDF, DOCX, TXT, and Markdown support
- **REST API**: Django REST Framework for integrations
- **Vector Search**: FAISS/ChromaDB/Pinecone vector databases
- **Conversational Mode**: Context-aware multi-turn conversations
- **Session Management**: User session tracking and conversation history
- **CLI Tools**: Command-line interface for batch operations
- **🎯 Semantic Coherence Validation**: Post-retrieval tracking with automatic fallback behaviors
  - Monitors semantic consistency across query→chunk→generation pipeline
  - Automatic k-boosting when coherence drops
  - Smart output hedging for uncertain answers
  - Configurable coherence thresholds and fallback strategies

### Azure RAG Pipeline (New! - Experimental)

#### Azure Services Integration
- **Azure OpenAI**: Embeddings (Ada-002) and Chat Completion (GPT-4)
- **Azure AI Search**: Vector search with hybrid (vector + keyword) and semantic ranking
- **Azure Document Intelligence**: Advanced document processing with layout analysis, table extraction, and OCR
- **Azure Key Vault**: Secure secrets management (optional)
- **Azure Storage**: Document storage with blob containers (optional)

#### Features
- **Managed Identity Authentication**: Secure, credential-free authentication for Azure-hosted apps
- **Automatic Retry Logic**: Exponential backoff for transient failures
- **Query Result Caching**: In-memory cache with configurable TTL
- **Hybrid Search**: Combines vector similarity with keyword search for better accuracy
- **Semantic Ranking**: Azure AI Search semantic ranking for improved relevance
- **Performance Monitoring**: Built-in metrics and logging
- **Error Handling**: Comprehensive error handling and recovery
- **Health Checks**: Validation endpoints for all Azure services

#### Authentication Options
1. **Managed Identity** (Production - Recommended):
   - No credentials in code or environment
   - Automatic credential rotation
   - Azure RBAC for fine-grained access control
   
2. **Service Principal** (CI/CD):
   - Client ID, Secret, and Tenant ID
   - Suitable for deployment pipelines
   
3. **API Keys** (Development):
   - Simple setup for local development
   - Not recommended for production

## 🚀 Quick Start

### Requirements
- Python 3.8+
- **For Standard Pipeline**: OpenAI API key
- **For Azure Pipeline**: Azure subscription with OpenAI, AI Search, and Document Intelligence resources

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/djleamen/doc-reader
cd doc-reader
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:

```bash
cp .env.example .env
# Edit .env with your API keys and Azure configuration
```

**For Standard Pipeline (OpenAI)**:
- Set `OPENAI_API_KEY` in `.env`

**For Azure Pipeline**:
- Set Azure service endpoints and credentials

5. **Run setup and start server**:

```bash
python main.py start
```

Open your browser to `http://localhost:8000`

## 📖 Usage

### Web Interface
- Upload documents via the web UI
- Ask questions in natural language
- View sources and confidence scores
- Use conversational mode for follow-up questions

### REST API
```bash
# Upload documents
curl -X POST "http://localhost:8000/api/upload-documents/" \
  -F "files=@document.pdf" \
  -F "index_name=default"

# Query documents
curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "index_name": "default"}'
```

### Command Line
```bash
# Add documents
python main.py cli add document.pdf

# Query documents
python main.py cli query "What are the key findings?"

# Interactive mode
python main.py cli interactive --conversational
```

## ⚙️ Configuration

Key environment variables in `.env`:

```env
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
VECTOR_DB_TYPE=faiss              # faiss, chroma, or pinecone
CHUNK_SIZE=1000                   # Text chunk size
CHUNK_OVERLAP=200                 # Overlap between chunks
TOP_K_RESULTS=5                   # Number of results to retrieve
CHAT_MODEL=gpt-4-turbo-preview    # OpenAI model to use

# Semantic Coherence Settings
ENABLE_COHERENCE_VALIDATION=True  # Enable semantic coherence tracking
COHERENCE_HIGH_THRESHOLD=0.8      # High coherence threshold
COHERENCE_LOW_THRESHOLD=0.4       # Low coherence threshold
BOOST_K_MULTIPLIER=2.0            # K boosting multiplier
```

## 🎯 Semantic Coherence Validation

The system includes advanced semantic coherence tracking that monitors the consistency between queries, retrieved chunks, and generated answers. When coherence drops, automatic fallback behaviors are triggered:

- **K-Boosting**: Automatically increases retrieval count for better context
- **Output Hedging**: Adds uncertainty language when confidence is low  
- **Uncertainty Flagging**: Warns users about potentially unreliable answers

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Django App    │    │   Vector Store   │    │   OpenAI API    │
│   (Web/API)     │───▶│   (FAISS/etc.)   │───▶│   (GPT-4)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Embeddings     │    │   AI Responses  │
│   Processing    │    │   & Search       │    │   with Sources  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components
- **Django App**: Web interface, API, and data management
- **Document Processor**: Extracts and chunks text from files
- **Vector Store**: Handles embeddings and similarity search
- **RAG Engine**: Orchestrates retrieval and generation
- **CLI Tools**: Command-line utilities

## 🐋 Docker Deployment

```bash
# Quick start with Docker
docker-compose up

# Or build manually
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

## 🧪 Testing

```bash
# Run tests
pytest

# Test with coverage
pytest --cov=src --cov=rag_app
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🆘 Troubleshooting

**Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

**Memory issues with large docs**: Reduce `CHUNK_SIZE` in `.env` or process documents individually

**Port conflicts**: Use `python main.py start --port 8001` to use a different port

**Poor answer quality**: Increase `TOP_K_RESULTS` and `CHUNK_OVERLAP` for better context retrieval
