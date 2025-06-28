# RAG Document Q&A System

A specialized large document (800k+ words) Q&A AI agent using Retrieval-Augmented Generation (RAG). This system can efficiently process, index, and query massive documents to provide accurate, contextual answers.

## 🚀 Features

- **Large Document Support**: Handle documents up to 800k+ words efficiently
- **Multiple Format Support**: PDF, DOCX, TXT, and Markdown files
- **Advanced RAG Pipeline**: Combines retrieval and generation for accurate answers
- **Vector Database Options**: FAISS, ChromaDB, and Pinecone support
- **Conversational Mode**: Maintains context across multiple queries
- **Web Interface**: Beautiful Streamlit UI for easy interaction
- **REST API**: FastAPI-based API for integration
- **CLI Tool**: Command-line interface for batch processing
- **Scalable Architecture**: Modular design for easy extension

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- 8GB+ RAM recommended for large documents
- 2GB+ disk space for vector indexes

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
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
# Edit .env with your API keys and configurations
```

5. **Create necessary directories**:
```bash
mkdir -p documents indexes logs
```

## ⚙️ Configuration

Edit the `.env` file with your settings:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Vector Database
VECTOR_DB_TYPE=faiss  # Options: faiss, chroma, pinecone
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Model Settings
EMBEDDING_MODEL=text-embedding-ada-002
CHAT_MODEL=gpt-4-turbo-preview
TEMPERATURE=0.1
MAX_TOKENS=4000
```

## 🎯 Quick Start

### 1. Web Interface (Recommended)

Start the API server:
```bash
python -m src.api
```

In another terminal, start the Streamlit app:
```bash
streamlit run src/streamlit_app.py
```

Open your browser to `http://localhost:8501`

### 2. Command Line Interface

**Add documents to the index**:
```bash
python -m src.cli add path/to/document1.pdf path/to/document2.docx
```

**Query the documents**:
```bash
python -m src.cli query "What are the main findings in the research?"
```

**Interactive mode**:
```bash
python -m src.cli interactive --conversational
```

### 3. Python API

```python
from src.rag_engine import RAGEngine

# Initialize the RAG engine
rag = RAGEngine(index_name="my_documents")

# Add documents
rag.add_documents([
    "path/to/large_document.pdf",
    "path/to/research_paper.docx"
])

# Query the documents
result = rag.query("What are the key conclusions?")
print(result.answer)

# Access source documents
for doc in result.source_documents:
    print(f"Source: {doc.metadata['filename']}")
    print(f"Content: {doc.page_content[:200]}...")
```

## 📚 Usage Examples

### Processing Large Academic Papers

```python
from src.rag_engine import RAGEngine

# Create specialized index for academic papers
rag = RAGEngine(index_name="academic_papers")

# Add multiple research papers
papers = [
    "papers/machine_learning_survey_2024.pdf",
    "papers/deep_learning_advances.pdf", 
    "papers/nlp_transformers_review.pdf"
]

rag.add_documents(papers)

# Ask research questions
questions = [
    "What are the latest advances in transformer models?",
    "How do different ML approaches compare in performance?",
    "What are the main challenges in current NLP research?"
]

for question in questions:
    result = rag.query(question)
    print(f"Q: {question}")
    print(f"A: {result.answer}\n")
```

### Legal Document Analysis

```python
from src.rag_engine import ConversationalRAG

# Use conversational mode for complex legal queries
legal_rag = ConversationalRAG(index_name="legal_docs")

# Add legal documents
legal_rag.add_documents([
    "contracts/service_agreement_800k_words.pdf",
    "regulations/compliance_manual.docx"
])

# Interactive legal consultation
result1 = legal_rag.conversational_query(
    "What are the termination clauses in the service agreement?"
)

result2 = legal_rag.conversational_query(
    "How do these clauses relate to the compliance requirements?"
)
```

### Technical Documentation Q&A

```bash
# CLI example for technical docs
python -m src.cli add \
    manuals/software_manual_v2.pdf \
    docs/api_documentation.md \
    guides/troubleshooting_guide.docx

# Query with high precision
python -m src.cli query \
    "How do I configure the authentication module?" \
    --top-k 3 \
    --include-sources \
    --include-scores
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector Store   │    │   LLM Engine    │
│   Processor     │───▶│   (FAISS/Chroma) │───▶│   (GPT-4)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Chunks   │    │   Embeddings     │    │   Contextual    │
│   + Metadata    │    │   + Similarity   │    │   Answers       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **Document Processor**: Extracts and chunks text from various formats
- **Vector Store Manager**: Handles embedding storage and similarity search  
- **RAG Engine**: Orchestrates retrieval and generation
- **API Layer**: FastAPI for REST endpoints
- **UI Layer**: Streamlit for web interface
- **CLI**: Command-line tools for batch operations

## 🔧 Advanced Configuration

### Custom Chunking Strategy

```python
from src.document_processor import DocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create custom text splitter for code documents
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    length_function=len
)

processor = DocumentProcessor()
processor.text_splitter = code_splitter
```

### Multiple Vector Stores

```python
# Use different vector stores for different document types
academic_rag = RAGEngine(index_name="academic")  # Uses FAISS
legal_rag = RAGEngine(index_name="legal")        # Uses ChromaDB

# Configure in .env:
# VECTOR_DB_TYPE=faiss  # or chroma, pinecone
```

### Custom Prompts

```python
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""You are a legal expert assistant. Use the provided context to answer legal questions accurately.

Context: {context}

Question: {question}

Please provide a detailed legal analysis with relevant citations from the context.

Answer:""",
    input_variables=["context", "question"]
)

rag.prompt_template = custom_prompt
```

## 📊 Performance Optimization

### For Large Documents (800k+ words)

1. **Increase chunk overlap** for better context:
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

2. **Use hierarchical chunking**:
```python
# Process in sections first, then chunks
processor.hierarchical_chunking = True
```

3. **Optimize vector search**:
```env
TOP_K_RESULTS=10  # Retrieve more candidates
```

4. **Use efficient vector store**:
```env
VECTOR_DB_TYPE=faiss  # Fastest for large datasets
```

### Memory Management

```python
# Process documents in batches
from src.utils import chunk_list

large_doc_list = ["doc1.pdf", "doc2.pdf", ...]
for batch in chunk_list(large_doc_list, batch_size=5):
    rag.add_documents(batch)
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -k "test_document_processor"
pytest tests/ -k "test_rag_engine"

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 API Documentation

Start the API server and visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /upload-documents`: Upload and process documents
- `POST /query`: Query the document index
- `POST /conversational-query`: Query with conversation context
- `GET /index-stats`: Get index statistics
- `DELETE /index`: Clear the index

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "src.api"]
```

### Production Considerations

1. **Use production vector database** (Pinecone, Weaviate)
2. **Implement rate limiting** and authentication
3. **Scale with load balancer** for high traffic
4. **Monitor API performance** and costs
5. **Backup vector indexes** regularly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Q: "Import errors when running the application"**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

**Q: "Out of memory when processing large documents"**
```python
# Reduce chunk size and batch processing
CHUNK_SIZE=500
# Process documents one at a time
```

**Q: "API server not starting"**
```bash
# Check if port is available
lsof -i :8000

# Use different port
python -m src.api --port 8001
```

**Q: "Poor answer quality"**
```python
# Increase context retrieval
TOP_K_RESULTS=10

# Adjust chunk overlap
CHUNK_OVERLAP=400

# Try different models
CHAT_MODEL=gpt-4-turbo-preview
```

## 📞 Support

- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

**Built with ❤️ for efficient large document processing**
