# doc-reader

A Django-based RAG document Q&A system for large text corpora.

This project lets you upload long documents, index them into chunks, retrieve relevant context with vector search, and generate answers grounded in those retrieved sections. It includes a web UI, REST API, CLI, and an experimental semantic coherence layer that checks whether retrieved context and generated answers stay meaningfully aligned.

## What it does

- Ingests PDF, DOCX, TXT, and Markdown documents
- Chunks and indexes large documents for retrieval
- Answers questions over indexed content through:
  - a Django web interface
  - REST API endpoints
  - a CLI
- Supports conversational querying
- Tracks semantic coherence across retrieval and generation
- Includes an experimental Azure-based pipeline alongside the standard local/OpenAI flow

## Why this project exists

This repo was built around a practical long-document retrieval problem: asking useful questions over very large documents, including book-length text. The focus is less on “chat with a PDF” and more on building a system that can handle long inputs, retrieval quality issues, and uncertainty more explicitly.

## Core features

### Document ingestion
- Supports PDF, DOCX, TXT, and Markdown
- Configurable chunk size and chunk overlap
- Designed to handle very large documents

### Query interfaces
- Django web app
- REST API
- Command-line interface
- Conversational mode for follow-up questions

### Retrieval
- FAISS-based vector retrieval
- Configurable top-k retrieval
- Optional local embeddings via `sentence-transformers`
- Optional OpenAI embeddings

### Semantic coherence validation
After retrieval and answer generation, the system compares embeddings across:
- query → retrieved chunks
- retrieved chunks → generated answer
- query → generated answer

Answer support is measured separately: each answer sentence is compared with its
best matching source passage (short overlapping windows avoid truncating long
chunks in local embedding models). Question/chunk similarity remains a retrieval
diagnostic; a generic summary question no longer makes a source-backed answer
automatically "critical".

If answer/source similarity is low, the system can increase retrieval depth once
and regenerate, then rescore the **new** answer against the **new** sources.
Similarity diagnostics are shown separately rather than inserting random
low-confidence language into the answer. Embedding failures are reported as
unavailable, not fabricated zero scores.

These are heuristic text-similarity scores, **not probabilities of factual
correctness or an entailment check**; related but incorrect claims can still score
well. Existing thresholds are unchanged. `confidence_scores` remains the API
field for retrieval relevance scores for compatibility; it is not answer
confidence. `metadata.answer_support_cosine` and
`metadata.coherence_validation_status` expose the separate support diagnostic.

Explicit corpus-wide requests such as "Summarize all documents" or "What are the
main topics covered in the documents?" rank across the active index, prefer
overview/introductory passages, and select sources in balanced rounds. When `k`
is smaller than the number of files, only `k` files can be represented. Focused
definition queries retain relevance-first retrieval. Corpus overviews scan all
indexed candidates, so they cost more than focused lookups on large indexes.

## Architecture

### Standard pipeline
1. Upload or add documents
2. Extract and chunk text
3. Generate embeddings
4. Store chunks in a vector index
5. Retrieve top-k chunks for a question
6. Generate an answer from retrieved context
7. Run semantic coherence validation on the result

### Experimental Azure pipeline
The repo also includes an experimental Azure-native path using:
- Azure OpenAI
- Azure AI Search
- Azure Document Intelligence
- optional Azure Key Vault / Storage integration

This path is still experimental and should be treated as a separate integration track rather than the default setup.

## Tech stack

- Python
- Django + Django REST Framework
- LangChain
- FAISS
- OpenAI or local sentence-transformer embeddings
- Optional Azure OpenAI / AI Search / Document Intelligence

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/djleamen/doc-reader
cd doc-reader
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
For the standard pipeline, set at least:
```
DJANGO_SECRET_KEY=<a generated secret>
OPENAI_API_KEY=your_key_here
```
`DJANGO_SECRET_KEY` has no default — the app raises at startup without it.
Generate one with:
```bash
python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"
```

For local embeddings, enable:
```
USE_LOCAL_EMBEDDINGS=true
```

For the Azure pipeline, fill in the Azure-specific settings from .env.example.

### 5. Start the app
```bash
python main.py start
```
Then open:
```
http://localhost:8000
```

## Usage

### Web UI

Sign in at `/login/`, then upload documents and ask questions through the
browser. Django authentication protects the workspace, Django sessions isolate
multi-turn conversation memory, and persisted query sessions are associated
with the signed-in user. Create the first account with:

```bash
python manage.py createsuperuser
```

### Answer-generation failures

The Classic OpenAI pipeline needs available **API credits**, even when embeddings
run locally. A ChatGPT subscription does not provide API credits. If the app
reports exhausted credits/quota, check billing, project budget, and usage limits
for the project associated with `OPENAI_API_KEY`. If you change the key, restart
the app. Existing documents and indexes do not need to be uploaded again.

Generation failures return HTTP 503 with an actionable `error` and an
`error_code`, rather than being saved or scored as low-confidence answers.
For example, exhausted credits return `openai_quota_exceeded`.

### API

Authentication is required **by default**: every API endpoint (except
`/api/health/` and `/api/azure/health/`) needs an authenticated session. Log in
at `/login/` before making API requests, or use session cookies from an active
browser session.

The health check endpoints are intentionally public so that load balancers and
monitoring tools can reach them without credentials. For unauthenticated callers
they return only a minimal liveness status — index names and Azure
service/configuration details are exposed only to authenticated requests.

The default (`REQUIRE_API_AUTH=True`) is secure out of the box. For local
development you can opt out by setting `REQUIRE_API_AUTH=False` in your `.env`
file. **Never disable authentication in a production deployment.**

```bash
curl -X POST "http://localhost:8000/api/upload-documents/" \
  -F "files=@document.pdf" \
  -F "index_name=default"

curl -X POST "http://localhost:8000/api/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "index_name": "default"}'
```

### CLI
```bash
python main.py cli add document.pdf
python main.py cli query "What are the key findings?"
python main.py cli interactive --conversational
```

## Configuration

Important settings include:

```
VECTOR_DB_TYPE=faiss
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
CHAT_MODEL=gpt-4o

ENABLE_COHERENCE_VALIDATION=True
COHERENCE_HIGH_THRESHOLD=0.8
COHERENCE_LOW_THRESHOLD=0.4
BOOST_K_MULTIPLIER=2.0
```

## Project status

This is a working RAG application with multiple interfaces and an experimental retrieval-quality layer. The standard pipeline is the main path. The Azure pipeline is included as an experimental integration.

## License

MIT
