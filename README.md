# Simple RAG System with Elasticsearch

A minimal RAG (Retrieval-Augmented Generation) system using Elasticsearch's ingest attachment and HuggingFace text embedding capabilities.

## Features

- Document ingestion with automatic content extraction using attachment processor
- Text embedding generation using HuggingFace text-embeddings-inference API (CPU version)
- Hybrid search combining BM25 and vector similarity using RRF (Reciprocal Rank Fusion)
- Support for various document formats (PDF, DOC, TXT, etc.)
- Docker-based deployment with Elasticsearch 8.17 and HuggingFace embedding service
- Uses BAAI/bge-small-en-v1.5 model with HF_MIRROR support for China users

## Quick Start with Docker

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (For China users) Set HuggingFace mirror before starting services:
```bash
export HF_MIRROR=https://hf-mirror.com
docker-compose up -d
```

3. Or create a `.env` file in the project root:
```env
HF_MIRROR=https://hf-mirror.com
```

4. Start services:
```bash
docker-compose up -d
```

5. Wait for services to be ready (first time may take longer for model download):
```bash
python main.py wait
```

6. Initialize the system:
```bash
python main.py setup
```

## Model Information

- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Model Size**: ~133MB (smaller and faster to download)
- **Vector Dimensions**: 384
- **Language**: English
- **Performance**: Good balance of speed and quality for general RAG tasks

## Configuration for China Users

For faster model downloading in China, set the HF_MIRROR environment variable:

### Method 1: Environment Variable (Recommended)
```bash
export HF_MIRROR=https://hf-mirror.com
docker-compose up -d
```

### Method 2: .env File
Create a `.env` file in the project root:
```env
HF_MIRROR=https://hf-mirror.com
```
Then run:
```bash
docker-compose up -d
```

### Method 3: Inline Environment Variable
```bash
HF_MIRROR=https://hf-mirror.com docker-compose up -d
```

## Troubleshooting

### Model Download Issues
If the embedding service fails to download the model:

1. Check if you've set the HF_MIRROR before starting:
```bash
echo $HF_MIRROR
```

2. Check the embedding service logs:
```bash
docker-compose logs -f huggingface-embedding
```

3. Stop and remove containers, then restart with mirror:
```bash
docker-compose down
export HF_MIRROR=https://hf-mirror.com
docker-compose up -d
```

4. For persistent issues, clear the cache volume:
```bash
docker-compose down -v
export HF_MIRROR=https://hf-mirror.com
docker-compose up -d
```

### First Time Setup
- The first startup may take 3-5 minutes for model download
- Use `docker-compose logs -f huggingface-embedding` to monitor progress
- The embedding service will be ready once model download completes

## Usage

### Adding Documents
```bash
python main.py add /path/to/document.pdf
```

### Searching
```bash
python main.py search "your search query"
```

### Programmatic Usage
```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.setup()

# Add document
rag.add_document("/path/to/file.pdf")

# Search
results = rag.search("machine learning")

# Get answer with context
answer = rag.get_answer("What is machine learning?")
```

## Docker Services

The system includes:
- **Elasticsearch 8.17**: Document storage and search
- **HuggingFace Text Embeddings Inference (CPU)**: Embedding generation using BAAI/bge-small-en-v1.5

### Service URLs
- Elasticsearch: http://localhost:9200
- HuggingFace Embedding API: http://localhost:8080

### Managing Services
```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Check embedding service logs specifically
docker-compose logs -f huggingface-embedding

# Stop services
docker-compose down

# Clean up (removes volumes and downloaded models)
docker-compose down -v
```

## System Requirements

- Docker and Docker Compose
- At least 4GB RAM for Elasticsearch
- Additional 1.5GB RAM for CPU-based embedding service
- No GPU required (CPU-only deployment)
- Internet connection for initial model download (~133MB)

## Performance Notes

- The bge-small-en-v1.5 model is optimized for both speed and quality
- Model is downloaded once and cached for future use
- Good performance for English text embedding tasks
- First startup includes model download time (3-5 minutes)

This implementation provides a minimal but functional RAG system that leverages modern Elasticsearch capabilities and reliable HuggingFace embeddings with CPU-only requirements.