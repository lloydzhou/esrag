# RAG SDK - Retrieval-Augmented Generation System

A powerful Python SDK for building RAG (Retrieval-Augmented Generation) systems using Elasticsearch with advanced features like multi-model support, user authentication, and hybrid search.

## Features

- **Multi-Model Support**: Supports multiple embedding models (OpenAI, Hugging Face, etc.)
- **User Authentication**: Built-in user management with API key authentication
- **Hybrid Search**: Combines text search and vector search with RRF (Reciprocal Rank Fusion)
- **Document Processing**: Automatic text chunking and embedding generation
- **Template-Based Architecture**: Pre-defined index templates for efficient scaling
- **Collection Management**: Organize documents into collections with model-specific configurations

## Core Concepts

### Client
The main entry point for interacting with the RAG system. Manages Elasticsearch connections, user authentication, and model registration.

### User
Handles user authentication and authorization. Each user can have multiple collections and maintain their own document spaces.

### Model
Represents an embedding model configuration. Each model creates its own inference service, processing pipeline, and index template for optimal performance.

### Collection
A knowledge base that belongs to a user and uses a specific model. Collections store and organize documents with automatic embedding generation.

## Quick Start

### 1. Installation

```bash
pip install elasticsearch asyncio
```

### 2. Initialize Client

```python
from rag import Client

# Connect to Elasticsearch
client = Client('http://localhost:9200')
```

### 3. Setup User and Model

```python
# Add a user
client.add_user('john_doe', 'secure_api_key', metadata={
    "email": "john@example.com",
    "role": "user"
})

# Authenticate user
user = client.authenticate('john_doe', 'secure_api_key')

# Register a model
model_config = {
    "service": "hugging_face",
    "service_settings": {
        "api_key": "your_hf_token",
        "url": "http://your-embedding-service:8080/embed",
    },
    "dimensions": 384
}
client.register_model("bge-small-en-v1.5", model_config)
```

### 4. Create Collection and Add Documents

```python
# Get collection with specific model
collection = client.get_collection("my_documents", model_id="bge-small-en-v1.5")

# Add document from file
with open("document.pdf", "rb") as f:
    collection.add(
        document_id="doc1",
        name="Important Document",
        file_content=f.read(),
        metadata={"category": "research", "author": "John Doe"}
    )

# Add document from text
collection.add(
    document_id="doc2", 
    name="Text Document",
    text_content="This is important content...",
    metadata={"category": "notes"}
)
```

### 5. Search Documents

```python
import asyncio

async def search_example():
    # Perform hybrid search (text + vector)
    results = await collection.query(
        query_text="important information",
        metadata_filter={"category": "research"},
        size=5
    )
    
    for result in results:
        print(f"Document: {result['document_name']}")
        print(f"Content: {result['chunk_content'][:200]}...")
        print(f"Score: {result['final_score']:.4f}")
        print("-" * 50)

asyncio.run(search_example())
```

## API Reference

### Client

#### Methods

- `add_user(username, api_key, metadata=None)` - Add or update user
- `authenticate(username, api_key)` - Authenticate user
- `register_model(model_id, config)` - Register embedding model
- `get_model(model_id, service_config=None)` - Get or create model
- `get_collection(name, model_id=None)` - Get or create collection
- `list_models()` - List available models
- `list_collections()` - List user's collections

### User

#### Methods

- `validate()` - Validate user credentials
- `get_info()` - Get user information
- `update_metadata(metadata)` - Update user metadata
- `delete()` - Delete user

### Model

#### Properties

- `model_id` - Model identifier
- `config` - Model configuration
- `inference_id` - Elasticsearch inference service ID
- `pipeline_id` - Processing pipeline ID
- `template_name` - Index template name

#### Methods

- `get_dimensions()` - Get embedding vector dimensions

### Collection

#### Methods

- `add(document_id, name, file_content=None, text_content=None, metadata=None)` - Add document
- `query(query_text, metadata_filter=None, size=5, include_embedding=True)` - Search documents
- `get(document_id)` - Get specific document
- `delete(document_id)` - Delete document
- `list_documents(offset=0, limit=10)` - List documents
- `drop()` - Delete entire collection

## Advanced Usage

### Custom Model Configuration

```python
# Register OpenAI model
openai_config = {
    "service": "openai",
    "service_settings": {
        "api_key": "your_openai_key",
        "model_id": "text-embedding-ada-002",
    },
    "dimensions": 1536
}
client.register_model("text-embedding-ada-002", openai_config)
```

### Multiple Collections with Different Models

```python
# Create collections with different models
collection_bge = client.get_collection("documents_bge", model_id="bge-small-en-v1.5")
collection_openai = client.get_collection("documents_openai", model_id="text-embedding-ada-002")
```

### Metadata Filtering

```python
# Search with metadata filters
results = await collection.query(
    query_text="machine learning",
    metadata_filter={
        "category": ["research", "tutorial"],
        "author": "Jane Smith"
    }
)
```

## Command Line Interface

The SDK includes a CLI for common operations:

```bash
# Setup system
python rag.py setup

# List models
python rag.py list_models

# List collections
python rag.py list_collections

# Add document
python rag.py add document.pdf my_collection bge-small-en-v1.5

# Search documents
python rag.py search "important query" my_collection bge-small-en-v1.5
```

## Configuration

### Environment Variables

- `HF_API_KEY` - Hugging Face API key
- `BGE_MODEL_URL` - BGE model service URL
- `OPENAI_API_KEY` - OpenAI API key

### Index Naming Convention

- Collections with model: `{model_id}__{username}__{collection_name}`
- Collections without model: `{username}__{collection_name}`

## Requirements

- Python 3.7+
- Elasticsearch 8.0+
- elasticsearch-py
- asyncio

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support, please open an issue on GitHub.
