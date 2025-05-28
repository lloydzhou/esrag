import os

# Elasticsearch configuration
ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = int(os.getenv('ES_PORT', 9200))
ES_USERNAME = os.getenv('ES_USERNAME', 'elastic')
ES_PASSWORD = os.getenv('ES_PASSWORD', 'changeme')

# HuggingFace Mirror for China users
HF_MIRROR = os.getenv('HF_MIRROR', 'https://hf-mirror.com')

# Index configuration
INDEX_NAME = 'knowledge_base'
PIPELINE_NAME = 'attachment_embedding_pipeline'

# Text embedding inference service - using a smaller, more reliable model
INFERENCE_SERVICE_ID = 'text_embedding_service1'
INFERENCE_SERVICE_URL = 'http://192.168.10.12:8080/embed'
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Smaller, more reliable model

# Vector dimensions (bge-small-en-v1.5 produces 384-dimensional vectors)
VECTOR_DIMS = 384

# Chunk settings for large documents
MAX_CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50    # tokens
