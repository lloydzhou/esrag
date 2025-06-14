"""
ElasticRAG - Elasticsearch-based RAG system with ingest pipeline processing.
"""

from .client import Client
from .collection import Collection
from .model import Model
from .user import User
from .splitter import Splitter, JinaTextSegmenter

__version__ = "0.1.0"
__all__ = ["Client", "Collection", "Model", "User", "Splitter", "JinaTextSegmenter"]
