import asyncio
import logging
import base64
import os
from functools import partial
from typing import Dict, List, Optional, Union, Any
from elasticsearch import Elasticsearch, AsyncElasticsearch, NotFoundError


class Client:
    """Main client for the RAG system, managing ES connections and global configurations"""
    
    def __init__(self, hosts: Union[str, List[str]], force_recreate=None, **kwargs):
        """
        Initialize the client
        
        Args:
            hosts: Elasticsearch host addresses
            **kwargs: Other ES connection parameters
        """
        self.hosts = hosts if isinstance(hosts, list) else [hosts]
        self.client = Elasticsearch(self.hosts, **kwargs).options(ignore_status=404)
        self.async_client = AsyncElasticsearch(self.hosts, **kwargs)
        self._collections = {}
        self._predefined_models = {}
        self._user = None
        self.splitter = None
        self.force_recreate = force_recreate if force_recreate is not None else os.getenv("FORCE_RECREATE", "false").lower() == "true"
        self._init_spliter()
        self._load_existing_models()

    def add_user(self, username: str, api_key: str, metadata: Optional[Dict] = None) -> bool:
        """Add or update user credentials"""
        user = User(self, username, api_key)
        return user.create_or_update(metadata)

    def delete_user(self, username: str) -> bool:
        """Delete user credentials"""
        user = User(self, username, "")
        return user.delete()
        
    def authenticate(self, username: str, api_key: str) -> 'User':
        """Authenticate user"""
        user = User(self, username, api_key)
        if user.validate():
            self._user = user
            return self._user
        else:
            raise ValueError(f"User authentication failed: {username}")
    
    def register_model(self, model_id: str, config: Dict) -> 'Model':
        """Register a predefined model"""
        model = Model(
            client=self,
            model_id=model_id,
            config=config
        )
        self._predefined_models[model_id] = model
        return model

    def get_model(self, model_id: str) -> 'Model':
        if model_id not in self._predefined_models:
            raise ValueError(f"Model {model_id} is not predefined")
        return self._predefined_models[model_id]

    def list_models(self) -> List[Dict]:
        """List available models"""
        return [
            {
                "model_id": model_id,
                "config": model.config,
                "dimensions": model.get_dimensions()
            }
            for model_id, model in self._predefined_models.items()
        ]

    def get_collection(self, name: str, model_id: Optional[str] = None) -> 'Collection':
        """Get or create a collection (knowledge base)"""
        if not self._user:
            raise ValueError("Please call authenticate() to authenticate the user first")
        
        # If model_id is specified, use that model; otherwise, use the default model
        if model_id:
            model = self.get_model(model_id)
            collection_key = f"{model_id}__{self._user.username}__{name}"
        else:
            model = None
            collection_key = f"{self._user.username}__{name}"
            
        if collection_key not in self._collections:
            self._collections[collection_key] = Collection(
                client=self,
                name=name,
                user=self._user,
                model=model
            )
        return self._collections[collection_key]
    
    def list_collections(self) -> List[str]:
        """List all collections of the user"""
        if not self._user:
            raise ValueError("Please call authenticate() to authenticate the user first")
        
        try:
            pattern = f"*__{self._user.username}__*"
            response = self.client.cat.indices(index=pattern, format='json', ignore=[404])
            logging.debug(f"Listing collections: {response}")
            if response:
                # Support new index naming format: {model_id}__{username}__{collection_name}
                prefix = f"{self._user.username}__"
                collections = []
                for idx in response:
                    if prefix in idx['index']:
                        # Parse index name: {model_id}__{username}__{collection_name} or {username}__{collection_name}
                        index_parts = idx['index'].replace(prefix, "").split("__")
                        if len(index_parts) == 2:
                            model_id, collection_name = index_parts
                        else:
                            model_id = "default"
                            collection_name = "__".join(index_parts)
                        
                        collections.append({
                            "name": collection_name,
                            "model_id": model_id,
                            "index": idx['index'],
                            "health": idx.get('health', 'unknown'),
                            "status": idx.get('status', 'unknown'),
                            "doc_count": idx.get('docs.count', '0'),
                            "store_size": idx.get('store.size', '0b')
                        })
                return collections
            return []
        except Exception:
            return []

    def _load_existing_models(self):
        """Load all existing model inference service configurations from ES"""
        try:
            # Get all inference services
            response = self.client.inference.get()
            for config in response.get('endpoints', {}):
                inference_id = config.get('inference_id', '')
                if inference_id.endswith('__inference'):
                    model_id = inference_id.replace('__inference', '')
                    # If not in predefined models, rebuild from configuration
                    if model_id not in self._predefined_models:
                        service_config = {
                            "service": config.get('service', 'openai'),
                            "service_settings": config.get('service_settings', {}),
                            "dimensions": config.get('service_settings', {}).get('dimensions', 384)
                        }
                        model = Model(
                            client=self,
                            model_id=model_id,
                            config=service_config
                        )
                        # Mark as existing to avoid repeated initialization
                        model._exists = True
                        self._predefined_models[model_id] = model
            logging.debug(f"Loaded {len(self._predefined_models)} models")
        except Exception as e:
            logging.warning(f"Failed to load existing models: {e}")

    def _init_spliter(self):
        self.splitter = Splitter()
        self.splitter.init_script(self.client, force_recreate=self.force_recreate)


class Splitter:
    """Default text splitter with configurable chunk size and overlap"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 16):
        self.splitter_id = "text_splitter"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def get_script_source(self) -> str:
        """Return the Elasticsearch Painless script source code for text splitting (not Java)"""
        return """
        // Painless script implementing text splitting with smart boundary detection
        if (ctx.attachment?.content != null) {
            String content = ctx.attachment.content;
            Map config = params.splitter_config;
            int chunkSize = config.chunk_size;
            int overlap = config.chunk_overlap;
            List chunks = new ArrayList();

            String text = content.trim();
            int textLength = text.length();
            
            if (textLength <= chunkSize) {
                // Text fits in one chunk
                Map data = new HashMap();
                data.put("content", text);
                Map meta = new HashMap();
                meta.put("index", 0);
                meta.put("offset", 0);
                meta.put("length", textLength);
                data.put("metadata", meta);
                chunks.add(data);
            } else {
                // Split into chunks with smart boundary detection
                int start = 0;
                int chunkIndex = 0;
                
                while (start < textLength) {
                    int end = (int)Math.min(start + chunkSize, textLength);
                    
                    // If we're not at the end of text, try to find a better split point
                    if (end < textLength) {
                        int bestEnd = end;
                        
                        // Look for sentence boundaries within last 30% of chunk
                        int searchStart = start + (int)(chunkSize * 0.7);
                        for (int i = end - 1; i >= searchStart; i--) {
                            char c = text.charAt(i);
                            if (c == 46 || c == 33 || c == 63) { // '.', '!', '?'
                                // Check if followed by space and capital letter (sentence boundary)
                                if (i + 1 < textLength && text.charAt(i + 1) == 32 && 
                                    i + 2 < textLength && Character.isUpperCase(text.charAt(i + 2))) {
                                    bestEnd = i + 1;
                                    break;
                                }
                            }
                        }
                        
                        // If no sentence boundary found, look for word boundaries (spaces)
                        if (bestEnd == end) {
                            for (int i = end - 1; i >= searchStart; i--) {
                                if (text.charAt(i) == 32) { // space
                                    bestEnd = i + 1;
                                    break;
                                }
                            }
                        }
                        
                        end = bestEnd;
                    }
                    
                    String chunk = text.substring(start, end).trim();
                    
                    if (chunk.length() > 0) {
                        Map data = new HashMap();
                        data.put("content", chunk);
                        Map meta = new HashMap();
                        meta.put("index", chunkIndex);
                        meta.put("offset", start);
                        meta.put("length", chunk.length());
                        data.put("metadata", meta);
                        chunks.add(data);
                        chunkIndex++;
                    }
                    
                    // Calculate next start position with overlap
                    if (end >= textLength) {
                        break; // We've reached the end
                    }
                    
                    // Calculate overlap start position
                    int overlapStart = end - overlap;
                    if (overlapStart <= start) {
                        // If overlap would not advance, just move forward minimally
                        start = start + (chunkSize / 2);
                    } else {
                        // Try to find a good overlap boundary
                        int nextStart = overlapStart;
                        
                        // Look for word boundary within overlap region
                        for (int i = overlapStart; i < end && i >= overlapStart - 5; i++) {
                            if (text.charAt(i) == 32) { // space
                                nextStart = i + 1;
                                break;
                            }
                        }
                        
                        start = nextStart;
                    }
                    
                    // Safety check to prevent infinite loop
                    if (start >= textLength) {
                        break;
                    }
                }
            }
            
            ctx.chunks = chunks;
        }
        """
    
    def get_processor(self) -> Dict:
        """Return the processor configuration for the pipeline"""
        return {
            "script": {
                "id": self.splitter_id,
                "params": {
                    "splitter_config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    }
                }
            }
        }

    def init_script(self, es_client: Elasticsearch, force_recreate: bool = False):
        """Initialize the script in Elasticsearch"""
        if es_client.get_script(id=self.splitter_id):
            logging.debug(f"Splitter script already exists: {self.splitter_id}")
            if not force_recreate:
                return
        try:
            es_client.put_script(
                id=self.splitter_id,
                body={
                    "script": {
                        "lang": "painless",
                        "source": self.get_script_source(),
                    }
                }
            )
            logging.debug(f"Splitter script initialized successfully: {self.splitter_id}")
        except Exception as e:
            logging.warning(f"Splitter script initialization failed: {e}")


class User:
    """User management"""
    
    def __init__(self, client: Client, username: str, api_key: str, auth_index: str = "user_auth"):
        self.client = client
        self.username = username
        self.api_key = api_key
        self.auth_index = auth_index
    
    @staticmethod
    def init_auth_index(es_client: Elasticsearch, auth_index: str):
        """Initialize user authentication index"""
        if es_client.indices.exists(index=auth_index):
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "username": {
                        "type": "keyword"
                    },
                    "api_key": {
                        "type": "keyword"
                    },
                    "created_at": {
                        "type": "date"
                    },
                    "last_login": {
                        "type": "date"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": False
                    }
                }
            },
            "settings": {
                "index": {
                    "number_of_replicas": 0,
                }
            }
        }
        
        try:
            es_client.indices.create(index=auth_index, body=mapping)
            logging.debug(f"Created user authentication index successfully: {auth_index}")
        except Exception as e:
            logging.error(f"Failed to create user authentication index: {e}")
            raise
    
    def create_or_update(self, metadata: Optional[Dict] = None) -> bool:
        """Create or update user credentials"""
        try:
            User.init_auth_index(self.client.client, self.auth_index)
            from datetime import datetime
            doc_data = {
                "username": self.username,
                "api_key": self.api_key,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            response = self.client.client.index(
                index=self.auth_index,
                id=self.username,  # Use username as document ID
                body=doc_data,
                refresh='wait_for'
            )
            logging.debug(f"User {self.username} added successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to add user: {e}")
            return False
    
    def validate(self) -> bool:
        """Validate user credentials"""
        try:
            response = self.client.client.get(
                index=self.auth_index,
                id=self.username
            )
            stored_api_key = response['_source'].get('api_key')
            
            if stored_api_key == self.api_key:
                # Update last login time
                self._update_last_login()
                return True
            else:
                logging.warning(f"User {self.username} API key validation failed")
                return False
                
        except NotFoundError:
            logging.warning(f"User {self.username} does not exist")
            return False
        except Exception as e:
            logging.error(f"Failed to validate user: {e}")
            return False
    
    def _update_last_login(self):
        """Update last login time"""
        try:
            from datetime import datetime
            self.client.client.update(
                index=self.auth_index,
                id=self.username,
                body={
                    "doc": {
                        "last_login": datetime.now().isoformat()
                    }
                },
                refresh='wait_for'
            )
        except Exception as e:
            logging.warning(f"Failed to update login time: {e}")
    
    def delete(self) -> bool:
        """Delete user"""
        try:
            self.client.client.delete(
                index=self.auth_index,
                id=self.username,
                refresh='wait_for'
            )
            logging.debug(f"User {self.username} deleted successfully")
            return True
        except NotFoundError:
            logging.warning(f"User {self.username} does not exist")
            return False
        except Exception as e:
            logging.error(f"Failed to delete user: {e}")
            return False
    
    def get_info(self) -> Optional[Dict]:
        """Get user information"""
        try:
            response = self.client.client.get(
                index=self.auth_index,
                id=self.username
            )
            user_info = response['_source'].copy()
            # Do not return API key
            user_info.pop('api_key', None)
            return user_info
        except NotFoundError:
            return None
        except Exception as e:
            logging.error(f"Failed to get user information: {e}")
            return None
    
    def update_metadata(self, metadata: Dict) -> bool:
        """Update user metadata"""
        try:
            self.client.client.update(
                index=self.auth_index,
                id=self.username,
                body={
                    "doc": {
                        "metadata": metadata
                    }
                },
                refresh='wait_for'
            )
            logging.debug(f"User {self.username} metadata updated successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to update user metadata: {e}")
            return False

    @classmethod
    def list_all_users(cls, es_client: Elasticsearch, auth_index: str, 
                      offset: int = 0, limit: int = 10) -> Dict:
        """List all users (class method for admin functionality)"""
        try:
            response = es_client.search(
                index=auth_index,
                body={
                    "query": {"match_all": {}},
                    "_source": ["username", "created_at", "last_login", "metadata"],
                    "from": offset,
                    "size": limit,
                    "sort": [{"created_at": {"order": "desc"}}]
                }
            )
            
            return {
                "total": response['hits']['total']['value'],
                "users": [
                    {
                        "username": hit['_source'].get('username', ''),
                        "created_at": hit['_source'].get('created_at', ''),
                        "last_login": hit['_source'].get('last_login', ''),
                        "metadata": hit['_source'].get('metadata', {})
                    }
                    for hit in response['hits']['hits']
                ]
            }
        except Exception as e:
            logging.error(f"Failed to list users: {e}")
            return {"total": 0, "users": []}


class Model:
    """Model management, responsible for inference service, pipeline, and index template"""
    
    def __init__(self, client: Client, model_id: str, config: Dict):
        self.client = client
        self.model_id = model_id
        self.config = config
        self.inference_id = f"{model_id}__inference"
        self.pipeline_id = f"{model_id}__pipeline"
        self.template_name = f"{model_id}__template"
        self._exists = False
        
        # Initialize the three components of the model
        self._init_inference()
        self._create_model_pipeline()
        self._create_index_template()
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vector"""
        return self.config.get("dimensions") or self.config.get("service_settings", {}).get("dimensions", 384)
    
    def _init_inference(self):
        """Initialize inference service"""
        if self._exists or 'rate_limit' in self.config.get("service_settings", {}):
            return
        if self.client.client.inference.get(inference_id=self.inference_id):
            logging.debug(f'Inference service already exists: {self.inference_id}')
            if not self.client.force_recreate:
                return

        try:
            self.client.client.inference.delete(inference_id=self.inference_id, force=True)
            response = self.client.client.inference.put(
                task_type="text_embedding",
                inference_id=self.inference_id,
                body={
                    "service": self.config.get("service", "openai"),
                    "service_settings": self.config.get("service_settings", {})
                }
            )
            logging.debug(f'Created inference service successfully: {self.inference_id}')
        except Exception as e:
            logging.error(f"Failed to create inference service: {e}")
            raise

    def _create_model_pipeline(self):
        """Create a processing pipeline dedicated to the model"""
        if self.client.client.ingest.get_pipeline(id=self.pipeline_id):
            logging.debug(f'Pipeline already exists: {self.pipeline_id}')
            if not self.client.force_recreate:
                return
            
        processors = [
            {
                "attachment": {
                    "field": "data",
                    "target_field": "attachment",
                    "properties": ["content", "title", "content_type"],
                    "remove_binary": True,
                    "ignore_missing": True
                }
            },
            self.client.splitter.get_processor(),
            {
                "foreach": {
                    "field": "chunks",
                    "processor": {
                        "inference": {
                            "model_id": self.inference_id,
                            "input_output": {
                                "input_field": "_ingest._value.content",
                                "output_field": "_ingest._value.embedding"
                            }
                        }
                    },
                    "ignore_missing": True
                }
            },
            {
                "foreach": {
                    "field": "chunks",
                    "processor": {
                        "remove": {
                            "field": "_ingest._value.model_id",
                            "ignore_missing": True
                        }
                    },
                    "ignore_missing": True
                }
            }
        ]
        
        try:
            response = self.client.client.ingest.put_pipeline(
                id=self.pipeline_id,
                body={
                    "description": f"Processing pipeline for model {self.model_id}",
                    "processors": processors
                }
            )
            logging.debug(f'Created model Pipeline successfully: {self.pipeline_id}')
        except Exception as e:
            logging.error(f"Failed to create model Pipeline: {e}")
            raise

    def _create_index_template(self):
        """Create an index template dedicated to the model"""
        if self.client.client.indices.exists_index_template(name=self.template_name):
            logging.debug(f'Index template already exists: {self.template_name}')
            if not self.client.force_recreate:
                return

        dimensions = self.get_dimensions()
        
        template = {
            "index_patterns": [f"{self.model_id}__*"],
            "template": {
                "mappings": {
                    "properties": {
                        "name": {
                            "type": "text",
                            "analyzer": "ik_max_word"
                        },
                        "chunks": {
                            "type": "nested",
                            "properties": {
                                "content": {
                                    "type": "text",
                                    "analyzer": "ik_max_word"
                                },
                                "metadata": {
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "offset": {"type": "integer"},
                                        "length": {"type": "integer"}
                                    }
                                },
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": dimensions,
                                    "index": True,
                                    "similarity": "dot_product"
                                }
                            }
                        },
                        "metadata": {
                            "properties": {
                                "enable": {"type": "integer"},
                                "source": {"type": "keyword"},
                                "category": {"type": "keyword"},
                                "path": {"type": "keyword"}
                            }
                        },
                        "attachment": {
                            "properties": {
                                "content": {"type": "text", "analyzer": "ik_max_word"},
                                "title": {"type": "text", "analyzer": "ik_max_word"},
                                "content_type": {"type": "keyword"}
                            }
                        },
                    }
                },
                "settings": {
                    "index": {
                        "default_pipeline": self.pipeline_id,
                        "number_of_replicas": 0,
                    }
                }
            }
        }
        
        try:
            self.client.client.indices.put_index_template(
                name=self.template_name,
                body=template
            )
            logging.debug(f'Created index template successfully: {self.template_name}')
        except Exception as e:
            logging.error(f"Failed to create index template: {e}")
            raise


class Collection:
    """Collection (knowledge base) abstraction, corresponding to an ES index"""
    
    def __init__(self, client: Client, name: str, user: User, model: Optional[Model] = None):
        self.client = client
        self.name = name
        self.user = user
        self.model = model
        
        # Index naming convention
        if model:
            self.index_name = f"{model.model_id}__{user.username}__{name}"
        else:
            self.index_name = f"{user.username}__{name}"
    
    def add(self, document_id: str, name: str, file_content: Optional[bytes] = None,
            text_content: Optional[str] = None, metadata: Optional[Dict] = None,
            chunks: Optional[List[Dict]] = None,
            timeout: int = 600) -> Dict:
        """
        Add a document to the collection
        
        Args:
            document_id: Document ID
            name: Document name
            file_content: File content (binary)
            text_content: Text content
            metadata: Metadata
            chunks: Pre-processed chunks of text with embeddings (optional)
            timeout: Timeout
        """
        if not file_content and not text_content and not chunks:
            raise ValueError("Must provide file_content or text_content or chunks")
        
        doc_data = {
            "name": name,
            "metadata": metadata or {},
        }
        
        if file_content:
            # Process file content
            doc_data["data"] = base64.b64encode(file_content).decode()
        elif text_content:
            # Process text content
            doc_data["attachment"] = {"content": text_content}
        elif chunks:
            doc_data["chunks"] = chunks
        
        try:
            response = self.client.client.index(
                index=self.index_name,
                id=document_id,
                body=doc_data,
                timeout=f"{timeout}s",
                refresh='wait_for'  # Ensure the document is immediately visible
            )
            return response
        except Exception as e:
            logging.error(f"Failed to add document: {e}")
            raise
    
    async def query(self, query_text: str, metadata_filter: Optional[Dict] = None, 
                   size: int = 5, include_embedding: bool = True) -> List[Dict]:
        """
        Query the collection for relevant documents
        
        Args:
            query_text: Query text
            metadata_filter: Metadata filter conditions
            size: Number of results to return
            include_embedding: Whether to include vector search
        """
        filter_conditions = []
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    filter_conditions.append({
                        "terms": {f"metadata.{key}": value}
                    })
                else:
                    filter_conditions.append({
                        "term": {f"metadata.{key}": value}
                    })
        
        # Text search
        text_search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "nested": {
                                "path": "chunks",
                                "query": {
                                    "match": {
                                        "chunks.content": query_text
                                    }
                                },
                                "inner_hits": {
                                    "_source": ["chunks.content", "chunks.metadata"],
                                    "size": size
                                }
                            }
                        }
                    ],
                    "filter": filter_conditions
                }
            },
            "size": size * 2,  # Get more results for RRF merging
            "_source": ["name", "metadata"],
        }
        
        # Execute search tasks
        searches = []
        search_results = []
        
        # Text search task
        text_search = partial(self.client.async_client.search, index=self.index_name, body=text_search_body)
        
        searches.append(text_search())
        
        # Vector search (if a model is configured)
        if include_embedding and self.model:
            vector_search_body = {
                "knn": {
                    "filter": filter_conditions,
                    "field": "chunks.embedding",
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": self.model.inference_id,
                            "model_text": query_text,
                        }
                    },
                    "k": size * 2,  # Get more results for RRF merging
                    "num_candidates": size * 10,
                    "inner_hits": {
                        "_source": ["chunks.content", "chunks.metadata"],
                        "size": size,
                    }
                },
                "size": size * 2,
                "_source": ["name", "metadata"],
            }
            
            # Create vector search task
            vector_search = partial(self.client.async_client.search, index=self.index_name, body=vector_search_body)
            
            searches.append(vector_search())
        
        # Execute all searches
        responses = await asyncio.gather(*searches, return_exceptions=True)
        
        # Process results and prepare for RRF merging
        search_results = []
        all_chunk_data = {}  # Store detailed information for all chunks
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logging.warning(f"Search execution failed: {response}")
                continue
            
            search_type = "text" if i == 0 else "vector"
            chunk_results = []
            
            for doc in response['hits']['hits']:
                if 'inner_hits' in doc and 'chunks' in doc['inner_hits']:
                    for chunk in doc['inner_hits']['chunks']['hits']['hits']:
                        chunk_key = f"{doc['_id']}_{chunk['_nested']['offset']}"
                        chunk_results.append((chunk_key, chunk['_score']))
                        
                        # Store detailed information for the chunk
                        if chunk_key not in all_chunk_data:
                            all_chunk_data[chunk_key] = {
                                'document_id': doc['_id'],
                                'document_name': doc['_source'].get('name', ''),
                                'chunk_content': chunk['_source'].get('content', ''),
                                'chunk_metadata': chunk['_source'].get('metadata', {}),
                                'score': chunk['_score'],
                                'document_metadata': doc['_source'].get('metadata', {}),
                                'search_type': search_type
                            }
            
            search_results.append(chunk_results)
        
        # If there is only one search result, return it directly
        if len(search_results) == 1:
            merged_results = search_results[0]
        elif len(search_results) > 1:
            # Merge results using the RRF algorithm
            merged_results = rrf(*search_results, k=60)
        else:
            merged_results = []
        
        # Build the final results
        final_results = []
        for chunk_key, rrf_score in merged_results[:size]:
            if chunk_key in all_chunk_data:
                result = all_chunk_data[chunk_key].copy()
                result['rrf_score'] = rrf_score
                # If there is an RRF score, use it as the main score
                result['final_score'] = rrf_score
                final_results.append(result)
        
        return final_results
    
    def get(self, document_id: str) -> Optional[Dict]:
        """Get the specified document"""
        try:
            response = self.client.client.get(
                index=self.index_name,
                id=document_id
            )
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            logging.error(f"Failed to get document: {e}")
            raise
    
    def delete(self, document_id: str) -> bool:
        """Delete the specified document"""
        try:
            self.client.client.delete(
                index=self.index_name,
                id=document_id,
                refresh='wait_for'
            )
            return True
        except NotFoundError:
            return False
        except Exception as e:
            logging.error(f"Failed to delete document: {e}")
            raise
    
    def list_documents(self, offset: int = 0, limit: int = 10) -> Dict:
        """List documents in the collection"""
        try:
            response = self.client.client.search(
                index=self.index_name,
                body={
                    "query": {"match_all": {}},
                    "_source": ["name", "metadata"],
                    "from": offset,
                    "size": limit
                }
            )            
            return {
                "total": response['hits']['total']['value'],
                "documents": [
                    {
                        "id": hit['_id'],
                        "name": hit['_source'].get('name', ''),
                        "metadata": hit['_source'].get('metadata', {})
                    }
                    for hit in response['hits']['hits']
                ]
            }
        except NotFoundError:
            return {"total": 0, "documents": []}
        except Exception as e:
            logging.error(f"Failed to list documents: {e}")
            raise
    
    def drop(self):
        """Delete the entire collection"""
        try:
            if self.client.client.indices.exists(index=self.index_name):
                self.client.client.indices.delete(index=self.index_name)
                logging.debug(f"Deleted index successfully: {self.index_name}")
            
            try:
                self.client.client.ingest.delete_pipeline(id=self.pipeline_id)
                logging.debug(f"Deleted Pipeline successfully: {self.pipeline_id}")
            except NotFoundError:
                pass
        except Exception as e:
            logging.error(f"Failed to delete collection: {e}")
            raise


# RRF algorithm implementation
def rrf(*queries, k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion algorithm"""
    ranks = [{d[0]: i + 1 for i, d in enumerate(q)} for q in queries]
    result = {}
    for rank in ranks:
        for d in rank.keys():
            result[d] = (result[d] if d in result else 0) + 1.0 / (k + rank[d])
    return sorted(result.items(), key=lambda kv: kv[1], reverse=True)


# Command line argument processing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    import os
    import asyncio

    def usage():
        print("Usage:")
        print("  python rag.py setup")
        print("  python rag.py list_users")
        print("  python rag.py list_models")
        print("  python rag.py list_collections")
        print("  python rag.py list_documents [collection_name] [model_id]")
        print("  python rag.py add <file_path> [collection_name] [model_id]")
        print("  python rag.py search <query> [collection_name] [model_id]")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        usage()
    
    command = sys.argv[1]
    
    async def main():
        # Create client
        client = Client('http://0.0.0.0:9200')
        
        collection_name = "test_documents123"  # Default collection name
        model_id = "bge-small-en-v1.5"  # Default model
        
        if command == "setup":
            # Initialize user
            success = client.add_user('test_user', 'test_api_key', metadata={
                "email": "test@test.com",
                "role": "admin",
                "preferences": {
                    "language": "zh",
                    "theme": "dark"
                }
            })
            if success:
                print("User initialized successfully")
            else:
                print("User initialization failed")
            # BGE model configuration
            config = {
                "service": "hugging_face",
                "service_settings": {
                    "api_key": os.getenv("TEXT_EMBEDDING_API_KEY", "placeholder"),
                    "url": os.getenv("TEXT_EMBEDDING_URL", "http://192.168.9.62:8080/embed"),
                },
                "dimensions": 384
            }
            client.register_model(model_id, config)
            print("Model registered successfully")
        elif command == "list_models":
            # List available models
            try:
                models = client.list_models()
                print(f"Available models (total: {len(models)}):")
                print("=" * 50)
                for model in models:
                    print(f"Model ID: {model['model_id']}")
                    print(f"Service Type: {model['config'].get('service', 'unknown')}")
                    print(f"Vector Dimensions: {model['dimensions']}")
                    print("-" * 30)
            except Exception as e:
                print(f"Failed to list models: {e}")
        elif command == "list_users":
            # List all users
            try:
                users_info = User.list_all_users(client.client, "user_auth")
                print(f"User list (total: {users_info['total']}):")
                print("=" * 50)
                for user in users_info['users']:
                    print(f"Username: {user['username']}")
                    print(f"Created At: {user['created_at']}")
                    print(f"Last Login: {user['last_login'] or 'Never'}")
                    if user['metadata']:
                        print(f"Metadata: {user['metadata']}")
                        print("-" * 30)
            except Exception as e:
                print(f"Failed to list users: {e}")
        elif command == "list_collections":
            # List all collections for the user
            try:
                # User authentication
                user = client.authenticate('test_user', 'test_api_key')
                collections = client.list_collections()
                print(f"Collections for user {user.username} (total: {len(collections)}):")
                print("=" * 50)
                for collection in collections:
                    print(f"Collection Name: {collection['name']}")
                    print(f"Model ID: {collection['model_id']}")
                    print(f"Index Name: {collection['index']}")
                    print(f"Health: {collection['health']}")
                    print(f"Status: {collection['status']}")
                    print(f"Document Count: {collection['doc_count']}")
                    print(f"Storage Size: {collection['store_size']}")
                    print("-" * 30)
            except Exception as e:
                print(f"Failed to list collections: {e}")
        elif command == "list_documents":
            # List documents in the specified collection
            if len(sys.argv) >= 3:
                collection_name = sys.argv[2]
            if len(sys.argv) >= 4:
                model_id = sys.argv[3]
            
            try:
                # User authentication
                user = client.authenticate('test_user', 'test_api_key')
                # Get collection
                collection = client.get_collection(collection_name, model_id)
                
                # List documents
                documents_info = collection.list_documents()
                print(f"Documents in collection '{collection_name}' (Model: {model_id}) (total: {documents_info['total']}):")
                print("=" * 50)
                for doc in documents_info['documents']:
                    print(f"Document ID: {doc['id']}")
                    print(f"Document Name: {doc['name']}")
                    if doc['metadata']:
                        print(f"Metadata: {doc['metadata']}")
                        print("-" * 30)
            except Exception as e:
                print(f"Failed to list documents: {e}")
        elif command == "add" and len(sys.argv) >= 3:
            # Add a document
            file_path = sys.argv[2]
            if len(sys.argv) >= 4:
                collection_name = sys.argv[3]
            if len(sys.argv) >= 5:
                model_id = sys.argv[4]
                
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return
                
            # User authentication
            user = client.authenticate('test_user', 'test_api_key')
            
            # Get collection
            collection = client.get_collection(collection_name, model_id)
            
            # Add document
            try:
                with open(file_path, 'rb') as f:
                    file_name = os.path.basename(file_path)
                    doc_id = f"doc_{hash(file_path) % 1000000}"
                    response = collection.add(
                        document_id=doc_id,
                        name=file_name,
                        file_content=f.read(),
                        metadata={'source': file_path, 'type': 'file'}
                    )
                    print(f"Added document successfully: {file_name} (Model: {model_id})")
            except Exception as e:
                print(f"Failed to add document: {e}")
                
        elif command == "search" and len(sys.argv) >= 3:
            # Search documents
            query = sys.argv[2]
            if len(sys.argv) >= 4:
                collection_name = sys.argv[3]
            if len(sys.argv) >= 5:
                model_id = sys.argv[4]
            
            # User authentication
            user = client.authenticate('test_user', 'test_api_key')

            # Get collection
            collection = client.get_collection(collection_name, model_id)
            
            # Query documents
            try:
                results = await collection.query(
                    query_text=query,
                    size=5
                )
                print(f"Results for query '{query}' in collection '{collection_name}' (Model: {model_id}) (total: {len(results)}):")
                print("=" * 50)
                for i, result in enumerate(results, 1):
                    print(f"{i}. Document: {result['document_name']}")
                    print(f"   Content: {result['chunk_content'][:200]}...")
                    print(f"   Score: {result.get('final_score', result['score']):.4f}")
                    print("-" * 30)
            except Exception as e:
                print(f"Search failed: {e}")
                
        else:
            print("Invalid command or arguments")
            usage()
    
    asyncio.run(main())
