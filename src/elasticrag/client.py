import logging
import os
from typing import Dict, List, Optional, Union

from elasticsearch import Elasticsearch, AsyncElasticsearch

from .model import Model
from .user import User
from .splitter import Splitter
from .collection import Collection


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
        self.client = Elasticsearch(self.hosts, **kwargs).options(ignore_status=404, request_timeout=600)
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
        user = User(self.client, username, api_key)
        return user.create_or_update(metadata)

    def delete_user(self, username: str) -> bool:
        """Delete user credentials"""
        user = User(self.client, username, "")
        return user.delete()
        
    def authenticate(self, username: str, api_key: str) -> 'User':
        """Authenticate user"""
        user = User(self.client, username, api_key)
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
