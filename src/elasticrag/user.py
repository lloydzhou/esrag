import logging
from typing import Dict, Optional
from elasticsearch import Elasticsearch, NotFoundError


class User:
    """User management"""
    
    def __init__(self, client: Elasticsearch, username: str, api_key: str, auth_index: str = "user_auth"):
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
            User.init_auth_index(self.client, self.auth_index)
            from datetime import datetime
            doc_data = {
                "username": self.username,
                "api_key": self.api_key,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            response = self.client.index(
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
            response = self.client.get(
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
            self.client.update(
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
            self.client.delete(
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
            response = self.client.get(
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
            self.client.update(
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