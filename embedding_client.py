import requests
import json
from typing import List, Union
from config import HF_EMBEDDING_API_URL

class EmbeddingClient:
    def __init__(self):
        self.api_url = HF_EMBEDDING_API_URL
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Get embeddings from HuggingFace text-embeddings-inference API"""
        if isinstance(texts, str):
            texts = [texts]
        
        payload = {
            "inputs": texts,
            "truncate": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            embeddings = response.json()
            return embeddings[0] if len(embeddings) == 1 else embeddings
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling embedding API: {e}")
    
    def health_check(self) -> bool:
        """Check if the embedding service is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
