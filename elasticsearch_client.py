import logging
from elasticsearch import Elasticsearch
from config import ES_HOST, ES_PORT, ES_USERNAME, ES_PASSWORD, INDEX_NAME, PIPELINE_NAME, INFERENCE_SERVICE_ID, INFERENCE_SERVICE_URL, VECTOR_DIMS

class ElasticsearchClient:
    def __init__(self):
        self.client = Elasticsearch(
            hosts=[f"http://{ES_HOST}:{ES_PORT}"],
            # basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False
        )
    
    def create_inference_service(self):
        """Create inference service for text embedding"""
        service_config = {
            "service": "hugging_face",
            "service_settings": {
                "api_key": "placeholder",
                "url": INFERENCE_SERVICE_URL
            }
        }
        
        try:
            return self.client.inference.put(
                inference_id=INFERENCE_SERVICE_ID,
                body=service_config,
                task_type="text_embedding"
            )
        except Exception as e:
            logging.exception(e)
            # Service might already exist
            print(f"Inference service creation: {e}")
            return None
    
    def create_pipeline(self):
        """Create ingest pipeline with attachment and text embedding processors"""
        pipeline_body = {
            "processors": [
                {
                    "attachment": {
                        "field": "data",
                        "target_field": "attachment",
                        "indexed_chars": -1
                    }
                },
                {
                    "inference": {
                        "model_id": INFERENCE_SERVICE_ID,
                        "input_output": {
                            "input_field": "attachment.content",
                            "output_field": "text_embedding"
                        }
                    }
                },
                {
                    "remove": {
                        "field": "data"
                    }
                }
            ]
        }
        
        return self.client.ingest.put_pipeline(
            id=PIPELINE_NAME,
            body=pipeline_body
        )
    
    def create_index(self):
        """Create index with proper mappings for text and vector fields"""
        mapping = {
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "attachment": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "author": {"type": "text"},
                            "content_type": {"type": "keyword"},
                            "content_length": {"type": "long"}
                        }
                    },
                    "text_embedding": {
                        "type": "dense_vector",
                        "dims": VECTOR_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {"type": "date"}
                }
            }
        }
        
        return self.client.indices.create(
            index=INDEX_NAME,
            body=mapping,
            ignore=400  # Ignore if index already exists
        )
    
    def setup(self):
        """Setup inference service, pipeline and index"""
        print("Creating inference service...")
        self.create_inference_service()
        print("Creating pipeline...")
        self.create_pipeline()
        print("Creating index...")
        self.create_index()
