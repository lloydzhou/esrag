import logging
from typing import Dict
from elasticsearch import Elasticsearch


class Model:
    """Model management, responsible for inference service, pipeline, and index template"""
    
    def __init__(self, client, model_id: str, config: Dict):
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