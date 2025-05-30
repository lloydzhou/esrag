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
            # Service might already exist
            print(f"Inference service creation: {e}")
            return None
    
    def get_chunking_script(self, chunk_size=1000, overlap=200):
        """Get the Painless script for text chunking"""
        return """
            if (ctx.attachment?.content != null) {
                def content = ctx.attachment.content;
                def chunks = [];
                int chunkSize = """ + str(chunk_size) + """;
                int overlap = """ + str(overlap) + """;
                int start = 0;
                
                // Ensure overlap is not larger than chunk size
                if (overlap >= chunkSize) {
                    overlap = chunkSize / 2;
                }
                
                while (start < content.length()) {
                    int end = (int)Math.min(start + chunkSize, content.length());
                    
                    // Only create chunk if there's content
                    if (start < content.length()) {
                        def chunk = content.substring(start, end);
                        def chunkMap = ['content': chunk];
                        chunks.add(chunkMap);
                    }
                    
                    // Calculate next start position, ensuring it doesn't go backwards
                    int nextStart = start + chunkSize - overlap;
                    if (nextStart <= start) {
                        nextStart = start + 1;
                    }
                    start = nextStart;
                    
                    // Prevent infinite loop
                    if (end >= content.length()) {
                        break;
                    }
                }
                ctx.chunks = chunks;
            } else {
                ctx.chunks = [];
            }
        """

    def test_painless_script(self):
        """Test painless script syntax"""
        try:            
            test_pipeline = {
                "processors": [
                    {
                        "script": {
                            "lang": "painless",
                            "source": self.get_chunking_script(50, 10)
                        }
                    }
                ]
            }
            
            test_doc = {"_source": {"attachment": {"content": "This is a test content that should be chunked properly without errors."}}}
            
            result = self.client.ingest.simulate(
                body={
                    "pipeline": test_pipeline,
                    "docs": [test_doc]
                }
            )
            print(f"Chunking script test result: {result}")
            return True
        except Exception as e:
            print(f"Chunking script test failed: {e}")
            return False

    def create_pipeline(self):
        """Create ingest pipeline with attachment and text embedding processors"""
        pipeline_body = {
            "processors": [
                {
                    "attachment": {
                        "field": "data",
                        "target_field": "attachment",
                        "remove_binary": True,
                        "indexed_chars": -1,
                        "ignore_missing": True
                    }
                },
                {
                    "script": {
                        "lang": "painless",
                        "source": self.get_chunking_script()
                    }
                },
                {
                    "foreach": {
                        "field": "chunks",
                        "processor": {
                            "inference": {
                                "model_id": INFERENCE_SERVICE_ID,
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
        }
        
        try:
            return self.client.ingest.put_pipeline(
                id=PIPELINE_NAME,
                body=pipeline_body
            )
        except Exception as e:
            logging.exception(e)
            print(f"Pipeline creation error: {e}")
            return None
    
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
                    "chunks": {
                        "type": "nested",
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": VECTOR_DIMS,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
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
    
    def delete_index(self):
        """Delete the index if it exists"""
        try:
            return self.client.indices.delete(index=INDEX_NAME, ignore=[400, 404])
        except Exception as e:
            logging.exception(e)
            print(f"Error deleting index: {e}")
            return None

    def setup(self, force_recreate_index=False):
        print("Testing painless script...")
        self.test_painless_script()
        
        """Setup inference service, pipeline and index"""
        print("Creating inference service...")
        self.create_inference_service()
        print("Creating pipeline...")
        self.create_pipeline()
        if force_recreate_index:
            print("Deleting existing index...")
            self.delete_index()
        
        print("Creating index...")
        self.create_index()
