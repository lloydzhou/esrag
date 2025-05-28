import base64
from datetime import datetime
from elasticsearch_client import ElasticsearchClient
from config import INDEX_NAME, PIPELINE_NAME

class DocumentProcessor:
    def __init__(self):
        self.es_client = ElasticsearchClient()
    
    def encode_file(self, file_path):
        """Encode file to base64"""
        with open(file_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    
    def ingest_document(self, file_path, filename=None):
        """Ingest document into knowledge base"""
        if filename is None:
            filename = file_path.split('/')[-1]
        
        # Encode file to base64
        encoded_content = self.encode_file(file_path)
        
        # Document body
        doc_body = {
            "data": encoded_content,
            "filename": filename,
            "created_at": datetime.now().isoformat()
        }
        
        # Index document with pipeline
        response = self.es_client.client.index(
            index=INDEX_NAME,
            body=doc_body,
            pipeline=PIPELINE_NAME
        )
        
        return response
    
    def ingest_from_base64(self, base64_content, filename):
        """Ingest document from base64 content"""
        doc_body = {
            "data": base64_content,
            "filename": filename,
            "created_at": datetime.now().isoformat()
        }
        
        response = self.es_client.client.index(
            index=INDEX_NAME,
            body=doc_body,
            pipeline=PIPELINE_NAME
        )
        
        return response
