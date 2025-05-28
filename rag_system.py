from elasticsearch_client import ElasticsearchClient
from document_processor import DocumentProcessor
from search_engine import SearchEngine

class RAGSystem:
    def __init__(self):
        self.es_client = ElasticsearchClient()
        self.document_processor = DocumentProcessor()
        self.search_engine = SearchEngine()
    
    def setup(self):
        """Initialize the RAG system"""
        print("Setting up RAG system...")
        self.es_client.setup()
        print("RAG system setup complete!")
    
    def add_document(self, file_path, filename=None):
        """Add document to knowledge base"""
        return self.document_processor.ingest_document(file_path, filename)
    
    def add_document_from_base64(self, base64_content, filename):
        """Add document from base64 content"""
        return self.document_processor.ingest_from_base64(base64_content, filename)
    
    def search(self, query, size=10):
        """Search in knowledge base"""
        return self.search_engine.search(query, size)
    
    def get_answer(self, query, size=5):
        """Get answer based on search results (basic implementation)"""
        search_results = self.search(query, size)
        
        if not search_results['results']:
            return "No relevant documents found."
        
        # Simple answer generation based on top results
        context = "\n\n".join([
            f"Document: {result['filename']}\nContent: {result['content']}"
            for result in search_results['results'][:3]
        ])
        
        return {
            'query': query,
            'context': context,
            'sources': [
                {
                    'filename': result['filename'],
                    'score': result['score']
                }
                for result in search_results['results'][:3]
            ]
        }
