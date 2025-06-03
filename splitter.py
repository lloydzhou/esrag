import logging
from elasticsearch import Elasticsearch
from typing import Dict


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


if __name__ == "__main__":
    # Example usage
    es = Elasticsearch("http://0.0.0.0:9200")
    splitter = Splitter(chunk_size=32, chunk_overlap=5)
    response = es.ingest.simulate(
        body={
            "pipeline": {
                "description": "Test pipeline for text splitting",
                "processors": [{
                    "script": {
                        "source": splitter.get_script_source(),
                        "lang": "painless",
                        "params": {
                            "splitter_config": {
                                "chunk_size": splitter.chunk_size,
                                "chunk_overlap": splitter.chunk_overlap
                            }
                        }
                    }
                }]
            },
            "docs": [
                {
                    "_source": {
                        "attachment": {
                            "content": "This is a test document. It contains multiple sentences. Let's see how it splits."
                        }
                    }
                }
            ]
        }
    )
    import pprint
    pprint.pprint(response['docs'])
