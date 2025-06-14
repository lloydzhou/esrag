import asyncio
import base64
import logging
from functools import partial
from typing import Dict, List, Optional
from elasticsearch import NotFoundError

from .utils import rrf


class Collection:
    """Collection (knowledge base) abstraction, corresponding to an ES index"""
    
    def __init__(self, client, name: str, user, model=None):
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
                                'score': [{
                                    'search_type': search_type,
                                    'score': chunk['_score'],
                                }],
                                'document_metadata': doc['_source'].get('metadata', {}),
                            }
                        else:
                            all_chunk_data[chunk_key]['score'].append({
                                'search_type': search_type,
                                'score': chunk['_score'],
                            })
            
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