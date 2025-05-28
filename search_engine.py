from elasticsearch_client import ElasticsearchClient
from config import INDEX_NAME, INFERENCE_SERVICE_ID

class SearchEngine:
    def __init__(self):
        self.es_client = ElasticsearchClient()
    
    def search(self, query, size=10):
        """Search using retriever API with RRF combining BM25 and vector similarity"""
        search_body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["attachment.content^2", "attachment.title", "filename"]
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "text_embedding",
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": INFERENCE_SERVICE_ID,
                                        "model_text": query
                                    }
                                },
                                "k": size * 2,
                                "num_candidates": size * 10
                            }
                        }
                    ],
                    "window_size": size * 2,
                    "rank_constant": 20
                }
            },
            "size": size,
            "_source": {
                "excludes": ["text_embedding"]
            },
            "highlight": {
                "fields": {
                    "attachment.content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
        
        response = self.es_client.client.search(
            index=INDEX_NAME,
            body=search_body
        )
        
        return self._format_results(response)
    
    def _format_results(self, response):
        """Format search results"""
        results = []
        for hit in response['hits']['hits']:
            result = {
                'id': hit['_id'],
                'score': hit['_score'],
                'filename': hit['_source'].get('filename'),
                'content': hit['_source']['attachment']['content'][:500] + '...' if len(hit['_source']['attachment']['content']) > 500 else hit['_source']['attachment']['content'],
                'title': hit['_source']['attachment'].get('title', ''),
                'content_type': hit['_source']['attachment'].get('content_type', ''),
                'highlights': hit.get('highlight', {}).get('attachment.content', [])
            }
            results.append(result)
        
        return {
            'total': response['hits']['total']['value'],
            'results': results
        }
