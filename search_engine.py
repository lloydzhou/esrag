from elasticsearch_client import ElasticsearchClient
from config import INDEX_NAME, INFERENCE_SERVICE_ID

class SearchEngine:
    def __init__(self):
        self.es_client = ElasticsearchClient()
    
    def search(self, query, size=10):
        """Search using basic features compatible with basic license"""
        search_results = []
        
        # BM25 search - now searches within chunks
        bm25_search = {
            "query": {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunks.content^2", "attachment.title", "filename"]
                        }
                    },
                    "inner_hits": {
                        "size": 3,
                        "highlight": {
                            "fields": {
                                "chunks.content": {
                                    "fragment_size": 150,
                                    "number_of_fragments": 2
                                }
                            }
                        }
                    }
                }
            },
            "size": size,
            "_source": {
                "excludes": ["chunks.embedding"]
            }
        }
        
        bm25_response = self.es_client.client.search(
            index=INDEX_NAME,
            body=bm25_search
        )
        search_results.append(("bm25", bm25_response))
        
        # Vector search - now searches within chunk embeddings
        try:
            vector_search = {
                "query": {
                    "nested": {
                        "path": "chunks",
                        "query": {
                            "knn": {
                                "field": "chunks.embedding",
                                "query_vector_builder": {
                                    "text_embedding": {
                                        "model_id": INFERENCE_SERVICE_ID,
                                        "model_text": query
                                    }
                                },
                                "k": size,
                                "num_candidates": size * 5
                            }
                        },
                        "inner_hits": {
                            "size": 3,
                            "_source": ["chunks.content"]
                        }
                    }
                },
                "size": size,
                "_source": {
                    "excludes": ["chunks.embedding"]
                }
            }
            
            vector_response = self.es_client.client.search(
                index=INDEX_NAME,
                body=vector_search
            )
            search_results.append(("vector", vector_response))
            
        except Exception as e:
            print(f"Vector search failed: {e}")
        
        # Could add more search methods here in the future
        # For example: fuzzy search, phrase search, etc.
        # fuzzy_response = self._fuzzy_search(query, size)
        # search_results.append(("fuzzy", fuzzy_response))
        
        # Merge all results using RRF
        if len(search_results) > 1:
            combined_results = self._rrf_merge_multiple(search_results, size)
        else:
            combined_results = search_results[0][1]
        
        return self._format_results(combined_results)
    
    def _rrf_merge_multiple(self, search_results, size, k=60):
        """
        Generic RRF merge for multiple search result lists
        
        Args:
            search_results: List of tuples (search_type, response)
            size: Number of results to return
            k: RRF constant (default 60)
        
        Returns:
            Merged response in Elasticsearch format
        """
        # Collect all unique document IDs and their hits
        all_doc_ids = set()
        search_hits = {}  # search_type -> {doc_id -> hit}
        
        for search_type, response in search_results:
            hits = {hit['_id']: hit for hit in response['hits']['hits']}
            search_hits[search_type] = hits
            all_doc_ids.update(hits.keys())
        
        # Calculate RRF scores for each document
        rrf_scores = {}
        
        for doc_id in all_doc_ids:
            rrf_score = 0
            
            # Add contribution from each search method
            for search_type, response in search_results:
                if doc_id in search_hits[search_type]:
                    # Find rank in this search result
                    rank = next((i for i, hit in enumerate(response['hits']['hits']) 
                               if hit['_id'] == doc_id), len(response['hits']['hits']))
                    
                    # Add RRF contribution: 1 / (k + rank + 1)
                    rrf_score += 1 / (k + rank + 1)
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score and create merged results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:size]
        
        merged_hits = []
        for doc_id, rrf_score in sorted_docs:
            # Choose the best hit (prefer BM25 for highlights, then others)
            hit = None
            
            # Priority order: bm25 (has highlights) -> vector -> others
            for search_type in ['bm25', 'vector']:
                if search_type in search_hits and doc_id in search_hits[search_type]:
                    hit = search_hits[search_type][doc_id].copy()
                    break
            
            # Fallback to any available hit
            if hit is None:
                for search_type, hits in search_hits.items():
                    if doc_id in hits:
                        hit = hits[doc_id].copy()
                        break
            
            if hit:
                # Update score to RRF score (normalized for display)
                hit['_score'] = rrf_score * 100  # Scale for better readability
                hit['_rrf_score'] = rrf_score     # Keep original RRF score
                merged_hits.append(hit)
        
        # Create response structure
        merged_response = {
            'hits': {
                'hits': merged_hits,
                'total': {'value': len(merged_hits)}
            }
        }
        
        return merged_response
    
    def _fuzzy_search(self, query, size):
        """
        Example of additional search method that could be added
        """
        fuzzy_search = {
            "query": {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunks.content", "attachment.title", "filename"],
                            "fuzziness": "AUTO"
                        }
                    },
                    "inner_hits": {
                        "size": 3
                    }
                }
            },
            "size": size,
            "_source": {
                "excludes": ["chunks.embedding"]
            }
        }
        
        return self.es_client.client.search(
            index=INDEX_NAME,
            body=fuzzy_search
        )
    
    def _phrase_search(self, query, size):
        """
        Example of phrase search that could be added
        """
        phrase_search = {
            "query": {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunks.content", "attachment.title"],
                            "type": "phrase"
                        }
                    },
                    "inner_hits": {
                        "size": 3
                    }
                }
            },
            "size": size,
            "_source": {
                "excludes": ["chunks.embedding"]
            }
        }
        
        return self.es_client.client.search(
            index=INDEX_NAME,
            body=phrase_search
        )
    
    def advanced_search(self, query, size=10, enable_fuzzy=False, enable_phrase=False):
        """
        Advanced search with configurable search methods
        
        Args:
            query: Search query
            size: Number of results
            enable_fuzzy: Enable fuzzy search
            enable_phrase: Enable phrase search
        """
        search_results = []
        
        # Always include BM25
        bm25_search = {
            "query": {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["chunks.content^2", "attachment.title", "filename"]
                        }
                    },
                    "inner_hits": {
                        "size": 3,
                        "highlight": {
                            "fields": {
                                "chunks.content": {
                                    "fragment_size": 150,
                                    "number_of_fragments": 2
                                }
                            }
                        }
                    }
                }
            },
            "size": size,
            "_source": {"excludes": ["chunks.embedding"]}
        }
        
        bm25_response = self.es_client.client.search(index=INDEX_NAME, body=bm25_search)
        search_results.append(("bm25", bm25_response))
        
        # Vector search
        try:
            vector_response = self._vector_search(query, size)
            search_results.append(("vector", vector_response))
        except Exception as e:
            print(f"Vector search failed: {e}")
        
        # Optional fuzzy search
        if enable_fuzzy:
            try:
                fuzzy_response = self._fuzzy_search(query, size)
                search_results.append(("fuzzy", fuzzy_response))
            except Exception as e:
                print(f"Fuzzy search failed: {e}")
        
        # Optional phrase search
        if enable_phrase:
            try:
                phrase_response = self._phrase_search(query, size)
                search_results.append(("phrase", phrase_response))
            except Exception as e:
                print(f"Phrase search failed: {e}")
        
        # Merge all results
        if len(search_results) > 1:
            combined_results = self._rrf_merge_multiple(search_results, size)
        else:
            combined_results = search_results[0][1]
        
        return self._format_results(combined_results)
    
    def _vector_search(self, query, size):
        """Separate vector search method"""
        vector_search = {
            "query": {
                "nested": {
                    "path": "chunks",
                    "query": {
                        "knn": {
                            "field": "chunks.embedding",
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": INFERENCE_SERVICE_ID,
                                    "model_text": query
                                }
                            },
                            "k": size,
                            "num_candidates": size * 5
                        }
                    },
                    "inner_hits": {
                        "size": 3,
                        "_source": ["chunks.content"]
                    }
                }
            },
            "size": size,
            "_source": {"excludes": ["chunks.embedding"]}
        }
        
        return self.es_client.client.search(index=INDEX_NAME, body=vector_search)
    
    def _format_results(self, response):
        """Format search results"""
        results = []
        for hit in response['hits']['hits']:
            # Extract relevant chunk information from inner_hits
            relevant_chunks = []
            highlights = []
            
            if 'inner_hits' in hit and 'chunks' in hit['inner_hits']:
                for inner_hit in hit['inner_hits']['chunks']['hits']['hits']:
                    chunk_content = inner_hit['_source'].get('content', '')
                    relevant_chunks.append(chunk_content)
                    
                    # Extract highlights if available
                    if 'highlight' in inner_hit:
                        chunk_highlights = inner_hit['highlight'].get('chunks.content', [])
                        highlights.extend(chunk_highlights)
            
            # Fallback to original content if no chunks found
            content = ' ... '.join(relevant_chunks) if relevant_chunks else hit['_source']['attachment']['content'][:500] + '...'
            
            result = {
                'id': hit['_id'],
                'score': hit['_score'],
                'filename': hit['_source'].get('filename'),
                'content': content,
                'title': hit['_source']['attachment'].get('title', ''),
                'content_type': hit['_source']['attachment'].get('content_type', ''),
                'highlights': highlights,
                'relevant_chunks': relevant_chunks[:3]  # Show top 3 relevant chunks
            }
            
            # Include RRF score if available
            if '_rrf_score' in hit:
                result['rrf_score'] = hit['_rrf_score']
            
            results.append(result)
        
        return {
            'total': response['hits']['total']['value'],
            'results': results
        }
