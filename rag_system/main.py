import logging
import sys
import time
import requests
from rag_system import RAGSystem
from config import INFERENCE_SERVICE_URL

def wait_for_services():
    """Wait for Elasticsearch and embedding service to be ready"""
    print("Waiting for services to be ready...")
    print("Note: First startup may take 2-3 minutes for model download...")
    
    # Wait for embedding service (longer timeout for model download)
    for i in range(150):  # 5 minutes timeout
        try:
            # Test the health endpoint
            health_url = INFERENCE_SERVICE_URL.replace('/embed', '/health')
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print("✓ Embedding service is ready")
                break
        except:
            pass
        if i % 15 == 0:  # Print progress every 30 seconds
            print(f"  Waiting for embedding service... ({i+1}/150) - downloading model if first time")
        time.sleep(2)
    else:
        print("✗ Embedding service failed to start")
        print("  Try: docker-compose logs huggingface-embedding")
        return False
    
    # Wait for Elasticsearch
    rag = RAGSystem()
    for i in range(30):
        try:
            rag.es_client.client.cluster.health(wait_for_status='yellow', timeout='1s')
            print("✓ Elasticsearch is ready")
            return True
        except:
            print(f"  Waiting for Elasticsearch... ({i+1}/30)")
            time.sleep(2)
    
    print("✗ Elasticsearch failed to start")
    return False

def main():
    rag = RAGSystem()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py setup")
        print("  python main.py wait")
        print("  python main.py add <file_path>")
        print("  python main.py search <query>")
        print("  python main.py advanced-search <query> [--fuzzy] [--phrase]")
        return
    
    command = sys.argv[1]
    
    if command == "wait":
        if wait_for_services():
            print("All services are ready!")
        else:
            print("Some services failed to start")
            sys.exit(1)
    
    elif command == "setup":
        if wait_for_services():
            rag.setup()
        else:
            print("Cannot setup: services not ready")
            sys.exit(1)
    
    elif command == "add" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        try:
            result = rag.add_document(file_path)
            print(f"Document added successfully: {result['_id']}")
        except Exception as e:
            logging.exception(e)
            print(f"Error adding document: {e}")
    
    elif command == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        try:
            results = rag.search(query)
            print(f"Found {results['total']} results:")
            for i, result in enumerate(results['results'], 1):
                print(f"\n{i}. {result['filename']} (Score: {result['score']:.4f})")
                if 'rrf_score' in result:
                    print(f"   RRF Score: {result['rrf_score']:.6f}")
                
                # Show highlights if available
                if result.get('highlights'):
                    print("   📝 Highlights:")
                    for highlight in result['highlights'][:2]:  # Show top 2 highlights
                        print(f"      {highlight}")
                
                # Show relevant chunks
                if result.get('relevant_chunks'):
                    print("   📄 Relevant chunks:")
                    for j, chunk in enumerate(result['relevant_chunks'][:2], 1):
                        chunk_preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                        print(f"      {j}. {chunk_preview}")
                else:
                    # Fallback to content preview
                    print(f"   📄 Content: {result['content'][:200]}...")
                
                if result.get('title'):
                    print(f"   📋 Title: {result['title']}")
        except Exception as e:
            print(f"Error searching: {e}")
    
    elif command == "advanced-search" and len(sys.argv) > 2:
        # Parse arguments
        args = sys.argv[2:]
        enable_fuzzy = '--fuzzy' in args
        enable_phrase = '--phrase' in args
        
        # Remove flags from query
        query_words = [word for word in args if not word.startswith('--')]
        query = " ".join(query_words)
        
        try:
            results = rag.advanced_search(query, enable_fuzzy=enable_fuzzy, enable_phrase=enable_phrase)
            search_methods = ["BM25", "Vector"]
            if enable_fuzzy:
                search_methods.append("Fuzzy")
            if enable_phrase:
                search_methods.append("Phrase")
            
            print(f"Advanced search using: {', '.join(search_methods)}")
            print(f"Found {results['total']} results:")
            for i, result in enumerate(results['results'], 1):
                print(f"\n{i}. {result['filename']} (Score: {result['score']:.4f})")
                if 'rrf_score' in result:
                    print(f"   RRF Score: {result['rrf_score']:.6f}")
                
                # Show highlights if available
                if result.get('highlights'):
                    print("   📝 Highlights:")
                    for highlight in result['highlights'][:2]:
                        print(f"      {highlight}")
                
                # Show relevant chunks with more detail for advanced search
                if result.get('relevant_chunks'):
                    print("   📄 Relevant chunks:")
                    for j, chunk in enumerate(result['relevant_chunks'][:3], 1):
                        chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        print(f"      {j}. {chunk_preview}")
                else:
                    print(f"   📄 Content: {result['content'][:250]}...")
                
                if result.get('title'):
                    print(f"   📋 Title: {result['title']}")
                if result.get('content_type'):
                    print(f"   📄 Type: {result['content_type']}")
        except Exception as e:
            print(f"Error in advanced search: {e}")
    
    else:
        print("Invalid command or missing arguments")

if __name__ == "__main__":
    main()
