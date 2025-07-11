import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from .client import Client
from .user import User


def load_config():
    """Load configuration from .env file and environment variables"""
    # Try to find .env file in current directory or parent directories
    current_dir = Path.cwd()
    env_path = None
    
    # Look for .env file in current directory and up to 3 parent directories
    for path in [current_dir] + list(current_dir.parents)[:3]:
        potential_env = path / ".env"
        if potential_env.exists():
            env_path = potential_env
            break
    
    # Load .env file if found
    if env_path:
        load_dotenv(env_path)
        print(f"Loaded configuration from: {env_path}")
    else:
        # Try to load from default locations
        load_dotenv()


def get_config_value(key: str, default: str = None, required: bool = False) -> str:
    """Get configuration value from environment variables"""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required configuration '{key}' not found in environment variables")
    return value


def create_arg_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="ElasticRAG - Elasticsearch-based RAG system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  elasticrag setup
  elasticrag --host localhost:9200 -u admin -k secret list-models
  elasticrag search "your query" my_collection my_model
  elasticrag add document.pdf -c my_collection -m my_model
  elasticrag server --port 7860

Environment variables:
  ELASTICSEARCH_HOST     Elasticsearch host (default: http://localhost:9200)
  ELASTICRAG_USERNAME    Username for authentication (default: test_user)
  ELASTICRAG_API_KEY     API key for authentication (default: test_api_key)
  TEXT_EMBEDDING_URL     Text embedding service URL
  TEXT_EMBEDDING_API_KEY Text embedding API key
  ELASTICRAG_ADMIN_USERNAME  Admin username for web interface (default: admin)
  ELASTICRAG_ADMIN_PASSWORD  Admin password for web interface (default: admin123)
        """
    )
    
    # Global options (changed -h to --host only to avoid conflict with help)
    parser.add_argument(
        "--host", 
        help="Elasticsearch host (overrides ELASTICSEARCH_HOST)"
    )
    parser.add_argument(
        "-u", "--username",
        help="Username for authentication (overrides ELASTICRAG_USERNAME)"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="API key for authentication (overrides ELASTICRAG_API_KEY)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # setup command
    setup_parser = subparsers.add_parser("setup", help="Initialize system with default user and model")
    setup_parser.add_argument(
        "--embedding-url",
        help="Text embedding service URL (overrides TEXT_EMBEDDING_URL)"
    )
    setup_parser.add_argument(
        "--embedding-api-key",
        help="Text embedding API key (overrides TEXT_EMBEDDING_API_KEY)"
    )
    
    # list-models command
    subparsers.add_parser("list-models", help="List available models")
    
    # list-users command
    subparsers.add_parser("list-users", help="List all users")
    
    # list-collections command
    subparsers.add_parser("list-collections", help="List all collections")
    
    # list-documents command
    list_docs_parser = subparsers.add_parser("list-documents", help="List documents in a collection")
    list_docs_parser.add_argument("collection", nargs="?", help="Collection name")
    list_docs_parser.add_argument("model", nargs="?", help="Model ID")
    
    # add command
    add_parser = subparsers.add_parser("add", help="Add a document to collection")
    add_parser.add_argument("file_path", help="Path to the file to add")
    add_parser.add_argument("-c", "--collection", help="Collection name")
    add_parser.add_argument("-m", "--model", help="Model ID")
    
    # search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-c", "--collection", help="Collection name")
    search_parser.add_argument("-m", "--model", help="Model ID")
    search_parser.add_argument("-s", "--size", type=int, default=5, help="Number of results to return")
    
    # server command
    server_parser = subparsers.add_parser("server", help="Start Gradio web interface")
    server_parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port for web interface (default: 7860)"
    )
    server_parser.add_argument(
        "--host", 
        default="http://0.0.0.0:9200", 
        help="Host for web interface (default: http://0.0.0.0:9200)"
    )
    server_parser.add_argument(
        "--share", 
        action="store_true", 
        default=False,
        help="Create public link via Gradio share"
    )
    server_parser.add_argument(
        "--admin-username",
        help="Admin username (overrides ELASTICRAG_ADMIN_USERNAME)"
    )
    server_parser.add_argument(
        "--admin-password",
        help="Admin password (overrides ELASTICRAG_ADMIN_PASSWORD)"
    )
    
    return parser


def usage():
    parser = create_arg_parser()
    parser.print_help()
    sys.exit(1)


async def async_main():
    """Async CLI main entry point"""
    # Load configuration first
    load_config()
    
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if not args.command:
        usage()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration values
    es_host = args.host or get_config_value("ELASTICSEARCH_HOST", "http://localhost:9200")
    username = args.username or get_config_value("ELASTICRAG_USERNAME", "test_user")
    api_key = args.api_key or get_config_value("ELASTICRAG_API_KEY", "test_api_key")
    
    print(f"Connecting to Elasticsearch: {es_host}")
    print(f"Username: {username}")
    
    # Create client
    client = Client(es_host)
    
    # Default values
    default_collection = get_config_value("ELASTICRAG_DEFAULT_COLLECTION", "test_documents123")
    default_model = get_config_value("ELASTICRAG_DEFAULT_MODEL", "bge-small-en-v1.5")
    
    if args.command == "setup":
        # Get embedding configuration
        embedding_url = (
            args.embedding_url or 
            get_config_value("TEXT_EMBEDDING_URL", "http://192.168.9.62:8080/embed")
        )
        embedding_api_key = (
            args.embedding_api_key or 
            get_config_value("TEXT_EMBEDDING_API_KEY", "placeholder")
        )
        
        print(f"Embedding URL: {embedding_url}")
        
        # Initialize user
        success = client.add_user(username, api_key, metadata={
            "email": get_config_value("ELASTICRAG_USER_EMAIL", "admin@example.com"),
            "role": "admin",
            "preferences": {
                "language": "zh",
                "theme": "dark"
            }
        })
        if success:
            print("User initialized successfully")
        else:
            print("User initialization failed")
            
        # BGE model configuration
        config = {
            "service": "hugging_face",
            "service_settings": {
                "api_key": embedding_api_key,
                "url": embedding_url,
            },
            "dimensions": 384
        }
        client.register_model(default_model, config)
        print(f"Model '{default_model}' registered successfully")
        
    elif args.command == "list-models":
        try:
            models = client.list_models()
            print(f"Available models (total: {len(models)}):")
            print("=" * 50)
            for model in models:
                print(f"Model ID: {model['model_id']}")
                print(f"Service Type: {model['config'].get('service', 'unknown')}")
                print(f"Vector Dimensions: {model['dimensions']}")
                print("-" * 30)
        except Exception as e:
            print(f"Failed to list models: {e}")
            
    elif args.command == "list-users":
        try:
            users_info = User.list_all_users(client.client, "user_auth")
            print(f"User list (total: {users_info['total']}):")
            print("=" * 50)
            for user in users_info['users']:
                print(f"Username: {user['username']}")
                print(f"Created At: {user['created_at']}")
                print(f"Last Login: {user['last_login'] or 'Never'}")
                if user['metadata']:
                    print(f"Metadata: {user['metadata']}")
                    print("-" * 30)
        except Exception as e:
            print(f"Failed to list users: {e}")
            
    elif args.command == "list-collections":
        try:
            # User authentication
            user = client.authenticate(username, api_key)
            collections = client.list_collections()
            print(f"Collections for user {user.username} (total: {len(collections)}):")
            print("=" * 50)
            for collection in collections:
                print(f"Collection Name: {collection['name']}")
                print(f"Model ID: {collection['model_id']}")
                print(f"Index Name: {collection['index']}")
                print(f"Health: {collection['health']}")
                print(f"Status: {collection['status']}")
                print(f"Document Count: {collection['doc_count']}")
                print(f"Storage Size: {collection['store_size']}")
                print("-" * 30)
        except Exception as e:
            print(f"Failed to list collections: {e}")
            
    elif args.command == "list-documents":
        collection_name = args.collection or default_collection
        model_id = args.model or default_model
        
        try:
            # User authentication
            user = client.authenticate(username, api_key)
            # Get collection
            collection = client.get_collection(collection_name, model_id)
            
            # List documents
            documents_info = collection.list_documents()
            print(f"Documents in collection '{collection_name}' (Model: {model_id}) (total: {documents_info['total']}):")
            print("=" * 50)
            for doc in documents_info['documents']:
                print(f"Document ID: {doc['id']}")
                print(f"Document Name: {doc['name']}")
                if doc['metadata']:
                    print(f"Metadata: {doc['metadata']}")
                    print("-" * 30)
        except Exception as e:
            print(f"Failed to list documents: {e}")
            
    elif args.command == "add":
        file_path = args.file_path
        collection_name = args.collection or default_collection
        model_id = args.model or default_model
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return
            
        # User authentication
        user = client.authenticate(username, api_key)
        
        # Get collection
        collection = client.get_collection(collection_name, model_id)
        
        # Add document
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                doc_id = f"doc_{hash(file_path) % 1000000}"
                response = collection.add(
                    document_id=doc_id,
                    name=file_name,
                    file_content=f.read(),
                    metadata={'source': file_path, 'type': 'file'}
                )
                print(f"Added document successfully: {file_name} (Collection: {collection_name}, Model: {model_id})")
        except Exception as e:
            print(f"Failed to add document: {e}")
            
    elif args.command == "search":
        query = args.query
        collection_name = args.collection or default_collection
        model_id = args.model or default_model
        size = args.size
        
        # User authentication
        user = client.authenticate(username, api_key)

        # Get collection
        collection = client.get_collection(collection_name, model_id)
        
        # Query documents
        try:
            results = await collection.query(
                query_text=query,
                size=size
            )
            print(f"Results for query '{query}' in collection '{collection_name}' (Model: {model_id}) (total: {len(results)}):")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(f"{i}. Document: {result['document_name']}")
                print(f"   Content: {result['chunk_content'][:200]}...")
                print(f"   Score: {result.get('score')}")
                print(f"RRFScore: {result.get('rrf_score'):.4f}")
                print("-" * 30)
        except Exception as e:
            logging.exception(e)
            print(f"Search failed: {e}")
            
    elif args.command == "server":
        # Start Gradio web interface
        admin_username = (
            args.admin_username or 
            get_config_value("ELASTICRAG_ADMIN_USERNAME", "admin")
        )
        admin_password = (
            args.admin_password or 
            get_config_value("ELASTICRAG_ADMIN_PASSWORD", "admin123")
        )
        
        print(f"Starting ElasticRAG web interface...")
        print(f"Elasticsearch: {es_host}")
        print(f"Admin username: {admin_username}")
        print(f"Web interface: http://{args.host}:{args.port}")
        
        try:
            # Check if gradio is available
            try:
                import gradio
            except ImportError:
                print("\n❌ Error: Gradio is not installed.")
                print("Web interface requires gradio. Install it with:")
                print("  uv add 'elasticrag[web]'")
                print("  # or")
                print("  uv add gradio pandas")
                return
            
            from .server import create_server
            server = create_server(client, admin_username, admin_password)
            server.launch(port=args.port, share=args.share)
        except Exception as e:
            print(f"Failed to start web interface: {e}")
    
    else:
        print(f"Unknown command: {args.command}")
        usage()


def main():
    """Synchronous CLI entry point that runs the async main function"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if os.getenv("ELASTICRAG_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()