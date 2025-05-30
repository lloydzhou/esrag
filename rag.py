import asyncio
import logging
import base64
import json
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse
from elasticsearch import Elasticsearch, AsyncElasticsearch, NotFoundError


class Client:
    """RAG系统的主要客户端，管理ES连接和全局配置"""
    
    def __init__(self, hosts: Union[str, List[str]], **kwargs):
        """
        初始化客户端
        
        Args:
            hosts: Elasticsearch主机地址
            **kwargs: 其他ES连接参数
        """
        self.hosts = hosts if isinstance(hosts, list) else [hosts]
        self.client = Elasticsearch(self.hosts, **kwargs)
        self.async_client = AsyncElasticsearch(self.hosts, **kwargs)
        self._collections = {}
        self._inferences = {}
        self._user = None
        self._init_scripts()

    def add_user(self, username: str, api_key: str, metadata: Optional[Dict] = None) -> bool:
        """添加或更新用户凭据"""
        user = User(self, username, api_key)
        return user.create_or_update(metadata)

    def delete_user(self, username: str) -> bool:
        """删除用户凭据"""
        user = User(self, username, "")
        return user.delete()
        
    def authenticate(self, username: str, api_key: str) -> 'User':
        """用户认证"""
        user = User(self, username, api_key)
        if user.validate():
            self._user = user
            return self._user
        else:
            raise ValueError(f"用户认证失败: {username}")
    
    def get_collection(self, name: str, inference_config: Optional[Dict] = None) -> 'Collection':
        """获取或创建集合（知识库）"""
        if not self._user:
            raise ValueError("请先调用 authenticate() 进行用户认证")
            
        collection_key = f"{self._user.username}__{name}"
        if collection_key not in self._collections:
            self._collections[collection_key] = Collection(
                client=self,
                name=name,
                user=self._user,
                inference_config=inference_config
            )
        return self._collections[collection_key]
    
    def list_collections(self) -> List[str]:
        """列出用户的所有集合"""
        if not self._user:
            raise ValueError("请先调用 authenticate() 进行用户认证")
        
        try:
            pattern = f"{self._user.username}_*"
            response = self.client.cat.indices(index=pattern, format='json', ignore=[404])
            print(f"列出集合: {response}")
            if response:
                prefix = f"{self._user.username}__"
                collections = []
                for idx in response:
                    if idx['index'].startswith(prefix):
                        collection_name = idx['index'].replace(prefix, "")
                        collections.append({
                            "name": collection_name,
                            "index": idx['index'],
                            "health": idx.get('health', 'unknown'),
                            "status": idx.get('status', 'unknown'),
                            "doc_count": idx.get('docs.count', '0'),
                            "store_size": idx.get('store.size', '0b')
                        })
                return collections
            return []
        except Exception:
            return []
    
    def get_inference(self, model_id: str, service_config: Dict) -> 'InferenceService':
        """获取或创建推理服务配置"""
        if model_id not in self._inferences:
            self._inferences[model_id] = InferenceService(
                client=self.client,
                model_id=model_id,
                config=service_config
            )
        return self._inferences[model_id]
    
    def _init_scripts(self):
        if self.client.get_script(id="text_splitter", ignore=[404]):
            logging.debug("文本分片脚本已存在")
            return
        """初始化ES脚本"""
        script_source = """
            if (ctx.attachment?.content != null) {
                def content = ctx.attachment.content;
                def config = params.splitter_config;
                def chunks = [];
                int chunkSize = config.chunk_size;
                int overlap = config.chunk_overlap;
                
                for (int i = 0; i < content.length(); i += (chunkSize - overlap)) {
                    int end = (int)Math.min(i + chunkSize, content.length());
                    def chunkContent = content.substring(i, end);
                    
                    chunks.add([
                        'content': chunkContent,
                        'metadata': [
                            'index': chunks.size(),
                            'offset': i,
                            'length': chunkContent.length()
                        ]
                    ]);
                    
                    if (end >= content.length()) break;
                }
                
                ctx.chunks = chunks;
            } else {
                ctx.chunks = [];
            }
        """
        
        try:
            self.client.put_script(
                id="text_splitter",
                body={
                    "script": {
                        "lang": "painless",
                        "source": script_source,
                    }
                }
            )
            logging.debug("文本分片脚本初始化成功")
        except Exception as e:
            logging.warning(f"脚本初始化失败: {e}")


class User:
    """用户管理"""
    
    def __init__(self, client: Client, username: str, api_key: str, auth_index: str = "user_auth"):
        self.client = client
        self.username = username
        self.api_key = api_key
        self.auth_index = auth_index
    
    @staticmethod
    def init_auth_index(es_client: Elasticsearch, auth_index: str):
        """初始化用户认证索引"""
        if es_client.indices.exists(index=auth_index):
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "username": {
                        "type": "keyword"
                    },
                    "api_key": {
                        "type": "keyword"
                    },
                    "created_at": {
                        "type": "date"
                    },
                    "last_login": {
                        "type": "date"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": False
                    }
                }
            },
            "settings": {
                "index": {
                    "number_of_replicas": 0,
                }
            }
        }
        
        try:
            es_client.indices.create(index=auth_index, body=mapping)
            logging.debug(f"创建用户认证索引成功: {auth_index}")
        except Exception as e:
            logging.error(f"创建用户认证索引失败: {e}")
            raise
    
    def create_or_update(self, metadata: Optional[Dict] = None) -> bool:
        """创建或更新用户凭据"""
        try:
            User.init_auth_index(self.client.client, self.auth_index)
            from datetime import datetime
            doc_data = {
                "username": self.username,
                "api_key": self.api_key,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            response = self.client.client.index(
                index=self.auth_index,
                id=self.username,  # 使用username作为文档ID
                body=doc_data,
                refresh='wait_for'
            )
            logging.debug(f"用户 {self.username} 添加成功")
            return True
        except Exception as e:
            logging.error(f"添加用户失败: {e}")
            return False
    
    def validate(self) -> bool:
        """验证用户凭据"""
        try:
            response = self.client.client.get(
                index=self.auth_index,
                id=self.username
            )
            stored_api_key = response['_source'].get('api_key')
            
            if stored_api_key == self.api_key:
                # 更新最后登录时间
                self._update_last_login()
                return True
            else:
                logging.warning(f"用户 {self.username} API密钥验证失败")
                return False
                
        except NotFoundError:
            logging.warning(f"用户 {self.username} 不存在")
            return False
        except Exception as e:
            logging.error(f"验证用户失败: {e}")
            return False
    
    def _update_last_login(self):
        """更新最后登录时间"""
        try:
            from datetime import datetime
            self.client.client.update(
                index=self.auth_index,
                id=self.username,
                body={
                    "doc": {
                        "last_login": datetime.now().isoformat()
                    }
                },
                refresh='wait_for'
            )
        except Exception as e:
            logging.warning(f"更新登录时间失败: {e}")
    
    def delete(self) -> bool:
        """删除用户"""
        try:
            self.client.client.delete(
                index=self.auth_index,
                id=self.username,
                refresh='wait_for'
            )
            logging.debug(f"用户 {self.username} 删除成功")
            return True
        except NotFoundError:
            logging.warning(f"用户 {self.username} 不存在")
            return False
        except Exception as e:
            logging.error(f"删除用户失败: {e}")
            return False
    
    def get_info(self) -> Optional[Dict]:
        """获取用户信息"""
        try:
            response = self.client.client.get(
                index=self.auth_index,
                id=self.username
            )
            user_info = response['_source'].copy()
            # 不返回API密钥
            user_info.pop('api_key', None)
            return user_info
        except NotFoundError:
            return None
        except Exception as e:
            logging.error(f"获取用户信息失败: {e}")
            return None
    
    def update_metadata(self, metadata: Dict) -> bool:
        """更新用户元数据"""
        try:
            self.client.client.update(
                index=self.auth_index,
                id=self.username,
                body={
                    "doc": {
                        "metadata": metadata
                    }
                },
                refresh='wait_for'
            )
            logging.debug(f"用户 {self.username} 元数据更新成功")
            return True
        except Exception as e:
            logging.error(f"更新用户元数据失败: {e}")
            return False

    @classmethod
    def list_all_users(cls, es_client: Elasticsearch, auth_index: str, 
                      offset: int = 0, limit: int = 10) -> Dict:
        """列出所有用户（类方法，用于管理员功能）"""
        try:
            response = es_client.search(
                index=auth_index,
                body={
                    "query": {"match_all": {}},
                    "_source": ["username", "created_at", "last_login", "metadata"],
                    "from": offset,
                    "size": limit,
                    "sort": [{"created_at": {"order": "desc"}}]
                }
            )
            
            return {
                "total": response['hits']['total']['value'],
                "users": [
                    {
                        "username": hit['_source'].get('username', ''),
                        "created_at": hit['_source'].get('created_at', ''),
                        "last_login": hit['_source'].get('last_login', ''),
                        "metadata": hit['_source'].get('metadata', {})
                    }
                    for hit in response['hits']['hits']
                ]
            }
        except Exception as e:
            logging.error(f"列出用户失败: {e}")
            return {"total": 0, "users": []}


class InferenceService:
    """推理服务管理"""
    
    def __init__(self, client: Elasticsearch, model_id: str, config: Dict):
        self.client = client
        self.model_id = model_id
        self.config = config
        self.inference_id = f"{model_id}__inference"
        self._init_inference()
    
    def get_embedding_dims(self) -> int:
        """获取嵌入向量的维度"""
        # 从配置中获取维度，如果没有配置则使用默认值
        return self.config.get("embedding_dims", 384)
    
    def _init_inference(self):
        """初始化推理服务"""
        try:
            response = self.client.inference.get(inference_id=self.inference_id)
            logging.debug(f'推理服务已存在: {self.inference_id}')
        except NotFoundError:
            try:
                response = self.client.inference.put(
                    task_type="text_embedding",
                    inference_id=self.inference_id,
                    body={
                        "service": self.config.get("service", "openai"),
                        "service_settings": self.config.get("service_settings", {})
                    }
                )
                logging.debug(f'创建推理服务成功: {self.inference_id}')
            except Exception as e:
                logging.error(f"创建推理服务失败: {e}")
                raise


class Collection:
    """集合（知识库）抽象，对应一个ES索引"""
    
    def __init__(self, client: Client, name: str, user: User, inference_config: Optional[Dict] = None):
        self.client = client
        self.name = name
        self.user = user
        self.index_name = f"{user.username}__{name}"
        self.pipeline_id = f"{self.index_name}__pipeline"
        
        # 配置推理服务
        if inference_config:
            self.inference = client.get_inference(
                inference_config["model_id"],
                inference_config
            )
        else:
            self.inference = None
            
        self._init_collection()
    
    def _init_collection(self):
        """初始化集合和pipeline"""
        self._init_pipeline()
        self._init_index()
    
    def _init_index(self):
        """初始化索引"""
        if self.client.client.indices.exists(index=self.index_name):
            return
        
        # 获取嵌入向量维度
        embedding_dims = 384  # 默认值
        if self.inference:
            embedding_dims = self.inference.get_embedding_dims()
            
        mapping = {
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
                                "dims": embedding_dims,
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
                    "data": {"type": "text", "index": False}
                }
            },
            "settings": {
                "index": {
                    "default_pipeline": self.pipeline_id,
                    "number_of_replicas": 0,
                }
            }
        }
        
        try:
            self.client.client.indices.create(index=self.index_name, body=mapping)
            logging.debug(f"创建索引成功: {self.index_name}")
        except Exception as e:
            logging.error(f"创建索引失败: {e}")
            raise
    
    def _init_pipeline(self):
        """初始化处理pipeline"""
        try:
            self.client.client.ingest.get_pipeline(id=self.pipeline_id)
            logging.debug(f'Pipeline已存在: {self.pipeline_id}')
            return
        except NotFoundError:
            pass
            
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
            {
                "script": {
                    "id": "text_splitter",
                    "params": {
                        "splitter_config": {
                            "chunk_size": 1000,
                            "chunk_overlap": 100,
                        }
                    }
                }
            }
        ]
        
        # 如果配置了推理服务，添加向量化处理器
        if self.inference:
            processors.extend([
                {
                    "foreach": {
                        "if": "ctx?.model_id != null",
                        "field": "chunks",
                        "processor": {
                            "inference": {
                                "model_id": self.inference.inference_id,
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
                },
                {
                    "remove": {
                        "if": "ctx?.model_id != null",
                        "field": "model_id",
                        "ignore_missing": True
                    }
                }
            ])
        
        try:
            response = self.client.client.ingest.put_pipeline(
                id=self.pipeline_id,
                body={
                    "description": f"Processing pipeline for {self.name}",
                    "processors": processors
                }
            )
            logging.debug(f'创建Pipeline成功: {self.pipeline_id}')
        except Exception as e:
            logging.error(f"创建Pipeline失败: {e}")
            raise
    
    def add(self, document_id: str, name: str, file_content: Optional[bytes] = None, 
            text_content: Optional[str] = None, metadata: Optional[Dict] = None, 
            timeout: int = 600) -> Dict:
        """
        添加文档到集合
        
        Args:
            document_id: 文档ID
            name: 文档名称
            file_content: 文件内容（二进制）
            text_content: 文本内容
            metadata: 元数据
            timeout: 超时时间
        """
        if not file_content and not text_content:
            raise ValueError("必须提供 file_content 或 text_content")
        
        doc_data = {
            "name": name,
            "metadata": metadata or {},
        }
        
        # 如果有推理服务，添加model_id触发向量化
        if self.inference:
            doc_data["model_id"] = self.inference.inference_id
        
        if file_content:
            # 处理文件内容
            doc_data["data"] = base64.b64encode(file_content).decode()
        elif text_content:
            # 处理文本内容
            doc_data["attachment"] = {"content": text_content}
        
        try:
            response = self.client.client.index(
                index=self.index_name,
                id=document_id,
                body=doc_data,
                timeout=f"{timeout}s",
                refresh='wait_for'  # 确保文档立即可见
            )
            return response
        except Exception as e:
            logging.error(f"添加文档失败: {e}")
            raise
    
    async def query(self, query_text: str, metadata_filter: Optional[Dict] = None, 
                   size: int = 5, include_embedding: bool = True) -> List[Dict]:
        """
        查询集合中的相关文档
        
        Args:
            query_text: 查询文本
            metadata_filter: 元数据过滤条件
            size: 返回结果数量
            include_embedding: 是否包含向量搜索
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
        
        # 文本搜索
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
            "size": size * 2,  # 获取更多结果用于RRF合并
            "_source": ["name", "metadata"],
        }
        
        # 执行搜索任务
        searches = []
        search_results = []
        
        # 文本搜索任务
        async def text_search():
            return await self.client.async_client.search(
                index=self.index_name,
                body=text_search_body
            )
        
        searches.append(text_search())
        
        # 向量搜索（如果配置了推理服务）
        if include_embedding and self.inference:
            vector_search_body = {
                "knn": {
                    "filter": filter_conditions,
                    "field": "chunks.embedding",
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": self.inference.inference_id,
                            "model_text": query_text,
                        }
                    },
                    "k": size * 2,  # 获取更多结果用于RRF合并
                    "num_candidates": size * 10,
                    "inner_hits": {
                        "_source": ["chunks.content", "chunks.metadata"],
                        "size": size,
                    }
                },
                "size": size * 2,
                "_source": ["name", "metadata"],
            }
            
            async def vector_search():
                return await self.client.async_client.search(
                    index=self.index_name,
                    body=vector_search_body
                )
            
            searches.append(vector_search())
        
        # 执行所有搜索
        responses = await asyncio.gather(*searches, return_exceptions=True)
        
        # 处理结果并准备RRF合并
        search_results = []
        all_chunk_data = {}  # 存储所有chunk的详细信息
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logging.warning(f"搜索执行失败: {response}")
                continue
            
            search_type = "text" if i == 0 else "vector"
            chunk_results = []
            
            for doc in response['hits']['hits']:
                if 'inner_hits' in doc and 'chunks' in doc['inner_hits']:
                    for chunk in doc['inner_hits']['chunks']['hits']['hits']:
                        chunk_key = f"{doc['_id']}_{chunk['_nested']['offset']}"
                        chunk_results.append((chunk_key, chunk['_score']))
                        
                        # 存储chunk的详细信息
                        if chunk_key not in all_chunk_data:
                            all_chunk_data[chunk_key] = {
                                'document_id': doc['_id'],
                                'document_name': doc['_source'].get('name', ''),
                                'chunk_content': chunk['_source'].get('content', ''),
                                'chunk_metadata': chunk['_source'].get('metadata', {}),
                                'score': chunk['_score'],
                                'document_metadata': doc['_source'].get('metadata', {}),
                                'search_type': search_type
                            }
            
            search_results.append(chunk_results)
        
        # 如果只有一个搜索结果，直接返回
        if len(search_results) == 1:
            merged_results = search_results[0]
        elif len(search_results) > 1:
            # 使用RRF算法合并结果
            merged_results = rrf(*search_results, k=60)
        else:
            merged_results = []
        
        # 构建最终结果
        final_results = []
        for chunk_key, rrf_score in merged_results[:size]:
            if chunk_key in all_chunk_data:
                result = all_chunk_data[chunk_key].copy()
                result['rrf_score'] = rrf_score
                # 如果有RRF分数，也可以用作主要分数
                result['final_score'] = rrf_score
                final_results.append(result)
        
        return final_results
    
    def get(self, document_id: str) -> Optional[Dict]:
        """获取指定文档"""
        try:
            response = self.client.client.get(
                index=self.index_name,
                id=document_id
            )
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            logging.error(f"获取文档失败: {e}")
            raise
    
    def delete(self, document_id: str) -> bool:
        """删除指定文档"""
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
            logging.error(f"删除文档失败: {e}")
            raise
    
    def list_documents(self, offset: int = 0, limit: int = 10) -> Dict:
        """列出集合中的文档"""
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
        except Exception as e:
            logging.error(f"列出文档失败: {e}")
            raise
    
    def drop(self):
        """删除整个集合"""
        try:
            if self.client.client.indices.exists(index=self.index_name):
                self.client.client.indices.delete(index=self.index_name)
                logging.debug(f"删除索引成功: {self.index_name}")
            
            try:
                self.client.client.ingest.delete_pipeline(id=self.pipeline_id)
                logging.debug(f"删除Pipeline成功: {self.pipeline_id}")
            except NotFoundError:
                pass
        except Exception as e:
            logging.error(f"删除集合失败: {e}")
            raise


# RRF算法实现
def rrf(*queries, k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion算法"""
    ranks = [{d[0]: i + 1 for i, d in enumerate(q)} for q in queries]
    result = {}
    for rank in ranks:
        for d in rank.keys():
            result[d] = (result[d] if d in result else 0) + 1.0 / (k + rank[d])
    return sorted(result.items(), key=lambda kv: kv[1], reverse=True)


# 命令行参数处理
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    import os
    import asyncio

    def usage():
        print("Usage:")
        print("  python rag.py setup")
        print("  python rag.py list_users")
        print("  python rag.py list_collections")
        print("  python rag.py list_documents")
        print("  python rag.py add <file_path>")
        print("  python rag.py search <query>")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        usage()
    
    command = sys.argv[1]
    
    async def main():
        # 创建客户端
        client = Client('http://0.0.0.0:9200')
        # 创建推理服务配置
        inference_config = {
            "model_id": "bge-small-en-v1.5",
            "service": "hugging_face",
            "service_settings": {
                "api_key": "placeholder",
                "url": "http://192.168.9.62:8080/embed",
            },
            "embedding_dims": 384
        }
        
        if command == "setup":
            # 初始化用户
            success = client.add_user('test_user', 'test_api_key', metadata={
                "email": "test@test.com",
                "role": "admin",
                "preferences": {
                    "language": "zh",
                    "theme": "dark"
                }
            })
            if success:
                print("用户初始化成功")
            else:
                print("用户初始化失败")
        elif command == "list_users":
            # 列出所有用户
            try:
                users_info = User.list_all_users(client.client, "user_auth")
                print(f"用户列表 (共 {users_info['total']} 个用户):")
                print("=" * 50)
                for user in users_info['users']:
                    print(f"用户名: {user['username']}")
                    print(f"创建时间: {user['created_at']}")
                    print(f"最后登录: {user['last_login'] or '从未登录'}")
                    if user['metadata']:
                        print(f"元数据: {user['metadata']}")
                        print("-" * 30)
            except Exception as e:
                print(f"列出用户失败: {e}")
        elif command == "list_collections":
            # 列出用户的所有集合
            try:
                # 用户认证
                user = client.authenticate('test_user', 'test_api_key')
                collections = client.list_collections()
                print(f"用户 {user.username} 的集合列表 (共 {len(collections)} 个集合):")
                print("=" * 50)
                for collection in collections:
                    print(f"集合名: {collection['name']}")
                    print(f"索引名: {collection['index']}")
                    print(f"健康状态: {collection['health']}")
                    print(f"状态: {collection['status']}")
                    print(f"文档数量: {collection['doc_count']}")
                    print(f"存储大小: {collection['store_size']}")
                    print("-" * 30)
            except Exception as e:
                print(f"列出集合失败: {e}")
        elif command == "list_documents":
            # 列出指定集合中的文档
            collection_name = "test_documents"  # 默认集合名
            if len(sys.argv) >= 3:
                collection_name = sys.argv[2]
            
            try:
                # 用户认证
                user = client.authenticate('test_user', 'test_api_key')
                # 获取集合
                collection = client.get_collection(collection_name, inference_config)
                
                # 列出文档
                documents_info = collection.list_documents()
                print(f"集合 '{collection_name}' 中的文档列表 (共 {documents_info['total']} 个文档):")
                print("=" * 50)
                for doc in documents_info['documents']:
                    print(f"文档ID: {doc['id']}")
                    print(f"文档名: {doc['name']}")
                    if doc['metadata']:
                        print(f"元数据: {doc['metadata']}")
                        print("-" * 30)
            except Exception as e:
                print(f"列出文档失败: {e}")
        elif command == "add" and len(sys.argv) >= 3:
            # 添加文档
            file_path = sys.argv[2]
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return
                
            # 用户认证
            user = client.authenticate('test_user', 'test_api_key')
            
            # 获取集合
            collection = client.get_collection('test_documents', inference_config)
            
            # 添加文档
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
                    print(f"添加文档成功: {file_name}")
            except Exception as e:
                print(f"添加文档失败: {e}")
                
        elif command == "search" and len(sys.argv) >= 3:
            # 搜索文档
            query = " ".join(sys.argv[2:])
            
            # 用户认证
            user = client.authenticate('test_user', 'test_api_key')
            
            # 创建推理服务配置
            inference_config = {
                "model_id": "bge-small-en-v1.5",
                "service": "hugging_face",
                "service_settings": {
                    "api_key": "placeholder",
                    "url": "http://192.168.10.12:8080/embed",
                },
                "embedding_dims": 384
            }
            
            # 获取集合
            collection = client.get_collection('test_documents', inference_config)
            
            # 查询文档
            try:
                results = await collection.query(
                    query_text=query,
                    size=5
                )
                print(f"查询 '{query}' 的结果 ({len(results)} 个):")
                print("=" * 50)
                for i, result in enumerate(results, 1):
                    print(f"{i}. 文档: {result['document_name']}")
                    print(f"   内容: {result['chunk_content'][:200]}...")
                    print(f"   分数: {result.get('final_score', result['score']):.4f}")
                    print("-" * 30)
            except Exception as e:
                print(f"查询失败: {e}")
                
        else:
            print("无效的命令或参数")
            usage()
    
    asyncio.run(main())
