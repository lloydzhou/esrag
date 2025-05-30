import asyncio
import logging
import base64
import os
from typing import Dict, List, Optional, Union, Any
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
        self._predefined_models = {}
        self._user = None
        self._init_scripts()
        self._load_existing_models()

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
    
    def register_model(self, model_id: str, config: Dict) -> 'Model':
        """注册预定义模型"""
        model = Model(
            client=self.client,
            model_id=model_id,
            config=config
        )
        self._predefined_models[model_id] = model
        return model

    def get_model(self, model_id: str, service_config: Optional[Dict] = None) -> 'Model':
        """获取或创建模型"""
        print(f"获取模型配置: {model_id}, {service_config}")
        if model_id not in self._predefined_models:
            if not service_config:
                raise ValueError(f"模型 {model_id} 未预定义，需要提供 service_config")
            self._predefined_models[model_id] = Model(
                client=self.client,
                model_id=model_id,
                config=service_config
            )
        return self._predefined_models[model_id]

    def list_models(self) -> List[Dict]:
        """列出可用模型"""
        return [
            {
                "model_id": model_id,
                "config": model.config,
                "dimensions": model.get_dimensions()
            }
            for model_id, model in self._predefined_models.items()
        ]

    def get_collection(self, name: str, model_id: Optional[str] = None) -> 'Collection':
        """获取或创建集合（知识库）"""
        if not self._user:
            raise ValueError("请先调用 authenticate() 进行用户认证")
        
        # 如果指定了model_id，使用该模型；否则使用默认模型
        if model_id:
            model = self.get_model(model_id)
            collection_key = f"{model_id}__{self._user.username}__{name}"
        else:
            model = None
            collection_key = f"{self._user.username}__{name}"
            
        if collection_key not in self._collections:
            self._collections[collection_key] = Collection(
                client=self,
                name=name,
                user=self._user,
                model=model
            )
        return self._collections[collection_key]
    
    def list_collections(self) -> List[str]:
        """列出用户的所有集合"""
        if not self._user:
            raise ValueError("请先调用 authenticate() 进行用户认证")
        
        try:
            pattern = f"*__{self._user.username}__*"
            response = self.client.cat.indices(index=pattern, format='json', ignore=[404])
            print(f"列出集合: {response}")
            if response:
                # 支持新的索引命名格式：{model_id}__{username}__{collection_name}
                prefix = f"{self._user.username}__"
                collections = []
                for idx in response:
                    if prefix in idx['index']:
                        # 解析索引名：{model_id}__{username}__{collection_name} 或 {username}__{collection_name}
                        index_parts = idx['index'].replace(prefix, "").split("__")
                        if len(index_parts) == 2:
                            model_id, collection_name = index_parts
                        else:
                            model_id = "default"
                            collection_name = "__".join(index_parts)
                        
                        collections.append({
                            "name": collection_name,
                            "model_id": model_id,
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

    def _load_existing_models(self):
        """从ES加载所有已存在的模型推理服务配置"""
        try:
            # 获取所有推理服务
            response = self.client.inference.get()
            for config in response.get('endpoints', {}):
                inference_id = config.get('inference_id', '')
                if inference_id.endswith('__inference'):
                    model_id = inference_id.replace('__inference', '')
                    # 如果不在预定义模型中，从配置重建
                    if model_id not in self._predefined_models:
                        service_config = {
                            "service": config.get('service', 'openai'),
                            "service_settings": config.get('service_settings', {}),
                            "dimensions": config.get('service_settings', {}).get('dimensions', 384)
                        }
                        model = Model(
                            client=self.client,
                            model_id=model_id,
                            config=service_config
                        )
                        # 标记为已存在，避免重复初始化
                        model._exists = True
                        self._predefined_models[model_id] = model
            logging.debug(f"加载了 {len(self._predefined_models)} 个模型")
        except Exception as e:
            logging.warning(f"加载已存在的模型失败: {e}")

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


class Model:
    """模型管理，负责推理服务、pipeline和index template"""
    
    def __init__(self, client: Elasticsearch, model_id: str, config: Dict):
        self.client = client
        self.model_id = model_id
        self.config = config
        self.inference_id = f"{model_id}__inference"
        self.pipeline_id = f"{model_id}__pipeline"
        self.template_name = f"{model_id}__template"
        self._exists = False
        
        # 初始化模型的三个组件
        self._init_inference()
        self._create_model_pipeline()
        self._create_index_template()
    
    def get_dimensions(self) -> int:
        """获取嵌入向量的维度"""
        return self.config.get("dimensions", 384)
    
    def _init_inference(self):
        """初始化推理服务"""
        if self._exists:
            return
        try:
            # self.client.inference.delete(inference_id=self.inference_id, ignore=[404], force=True)
            # raise NotFoundError("", None, None)
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

    def _create_model_pipeline(self):
        """创建模型专用的处理pipeline"""
        try:
            self.client.ingest.get_pipeline(id=self.pipeline_id)
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
            },
            {
                "foreach": {
                    "if": "ctx?.model_id != null",
                    "field": "chunks",
                    "processor": {
                        "inference": {
                            "model_id": self.inference_id,
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
        ]
        
        try:
            response = self.client.ingest.put_pipeline(
                id=self.pipeline_id,
                body={
                    "description": f"Processing pipeline for model {self.model_id}",
                    "processors": processors
                }
            )
            logging.debug(f'创建模型Pipeline成功: {self.pipeline_id}')
        except Exception as e:
            logging.error(f"创建模型Pipeline失败: {e}")
            raise

    def _create_index_template(self):
        """创建模型专用的索引模板"""
        try:
            self.client.indices.exists_template(name=self.template_name)
            logging.debug(f'索引模板已存在: {self.template_name}')
            # return
        except:
            pass

        dimensions = self.get_dimensions()
        
        template = {
            "index_patterns": [f"{self.model_id}__*"],
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
                                "dims": dimensions,
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
            self.client.indices.put_template(
                name=self.template_name,
                body=template
            )
            logging.debug(f'创建索引模板成功: {self.template_name}')
        except Exception as e:
            logging.error(f"创建索引模板失败: {e}")
            raise


class Collection:
    """集合（知识库）抽象，对应一个ES索引"""
    
    def __init__(self, client: Client, name: str, user: User, model: Optional[Model] = None):
        self.client = client
        self.name = name
        self.user = user
        self.model = model
        
        # 索引命名规则
        if model:
            self.index_name = f"{model.model_id}__{user.username}__{name}"
        else:
            self.index_name = f"{user.username}__{name}"
    
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
        
        # 如果有模型，添加model_id触发向量化
        if self.model:
            doc_data["model_id"] = self.model.inference_id
        
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
        
        # 向量搜索（如果配置了模型）
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
        except NotFoundError:
            return {"total": 0, "documents": []}
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
        print("  python rag.py list_models")
        print("  python rag.py list_collections")
        print("  python rag.py list_documents [collection_name] [model_id]")
        print("  python rag.py add <file_path> [collection_name] [model_id]")
        print("  python rag.py search <query> [collection_name] [model_id]")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        usage()
    
    command = sys.argv[1]
    
    async def main():
        # 创建客户端
        client = Client('http://0.0.0.0:9200')
        
        collection_name = "test_documents123"  # 默认集合名
        model_id = "bge-small-en-v1.5"  # 默认模型
        
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
            # BGE模型配置
            config = {
                "service": "hugging_face",
                "service_settings": {
                    "api_key": os.getenv("HF_API_KEY", "placeholder"),
                    "url": os.getenv("BGE_MODEL_URL", "http://192.168.9.62:8080/embed"),
                },
                "dimensions": 384
            }
            client.register_model(model_id, config)
            print("模型注册成功")
        elif command == "list_models":
            # 列出可用模型
            try:
                models = client.list_models()
                print(f"可用模型列表 (共 {len(models)} 个模型):")
                print("=" * 50)
                for model in models:
                    print(f"模型ID: {model['model_id']}")
                    print(f"服务类型: {model['config'].get('service', 'unknown')}")
                    print(f"向量维度: {model['dimensions']}")
                    print("-" * 30)
            except Exception as e:
                print(f"列出模型失败: {e}")
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
                    print(f"模型ID: {collection['model_id']}")
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
            if len(sys.argv) >= 3:
                collection_name = sys.argv[2]
            if len(sys.argv) >= 4:
                model_id = sys.argv[3]
            
            try:
                # 用户认证
                user = client.authenticate('test_user', 'test_api_key')
                # 获取集合
                collection = client.get_collection(collection_name, model_id)
                
                # 列出文档
                documents_info = collection.list_documents()
                print(f"集合 '{collection_name}' (模型: {model_id}) 中的文档列表 (共 {documents_info['total']} 个文档):")
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
            if len(sys.argv) >= 4:
                collection_name = sys.argv[3]
            if len(sys.argv) >= 5:
                model_id = sys.argv[4]
                
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return
                
            # 用户认证
            user = client.authenticate('test_user', 'test_api_key')
            
            # 获取集合（使用链式调用）
            collection = client.with_model(model_id).get_collection(collection_name)
            
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
                    print(f"添加文档成功: {file_name} (模型: {model_id})")
            except Exception as e:
                print(f"添加文档失败: {e}")
                
        elif command == "search" and len(sys.argv) >= 3:
            # 搜索文档
            query = sys.argv[2]
            if len(sys.argv) >= 4:
                collection_name = sys.argv[3]
            if len(sys.argv) >= 5:
                model_id = sys.argv[4]
            
            # 用户认证
            user = client.authenticate('test_user', 'test_api_key')

            # 获取集合
            collection = client.get_collection(collection_name, model_id)
            
            # 查询文档
            try:
                results = await collection.query(
                    query_text=query,
                    size=5
                )
                print(f"在集合 '{collection_name}' (模型: {model_id}) 中查询 '{query}' 的结果 ({len(results)} 个):")
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
