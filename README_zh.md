# RAG SDK - 检索增强生成系统

一个强大的Python SDK，用于构建基于Elasticsearch的RAG（检索增强生成）系统，具有多模型支持、用户认证和混合搜索等高级功能。

## 特性

- **多模型支持**: 支持多种嵌入模型（OpenAI、Hugging Face等）
- **用户认证**: 内置用户管理和API密钥认证
- **混合搜索**: 结合文本搜索和向量搜索，使用RRF（倒数排名融合）算法
- **文档处理**: 自动文本分片和嵌入向量生成
- **模板化架构**: 预定义索引模板，高效扩展
- **集合管理**: 将文档组织到具有特定模型配置的集合中

## 核心概念

### Client（客户端）
与RAG系统交互的主要入口点。管理Elasticsearch连接、用户认证和模型注册。

### User（用户）
处理用户认证和授权。每个用户可以拥有多个集合并维护自己的文档空间。

### Model（模型）
表示一个嵌入模型配置。每个模型创建自己的推理服务、处理管道和索引模板以获得最佳性能。

### Collection（集合）
属于用户并使用特定模型的知识库。集合存储和组织文档，并自动生成嵌入向量。

## 快速开始

### 1. 安装

```bash
pip install elasticsearch asyncio
```

### 2. 初始化客户端

```python
from rag import Client

# 连接到Elasticsearch
client = Client('http://localhost:9200')
```

### 3. 设置用户和模型

```python
# 添加用户
client.add_user('john_doe', 'secure_api_key', metadata={
    "email": "john@example.com",
    "role": "user"
})

# 用户认证
user = client.authenticate('john_doe', 'secure_api_key')

# 注册模型
model_config = {
    "service": "hugging_face",
    "service_settings": {
        "api_key": "your_hf_token",
        "url": "http://your-embedding-service:8080/embed",
    },
    "dimensions": 384
}
client.register_model("bge-small-en-v1.5", model_config)
```

### 4. 创建集合并添加文档

```python
# 获取使用特定模型的集合
collection = client.get_collection("my_documents", model_id="bge-small-en-v1.5")

# 从文件添加文档
with open("document.pdf", "rb") as f:
    collection.add(
        document_id="doc1",
        name="重要文档",
        file_content=f.read(),
        metadata={"category": "research", "author": "John Doe"}
    )

# 从文本添加文档
collection.add(
    document_id="doc2", 
    name="文本文档",
    text_content="这是重要的内容...",
    metadata={"category": "notes"}
)

# 添加带有预处理块的文档
chunks = [
    {"content": "这是第一个块。", "metadata": {"index": 0}},
    {"content": "这是第二个块。", "metadata": {"index": 1}}
]
collection.add(
    document_id="doc3",
    name="分块文档",
    chunks=chunks,
    metadata={"category": "chunked"}
)
```

### 5. 搜索文档

```python
import asyncio

async def search_example():
    # 执行混合搜索（文本+向量）
    results = await collection.query(
        query_text="重要信息",
        metadata_filter={"category": "research"},
        size=5
    )
    
    for result in results:
        print(f"文档: {result['document_name']}")
        print(f"内容: {result['chunk_content'][:200]}...")
        print(f"分数: {result['final_score']:.4f}")
        print("-" * 50)

asyncio.run(search_example())
```

## API参考

### Client（客户端）

#### 方法

- `add_user(username, api_key, metadata=None)` - 添加或更新用户
- `authenticate(username, api_key)` - 用户认证
- `register_model(model_id, config)` - 注册模型
- `get_collection(collection_name, model_id=None)` - 获取或创建集合
- `list_models()` - 列出可用模型
- `list_collections()` - 列出用户的集合

### User（用户）

#### 方法

- `validate()` - 验证用户凭据
- `get_info()` - 获取用户信息
- `update_metadata(metadata)` - 更新用户元数据
- `delete()` - 删除用户

### Model（模型）

#### 属性

- `model_id` - 模型标识符
- `config` - 模型配置
- `inference_id` - Elasticsearch推理服务ID
- `pipeline_id` - 处理管道ID
- `template_name` - 索引模板名称

#### 方法

- `get_dimensions()` - 获取嵌入向量维度

### Collection（集合）

#### 方法

- `add(document_id, name, file_content=None, text_content=None, metadata=None, chunks=None)` - 添加文档
- `query(query_text, metadata_filter=None, size=5, include_embedding=True)` - 搜索文档
- `get(document_id)` - 获取特定文档
- `delete(document_id)` - 删除文档
- `list_documents(offset=0, limit=10)` - 列出文档
- `drop()` - 删除整个集合

## 高级用法

### 自定义模型配置

```python
# 注册OpenAI模型
openai_config = {
    "service": "openai",
    "service_settings": {
        "api_key": "your_openai_key",
        "model_id": "text-embedding-ada-002",
    },
    "dimensions": 1536
}
client.register_model("text-embedding-ada-002", openai_config)
```

### 使用不同模型的多个集合

```python
# 创建使用不同模型的集合
collection_bge = client.get_collection("documents_bge", model_id="bge-small-en-v1.5")
collection_openai = client.get_collection("documents_openai", model_id="text-embedding-ada-002")
```

### 元数据过滤

```python
# 使用元数据过滤器搜索
results = await collection.query(
    query_text="机器学习",
    metadata_filter={
        "category": ["research", "tutorial"],
        "author": "Jane Smith"
    }
)
```

## 命令行界面

SDK包含用于常见操作的CLI：

```bash
# 系统设置
python rag.py setup

# 列出模型
python rag.py list_models

# 列出集合
python rag.py list_collections

# 添加文档
python rag.py add document.pdf my_collection bge-small-en-v1.5

# 搜索文档
python rag.py search "重要查询" my_collection bge-small-en-v1.5
```

## 配置

### 环境变量

- `HF_API_KEY` - Hugging Face API密钥
- `BGE_MODEL_URL` - BGE模型服务URL
- `OPENAI_API_KEY` - OpenAI API密钥

### 索引命名约定

- 带模型的集合: `{model_id}__{username}__{collection_name}`
- 不带模型的集合: `{username}__{collection_name}`

## 系统要求

- Python 3.7+
- Elasticsearch 8.0+
- elasticsearch-py
- asyncio

## 许可证

MIT许可证

## 贡献

1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 添加测试
5. 提交拉取请求

## 支持

如有问题和支持需求，请在GitHub上提交issue。
