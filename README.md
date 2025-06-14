# ElasticRAG

ElasticRAG 是一个基于 Elasticsearch 的 RAG（Retrieval-Augmented Generation）系统，充分利用 Elasticsearch 的 ingest pipeline 功能来处理整个 RAG 工作流。

## 特性

- 🔍 基于 Elasticsearch 的向量搜索和文本搜索
- 🛠️ 使用 ingest pipeline 进行文档处理和向量化
- 👥 多用户支持和认证
- 🧠 多模型支持（OpenAI、HuggingFace 等）
- 📚 知识库（Collection）管理
- 🔄 混合搜索和 RRF（Reciprocal Rank Fusion）算法
- 📄 支持多种文档格式的文本分割
- ⚙️ 支持环境变量配置和命令行参数

## 安装

使用 uv 进行安装：

```bash
uv add elasticrag
```

或从源码安装：

```bash
git clone <repository-url>
cd elasticrag
uv sync
```

## 配置

### 环境变量配置

创建 `.env` 文件（从 `.env.example` 复制）：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=http://localhost:9200

# Authentication
ELASTICRAG_USERNAME=your_username
ELASTICRAG_API_KEY=your_api_key

# Text Embedding Service
TEXT_EMBEDDING_URL=http://your-embedding-service:8080/embed
TEXT_EMBEDDING_API_KEY=your_embedding_api_key
```

### 命令行参数

你也可以通过命令行参数覆盖环境变量：

```bash
elasticrag -h localhost:9200 -u admin -k secret setup
```

## 快速开始

### 1. 初始化系统

```bash
elasticrag setup
```

或使用自定义配置：

```bash
elasticrag -h localhost:9200 -u admin setup --embedding-url http://your-service:8080/embed
```

### 2. 列出可用模型

```bash
elasticrag list-models
```

### 3. 添加文档

```bash
elasticrag add document.pdf -c my_collection -m my_model
```

### 4. 搜索文档

```bash
elasticrag search "your query" -c my_collection -m my_model -s 10
```

## CLI 命令参考

### 全局选项

- `-h, --host`: Elasticsearch 主机地址
- `-u, --username`: 用户名
- `-k, --api-key`: API 密钥
- `-v, --verbose`: 启用详细日志

### 命令

- `setup`: 初始化系统
- `list-models`: 列出可用模型
- `list-users`: 列出所有用户
- `list-collections`: 列出所有集合
- `list-documents [collection] [model]`: 列出文档
- `add <file_path> [-c collection] [-m model]`: 添加文档
- `search <query> [-c collection] [-m model] [-s size]`: 搜索文档

## API 使用

```python
from elasticrag import Client

# 创建客户端
client = Client('http://localhost:9200')

# 认证用户
user = client.authenticate('username', 'api_key')

# 获取集合
collection = client.get_collection('my_collection', 'my_model')

# 添加文档
collection.add('doc_id', 'Document Name', text_content='Your content here')

# 搜索
results = await collection.query('your query')
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

# 代码格式化
uv run black .
uv run isort .
```

## 许可证

MIT License
