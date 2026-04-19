# RAG-LLM 智能客服系统

基于 LangChain + 通义千问 + Chroma 向量数据库的 Agent 应用，支持文档上传、知识库检索、网络搜索。

## 功能特性

- 📚 **知识库管理** - 支持 txt、md、csv、json 格式文档批量上传
- 🗑️ **文件管理** - 支持查看和删除知识库中的文件
- 🔍 **Agent 智能问答** - 基于 ReAct 模式的 Agent，自动选择工具
- 🔎 **网络搜索** - 实时搜索互联网获取最新信息
- 📝 **会话历史** - 基于文件的历史记录存储，包含对话摘要
- 🔄 **去重机制** - MD5 精确去重 + SimHash 近似去重

## 环境要求

- Python 3.10+
- Anaconda (推荐)

## 安装

```bash
# 创建并激活 conda 环境
conda create -n rag python=3.10
conda activate rag

# 安装依赖
pip install -r requirements.txt

# 安装搜索依赖
pip install ddgs
```

## 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env`，填入你的通义千问 API Key：
```
DASHSCOPE_API_KEY=your_api_key_here
```

获取 API Key: [阿里云 DashScope](https://dashscope.console.aliyun.com/)

## 运行

```bash
streamlit run RAG项目案例/app_qa.py
```

浏览器访问 http://localhost:8501

## Agent 功能

系统基于 ReAct 模式实现 Agent，具备以下能力：

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| search_knowledge_base | 知识库搜索 | 产品信息、公司资料等 |
| web_search | 网络搜索 | 实时新闻、天气等 |

### ReAct 工作流程

```
问题: "今天天气怎么样"
思考: 分析问题，需要获取实时天气
行动: web_search
行动输入: 今天天气
观察: 今天北京晴，25度
最终答案: 今天天气晴朗，气温25度
```

## 项目结构

```
RAG-LLM/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
└── RAG项目案例/
    ├── app_qa.py              # Streamlit Web 界面
    ├── agent.py               # Agent 核心服务 (ReAct)
    ├── rag.py                 # RAG 核心服务
    ├── knowledge_base.py      # 知识库管理
    ├── vector_stores.py       # 向量存储 & 混合检索
    ├── file_history_store.py  # 会话历史存储
    └── config_data.py         # 配置文件

# 数据存储
├── chroma_db/               # Chroma 向量数据库
├── md5.text                 # MD5 精确去重索引
├── simhash_index.json       # SimHash 近似去重索引
└── chat_history/            # 会话历史记录
```

## 配置说明

可在 `config_data.py` 或环境变量中修改：

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| embedding_model | EMBEDDING_MODEL_NAME | text-embedding-v4 | embedding 模型 |
| chat_model | CHAT_MODEL_NAME | qwen3-max | 对话模型 |
| chunk_size | - | 1000 | 文本分块大小 |
| chunk_overlap | - | 100 | 分块重叠长度 |
| top_k | - | 3 | 检索返回数量 |
| score_threshold | - | 0.7 | 向量相似度阈值 |

## 技术栈

- **LLM**: 通义千问 (Qwen)
- **Embedding**: DashScope Text Embedding
- **Vector DB**: Chroma
- **UI**: Streamlit
- **Framework**: LangChain
- **搜索**: DuckDuckGo (ddgs)