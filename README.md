# RAG-LLM 智能客服系统

基于 LangChain + 通义千问 + Chroma 向量数据库的 Agent 应用，支持文档上传、知识库检索、网络搜索。

## 功能特性

- 📚 **知识库管理** - 支持 txt、md、csv、json 格式文档批量上传（自动去重）
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
| search_knowledge_base | 知识库搜索 | 产品信息、公司资料、技术文档等 |
| web_search | 网络搜索 | 实时新闻、天气、股价等 |
| get_current_time | 获取时间 | 日期、时间相关问题 |

### 可扩展工具

以下工具可根据需求自行添加：

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| read_file | 文件读取 | 读取本地文件内容 |
| calculator | 数学计算 | 数学运算、折扣计算 |
| get_weather | 天气查询 | 查询城市天气 |
| fetch_url | URL抓取 | 读取网页内容 |
| search_history | 历史搜索 | 搜索历史对话记录 |


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

### 前端
- **UI 框架**: Streamlit
- **前端语言**: Python

### 后端
- **LLM**: 通义千问 (Qwen)
- **Embedding**: DashScope Text Embedding
- **向量数据库**: Chroma
- **开发框架**: LangChain
- **搜索**: DuckDuckGo (ddgs)
- **Python**: 3.10+

## 使用示例

### 1. 知识库问答
上传文档后，可以直接询问文档相关问题：
```
用户: 我们的退换货政策是什么？
Agent: 根据知识库中的文档，您的退换货政策是...
```

### 2. 网络搜索
询问实时信息时，Agent 会自动调用搜索工具：
```
用户: 今天北京天气怎么样？
Agent: (调用 web_search 后) 今天北京天气晴朗，气温 20-28°C...
```

### 3. 混合检索
结合知识库和网络搜索：
```
用户: 你们公司和竞争对手相比有什么优势？
Agent: (先搜索知识库，再搜索网络) 根据我的了解...
```

## 常见问题

### Q1: 如何更新知识库？
在 Web 界面中点击"上传文件"按钮，选择 txt、md、csv、json 格式的文档上传。

### Q2: 为什么搜索不到相关内容？
- 检查文档是否已成功上传
- 尝试调整 `score_threshold` 参数（当前默认 0.7）
- 检查文档内容是否与问题相关

### Q3: API 调用失败怎么办？
- 确认 `.env` 文件中的 `DASHSCOPE_API_KEY` 正确
- 检查网络连接
- 查看控制台错误日志

### Q4: 如何清空所有数据？
删除项目根目录下的 `chroma_db`、`md5.text`、`simhash_index.json` 文件即可。


## 许可证

MIT License

Copyright (c) 2024 RAG-LLM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.