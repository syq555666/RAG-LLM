# RAG-LLM 智能客服系统

基于 **LangChain + DeepSeek + Chroma** 的 RAG 智能客服，支持知识库检索、工具调用、流式对话。前后端分离架构。

## 功能

- 📚 **知识库管理** — 上传/删除文档，自动分块入库（txt、md、csv、json、pdf）
- 🔍 **混合检索** — 向量相似度 + BM25 关键词 + RRF 融合，兼顾语义和关键词匹配
- 🔄 **智能去重** — MD5 精确去重 + SimHash 近似去重
- 🤖 **Agent 对话** — DeepSeek 驱动，自动选择和调用工具
- 🌐 **网络搜索** — DuckDuckGo 实时搜索互联网信息
- ⚡ **SSE 流式输出** — 逐字返回，实时显示工具调用过程
- 💬 **对话管理** — 多会话切换，历史持久化 + LLM 自动摘要
- 🎨 **现代 UI** — React 19 + TypeScript，ChatGPT 风格界面

## 架构

```
frontend (React 19 + Vite)     backend (FastAPI + LangChain)     外部服务
    :5173                           :8000
    ┌──────────┐    SSE/HTTP     ┌──────────────┐    API Call    ┌──────────┐
    │ Chat UI  │ ◄──────────────►│ AgentService │ ◄─────────────►│ DeepSeek │
    │ KB Mgmt  │                 │   ├─ Retriever│               │ DashScope│
    │ Sessions │                 │   ├─ Tools    │               │ DuckDuckGo│
    └──────────┘                 │   └─ Chroma   │               └──────────┘
                                 └──────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+ / npm 9+

### 1. 安装后端依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `backend/.env`，确保以下配置正确：

```env
DEEPSEEK_API_KEY=你的key          # DeepSeek API Key
DASHSCOPE_API_KEY=你的key         # 阿里云 DashScope (Embedding)
```

### 3. 安装前端依赖

```bash
cd frontend && npm install
```

### 4. 启动

**终端 1 — 后端 :8000**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**终端 2 — 前端 :5173**

```bash
cd frontend
npm run dev
```

浏览器访问 **http://localhost:5173**

> API 文档：http://localhost:8000/docs

## 项目结构

```
RAG-LLM/
├── backend/
│   ├── main.py                          # FastAPI 入口
│   ├── config_data.py                   # 全局配置
│   ├── api/
│   │   ├── deps.py                      # 依赖注入（服务单例）
│   │   ├── chat.py                      # SSE 流式对话 + 追问建议
│   │   ├── knowledge.py                 # 知识库文件 CRUD
│   │   ├── sessions.py                  # 会话管理
│   │   └── health.py                    # 健康检查
│   ├── services/
│   │   ├── agent_service.py             # Agent（ReAct + 工具调用）
│   │   ├── knowledge_base_service.py    # 知识库（去重 + 入库）
│   │   ├── hybrid_retriever.py          # 混合检索（向量 + BM25 + RRF）
│   │   └── history_store.py             # 对话历史（JSON + LLM 摘要）
│   ├── schemas/                         # Pydantic 请求/响应模型
│   └── utils/
│       └── file_reader.py               # 多格式文件解析
├── frontend/
│   └── src/
│       ├── App.tsx                       # 根布局
│       ├── components/
│       │   ├── chat/                     # ChatContainer、MessageList、ChatInput …
│       │   ├── knowledge/                # FileUploader、FileList …
│       │   ├── layout/Sidebar.tsx        # 侧边栏
│       │   └── common/                   # Modal、Spinner
│       ├── hooks/                        # useChatStream、useSession …
│       ├── store/                        # Zustand（chatStore、kbStore）
│       ├── api/                          # HTTP 客户端 + SSE 解析
│       └── types/                        # TypeScript 类型
└── requirements.txt                      # Python 依赖
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 + 知识库状态 |
| `POST` | `/api/chat/stream` | SSE 流式对话 |
| `POST` | `/api/chat/suggestions` | 追问建议 |
| `GET` | `/api/knowledge/files` | 知识库文件列表 |
| `POST` | `/api/knowledge/upload` | 批量上传文件 |
| `DELETE` | `/api/knowledge/files/{name}` | 删除文件 |
| `GET` | `/api/sessions` | 会话列表 |
| `POST` | `/api/sessions` | 创建会话 |
| `GET` | `/api/sessions/{id}/history` | 会话历史 |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |

## Agent 工具

| 工具 | 触发条件 |
|------|----------|
| `search_knowledge_base` | 用户询问知识库相关内容 |
| `web_search` | 用户询问实时新闻、天气、股价等 |
| `get_current_time` | 用户询问日期或时间 |

## 可配置项

在 `backend/config_data.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 1000 | 文本分块大小 |
| `chunk_overlap` | 100 | 分块重叠长度 |
| `top_k` | 3 | 检索返回数量 |
| `score_threshold` | 0.7 | 向量相似度阈值 |
| `HISTORY_SUMMARY_THRESHOLD` | 10 | 触发对话摘要的轮数 |

## 技术栈

| 层 | 技术 |
|----|------|
| LLM | DeepSeek (`deepseek-chat`) via langchain-deepseek |
| Embedding | 阿里云 DashScope (`text-embedding-v4`) |
| 向量库 | Chroma（本地持久化） |
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | React 19 + TypeScript 6 |
| 构建工具 | Vite 8 |
| 状态管理 | Zustand 5 |
| Markdown 渲染 | react-markdown + remark-gfm |

## License

MIT
