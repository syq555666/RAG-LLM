"""Agent 工具定义：时间查询、网络搜索、知识库搜索"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# DDGS 单例（复用连接池，避免每次搜索重建）
_ddgs = None


def _get_ddgs():
    """获取 DDGS 单例（懒加载）"""
    global _ddgs
    if _ddgs is None:
        from ddgs import DDGS
        _ddgs = DDGS()
    return _ddgs


@tool
def get_current_time() -> str:
    """获取当前日期和时间。当你不知道今天是几号、现在几点时，必须使用此工具查询。不需要任何输入参数。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M')}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当用户询问实时新闻、天气、股价等实时信息时使用。query 是搜索关键词。"""
    try:
        ddgs = _get_ddgs()
        results = ddgs.text(query, max_results=3)
        if results:
            formatted = []
            for r in results:
                title = r.get('title', '')
                link = r.get('href') or r.get('url', '')
                formatted.append(f"- {title}: {link}")
            return "\n".join(formatted)
        return "未找到相关信息"
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return f"网络搜索失败: {str(e)}"


def create_rag_tool(retriever):
    """创建知识库搜索工具（闭包捕获 retriever 实例）"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """搜索知识库获取相关信息。当用户问技术问题、文档相关内容时使用。query 是搜索关键词。"""
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "知识库中没有找到相关内容"
            results = []
            for i, doc in enumerate(docs[:3], 1):
                results.append(f"相关文档 {i}: {doc.page_content[:200]}...")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return f"知识库搜索失败: {str(e)}"

    return search_knowledge_base


# 基础工具列表（不包含知识库搜索，它需要 retriever 参数动态创建）
BASE_TOOLS = [get_current_time, web_search]
