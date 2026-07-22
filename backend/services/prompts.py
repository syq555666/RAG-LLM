"""Agent 系统提示词和追问建议提示词"""

BASE_SYSTEM_PROMPT = """你是一个专业的智能客服助手。

## 你的能力
1. 回答用户问题
2. 使用工具获取实时信息
3. 搜索知识库获取相关文档

## 回答要求
1. 回答要简洁明了，用 Markdown 格式组织回答
2. 如果知识库没有找到相关信息，请明确告知用户
3. 如果需要使用工具，必须调用相关工具，不要假设结果
4. 不要重复调用同一个工具
5. 如果不确定信息，请如实告知用户"""

TOOL_INSTRUCTIONS = """
## 工具使用规则
- 当用户问时间、日期时 → 使用 get_current_time
- 当用户问知识库相关问题时 → 使用 search_knowledge_base
- 当用户问实时新闻、天气、股价等 → 使用 web_search
- 其他问题可以直接回答"""


def build_system_prompt(history: str = "") -> str:
    """构建 Agent 系统提示词，可选地包含历史对话上下文"""
    if history:
        return f"""{BASE_SYSTEM_PROMPT}

{TOOL_INSTRUCTIONS}

【历史对话】
{history}

现在开始回答用户问题。"""
    else:
        return f"""{BASE_SYSTEM_PROMPT}

{TOOL_INSTRUCTIONS}

现在开始回答用户问题。"""


def build_suggestions_prompt(query: str, response: str, history: str = "") -> str:
    """构建追问建议的提示词"""
    context = f"\n历史对话：{history}" if history else ""
    return f"""基于以下对话，生成 3 个用户可能会追问的相关问题。

用户问题：{query}
AI 回答：{response}{context}

要求：
1. 生成 3 个简洁的相关问题
2. 每个问题不超过 20 个字
3. 只输出问题，不要有其他内容
4. 用中文"""
