import time
import json

import pandas as pd
from agent import AgentService
from knowledge_base import KnowledgeBaseService
from file_history_store import SummarizingChatMessageHistory
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 历史存储路径
HISTORY_STORAGE_PATH = "./chat_history"

st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")

# 页面标题
st.title("🤖 智能客服")
st.divider()

# 公共函数
def get_kb_service():
    """获取知识库服务（单例）"""
    if "kb_service" not in st.session_state:
        st.session_state["kb_service"] = KnowledgeBaseService()
    return st.session_state["kb_service"]


def get_file_list(kb_service):
    """获取文件列表"""
    try:
        collection = kb_service.chroma.get()
        metadatas = collection.get('metadatas', []) if collection else []
        return set(m.get('source') for m in metadatas if m and m.get('source'))
    except Exception:
        return set()


def read_file(uploaded_file) -> str:
    """读取文件内容"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file).to_string()
        elif uploaded_file.name.endswith('.json'):
            return json.dumps(json.load(uploaded_file), ensure_ascii=False)
        else:
            return uploaded_file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"❌ 文件读取失败: {e}")
        return None


def get_session_id():
    """获取 session_id，每个用户/会话使用独立的 ID"""
    if "session_id" not in st.session_state:
        # 使用时间戳和随机数生成唯一 session_id
        import uuid
        st.session_state["session_id"] = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    return st.session_state["session_id"]


# 侧边栏 - 知识库管理
with st.sidebar:
    st.markdown("📚 **知识库管理**")

    # 显示知识库状态
    try:
        kb_service = get_kb_service()
        unique_files = len(get_file_list(kb_service))
        st.info(f"📊 当前知识库包含 {unique_files} 个文件")
    except Exception as e:
        st.info("📊 当前知识库: 0 个文件")

    # 上传成功后需要清空文件选择
    if "upload_key" not in st.session_state:
        st.session_state["upload_key"] = "file_uploader_0"

    # 上传成功后更新 key，强制清空文件选择
    if st.session_state.get("refresh_uploader"):
        st.session_state["upload_key"] = f"file_uploader_{int(time.time() * 1000)}"
        st.session_state["refresh_uploader"] = False

    uploaded_files = st.file_uploader(
        "📤 上传文档到知识库",
        type=['txt', 'md', 'csv', 'json'],
        accept_multiple_files=True,
        key=st.session_state["upload_key"],
        label_visibility="visible"
    )

    if uploaded_files:
        if st.button("🚀 批量上传", use_container_width=True):
            success_count = 0
            for uploaded_file in uploaded_files:
                st.write(f"📄 待上传: {uploaded_file.name}")
                content = read_file(uploaded_file)
                if content is not None:
                    try:
                        with st.spinner("🔄 上传中..."):
                            kb_service = get_kb_service()
                            kb_service.upload_by_str(content, uploaded_file.name)
                        success_count += 1
                    except Exception as e:
                        st.error(f"❌ 文件上传失败: {e}")

            if success_count > 0:
                st.session_state["refresh_uploader"] = True
                st.success(f"✅ 成功上传 {success_count} 个文件")
                time.sleep(1)
                st.rerun()

    # 刷新按钮
    if st.button("🔄 刷新知识库", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("🗑️ **文件管理**")

    # 显示文件列表
    # 检查是否有待显示的删除成功消息
    if st.session_state.get('show_delete_msg'):
        deleted = st.session_state.pop('show_delete_msg')
        st.success(f"✅ 已删除 {deleted}")
        time.sleep(0.5)
        st.rerun()

    try:
        kb_service = get_kb_service()
        file_names = get_file_list(kb_service)

        if file_names:
            for filename in file_names:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    if st.button("🗑️", key=f"del_{filename}"):
                        try:
                            kb_service.delete_by_filename(filename)
                            st.session_state['show_delete_msg'] = filename
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 删除失败: {str(e)}")
        else:
            st.caption("暂无文件")
    except Exception as e:
        st.caption("无法加载文件列表")

    # 底部提示
    st.caption("💡 支持 txt, md, csv, json")

# 主聊天区域
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    # 获取知识库的 vector_store 传给 Agent
    kb_service = get_kb_service()
    st.session_state["rag"] = AgentService(vector_store=kb_service.chroma)

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("💬 输入你的问题，我会从知识库中寻找答案...", disabled=False)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # Agent 检索（流式输出）
    try:
        # 获取 session_id 并创建历史记录实例（复用）
        session_id = get_session_id()
        history = SummarizingChatMessageHistory(session_id, HISTORY_STORAGE_PATH)
        history_str = history.get_context_for_llm()

        # 创建空的消息容器用于流式显示
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 使用简化版 stream（先执行工具，再流式输出）
            for chunk in st.session_state["rag"].stream(prompt, history=history_str):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)

            # 移除光标
            message_placeholder.markdown(full_response)

        # 保存回答到历史记录
        st.session_state["message"].append({"role": "assistant", "content": full_response})

        # 保存到历史记录（复用同一个 history 实例）
        history.add_messages([HumanMessage(content=prompt), AIMessage(content=full_response)])

        # 生成追问建议
        suggestions = st.session_state["rag"].generate_suggestions(prompt, full_response, history_str)
        if suggestions:
            st.markdown("**💡 你可能还想问：**")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with cols[i]:
                    if st.button(suggestion, key=f"suggest_{len(st.session_state['message'])}_{i}"):
                        st.session_state["suggestion"] = suggestion

    except Exception as e:
        st.error(f"❌ 检索失败: {str(e)}")

# 处理追问建议
if "suggestion" in st.session_state:
    suggestion = st.session_state.pop("suggestion")
    st.rerun()