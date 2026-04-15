from rag import RagService
from knowledge_base import KnowledgeBaseService
import streamlit as st
import config_data as config

# 炫酷渐变风主题配置
st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")

# 自定义 CSS - 炫酷渐变风
st.markdown("""
<style>
    /* 全局背景 - 统一深蓝色 */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stFooter"],
    [data-testid="stMain"], [data-testid="stMainContent"], footer, .stVerticalBlock,
    .stVerticalBlockBorderWrapper, [data-testid="stBottom"] {
        background: #0f172a !important;
        background-color: #0f172a !important;
    }

    /* 确保所有子元素也是深色背景 */
    .stApp > * {
        background: transparent !important;
    }

    /* 移除 Streamlit 默认的白色背景 */
    section, .stApp section {
        background: transparent !important;
    }

    .stDeployButton {
        display: none !important;
    }

    /* 标题样式 - 渐变文字 */
    h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        padding: 20px 0 !important;
        animation: gradientText 3s ease infinite !important;
        background-size: 200% auto !important;
    }

    @keyframes gradientText {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(0, 210, 255, 0.2) !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00d2ff !important;
    }

    /* 分割线 */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #00d2ff, transparent) !important;
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 210, 255, 0.5) !important;
    }

    /* 文件上传器 */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.08) 0%, rgba(138, 43, 226, 0.08) 100%) !important;
        border: 2px dashed rgba(0, 210, 255, 0.4) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 210, 255, 0.7) !important;
    }

    [data-testid="stFileUploader"] label {
        background: linear-gradient(90deg, #00d2ff, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    [data-testid="stFileUploader"] button[aria-label="Clear file"] {
        background: rgba(255, 82, 82, 0.2) !important;
        color: #ff5252 !important;
    }

    /* 聊天消息气泡 */
    [data-testid="stChatMessage"][aria-label="user"] {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.2) 0%, rgba(58, 123, 213, 0.2) 100%) !important;
        border-radius: 18px 18px 18px 4px !important;
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
    }

    [data-testid="stChatMessage"][aria-label="assistant"] {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.2) 0%, rgba(75, 0, 130, 0.2) 100%) !important;
        border-radius: 18px 18px 4px 18px !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
    }

    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #fff !important;
    }

    /* 加载动画 */
    [data-testid="stSpinner"] {
        border-color: rgba(0, 210, 255, 0.3) !important;
        border-top-color: #00d2ff !important;
    }

    /* 提示框 */
    .stAlert {
        background: rgba(0, 210, 255, 0.1) !important;
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
        border-radius: 12px !important;
    }

    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px !important;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05) !important;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d2ff, #3a7bd5) !important;
        border-radius: 4px !important;
    }

    /* 头像样式 */
    [data-testid="stChatMessageAvatar"] {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5) !important;
        border-radius: 50% !important;
    }

    /* 聊天输入框 */
    div[data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-radius: 16px !important;
        border: 2px solid rgba(0, 210, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    /* 输入框容器背景 */
    div[data-testid="stChatInput"] > div:first-child {
        background: transparent !important;
    }

    /* 聊天输入框区域背景 */
    [data-testid="stBottom"] {
        background: #0f172a !important;
    }

    [data-testid="stBottom"] > * {
        background: transparent !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #00d2ff !important;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.4) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #fff !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🤖 智能客服")
st.divider()

# 侧边栏 - 知识库管理
with st.sidebar:
    st.markdown("📚 **知识库管理**")

    uploaded_file = st.file_uploader(
        "📤 上传文档到知识库",
        type=['txt', 'md', 'csv', 'json'],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        # 读取文件内容
        if uploaded_file.name.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        elif uploaded_file.name.endswith('.json'):
            import json
            content = json.dumps(json.load(uploaded_file), ensure_ascii=False)
        else:
            content = uploaded_file.getvalue().decode('utf-8')

        if st.button("🚀 确认上传", use_container_width=True):
            with st.spinner("🔄 上传中..."):
                kb_service = KnowledgeBaseService()
                result = kb_service.upload_by_str(content, uploaded_file.name)
                st.success(f"✅ {result}")

    # 刷新按钮
    if st.button("🔄 刷新知识库", use_container_width=True):
        st.rerun()

    # 底部提示
    st.caption("💡 支持 txt, md, csv, json")

# 主聊天区域
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(f"""
        <div style="
            color: #fff;
            line-height: 1.6;
            padding: 5px 0;
        ">{message['content']}</div>
        """, unsafe_allow_html=True)

prompt = st.chat_input("💬 输入你的问题，我会从知识库中寻找答案...", disabled=False)

if prompt:
    with st.chat_message("user"):
        st.markdown(f"""
        <div style="
            color: #fff;
            line-height: 1.6;
        ">{prompt}</div>
        """, unsafe_allow_html=True)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("🤔 AI 正在思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        with st.chat_message("assistant"):
            full_response = st.write_stream(capture(res_stream, ai_res_list))

    st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})