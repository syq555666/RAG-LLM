from rag import RagService
from knowledge_base import KnowledgeBaseService
import streamlit as st
import config_data as config

st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")

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
        st.markdown(message["content"])

prompt = st.chat_input("💬 输入你的问题，我会从知识库中寻找答案...", disabled=False)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
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