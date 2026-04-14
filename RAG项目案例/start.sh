#!/bin/bash
# 使用 conda rag 环境的 Python 直接运行 Streamlit
cd "$(dirname "$0")"
/opt/anaconda3/envs/rag/bin/streamlit run app_qa.py
