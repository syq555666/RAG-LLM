import json
import pandas as pd


def read_file(uploaded_file) -> str | None:
    """读取上传文件内容，支持 txt, md, csv, json"""
    try:
        filename = getattr(uploaded_file, 'filename', '') or getattr(uploaded_file, 'name', '')
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file.file).to_string()
        elif filename.endswith('.json'):
            return json.dumps(json.load(uploaded_file.file), ensure_ascii=False)
        else:
            content = uploaded_file.file.read()
            return content.decode('utf-8')
    except Exception as e:
        print(f"文件读取失败: {e}")
        return None
