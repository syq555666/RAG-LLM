import csv
import io
import json


def read_file(uploaded_file) -> str | None:
    """读取上传文件内容，支持 txt, md, csv, json"""
    try:
        filename = getattr(uploaded_file, 'filename', '') or getattr(uploaded_file, 'name', '')
        if filename.endswith('.csv'):
            content = uploaded_file.file.read().decode('utf-8')
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if not rows:
                return ""
            # 格式化为简易表格
            col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
            lines = []
            for row in rows:
                lines.append("  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
            return "\n".join(lines)
        elif filename.endswith('.json'):
            return json.dumps(json.load(uploaded_file.file), ensure_ascii=False)
        else:
            content = uploaded_file.file.read()
            return content.decode('utf-8')
    except Exception as e:
        print(f"文件读取失败: {e}")
        return None
