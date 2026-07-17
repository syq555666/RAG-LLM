from fastapi import APIRouter, UploadFile, File, Depends
from api.deps import get_kb_service
from schemas.knowledge import FileListResponse, FileUploadResponse, FileUploadResult, FileDeleteResponse
from utils.file_reader import read_file

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/files", response_model=FileListResponse)
def list_files():
    """列出知识库中所有文件"""
    kb = get_kb_service()
    files = kb.get_file_list()
    return FileListResponse(files=sorted(files), count=len(files))


@router.post("/upload", response_model=FileUploadResponse)
def upload_files(files: list[UploadFile] = File(...)):
    """批量上传文件到知识库"""
    kb = get_kb_service()
    results = []

    for uploaded_file in files:
        try:
            content = read_file(uploaded_file)
            if content is None:
                results.append(FileUploadResult(
                    filename=uploaded_file.filename,
                    status="error",
                    message="文件读取失败"
                ))
                continue

            msg = kb.upload_by_str(content, uploaded_file.filename)

            # 解析返回消息
            if "【成功】" in msg:
                results.append(FileUploadResult(
                    filename=uploaded_file.filename,
                    status="success",
                    message=msg
                ))
            elif "【跳过】" in msg:
                results.append(FileUploadResult(
                    filename=uploaded_file.filename,
                    status="success",
                    chunks_added=0,
                    chunks_skipped=0,
                    message=msg
                ))
            else:
                results.append(FileUploadResult(
                    filename=uploaded_file.filename,
                    status="success",
                    message=msg
                ))
        except Exception as e:
            results.append(FileUploadResult(
                filename=uploaded_file.filename,
                status="error",
                message=str(e)
            ))

    return FileUploadResponse(results=results)


@router.delete("/files/{filename:path}", response_model=FileDeleteResponse)
def delete_file(filename: str):
    """从知识库中删除文件"""
    kb = get_kb_service()
    kb.delete_by_filename(filename)
    return FileDeleteResponse(filename=filename, status="deleted")
