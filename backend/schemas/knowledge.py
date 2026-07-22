from pydantic import BaseModel, Field
from typing import Literal


class FileListResponse(BaseModel):
    files: list[str] = Field(default_factory=list)
    count: int = 0


class FileUploadResult(BaseModel):
    filename: str
    status: Literal["success", "error"]
    chunks_added: int = 0
    chunks_skipped: int = 0
    message: str = ""


class FileUploadResponse(BaseModel):
    results: list[FileUploadResult] = Field(default_factory=list)


class FileDeleteResponse(BaseModel):
    filename: str
    status: str
