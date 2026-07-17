from pydantic import BaseModel, Field
from typing import Optional


class SessionInfo(BaseModel):
    id: str
    preview: Optional[str] = None
    updated_at: str = ""


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session_id: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage] = Field(default_factory=list)
    summary: Optional[str] = None
