from pydantic import BaseModel, Field
from typing import Literal, Optional


class StreamChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(..., description="用户消息")


class SuggestionsRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    query: str = Field(..., description="用户问题")
    response: str = Field(..., description="AI 回答")


class SuggestionsResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list, description="追问建议列表")
