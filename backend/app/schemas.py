"""Modelos Pydantic para la API."""
from typing import List, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "error"]
    index_loaded: bool
    chunks: int
