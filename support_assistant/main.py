from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.rag import ask_question


app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline RAG-based Zepto policy support assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Customer support question",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": True,
        "endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    response = ask_question(request.query)

    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
    )