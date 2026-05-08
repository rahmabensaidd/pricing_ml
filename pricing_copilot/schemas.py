from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CopilotConversationTurn(BaseModel):
    role: str
    content: str


class CopilotRequest(BaseModel):
    pricing_response: Optional[Dict[str, Any]] = None
    selected_question: str
    quote_request: Optional[Dict[str, Any]] = None
    pricing_api_request: Optional[Dict[str, Any]] = None
    selected_strategy: Optional[str] = None
    selected_model: Optional[str] = None
    selected_price: Optional[float] = None
    pricing_explanation: Optional[Dict[str, Any]] = None
    conversation_history: List[CopilotConversationTurn] = Field(default_factory=list)


class CopilotResponse(BaseModel):
    answer: str
    selected_model: Optional[str] = None
    recommended_price: Optional[float] = None
    confidence: str
    recommended_action: str


class CopilotAnalyzeResponse(BaseModel):
    selectedQuestion: str
    pricing: Dict[str, Any]
    copilot: CopilotResponse
