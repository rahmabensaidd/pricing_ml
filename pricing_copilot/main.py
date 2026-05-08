import traceback
import logging

from fastapi import FastAPI, HTTPException

try:
    from pricing_copilot.graph import copilot_graph
    from pricing_copilot.interpreter import build_analysis_context
    from pricing_copilot.pricing_backend import build_explained_quote
    from pricing_copilot.schemas import CopilotAnalyzeResponse, CopilotRequest, CopilotResponse
except ModuleNotFoundError:
    from graph import copilot_graph
    from interpreter import build_analysis_context
    from pricing_backend import build_explained_quote
    from schemas import CopilotAnalyzeResponse, CopilotRequest, CopilotResponse

app = FastAPI(title="Pricing Copilot AI", version="1.0.0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@app.get("/health")
def health():
    return {"status": "ok", "service": "pricing-copilot"}


def run_copilot(pricing: dict, request: CopilotRequest) -> CopilotResponse:
    result = copilot_graph.invoke({
        "pricing_response": pricing.get("pricingDetails") or {},
        "selected_question": request.selected_question,
        "quote_request": request.quote_request or {},
        "pricing_api_request": request.pricing_api_request or {},
        "selected_strategy": pricing.get("selectedStrategy"),
        "selected_model": pricing.get("selectedModel"),
        "selected_price": pricing.get("selectedPrice"),
        "pricing_explanation": pricing.get("explanation") or {},
        "conversation_history": [turn.model_dump() for turn in request.conversation_history],
        "analysis_context": {},
        "intent": "analysis",
        "answer_source": "base_context",
        "matched_fields": [],
        "scenario_plan": [],
        "scenario_results": [],
        "answer": "",
    })

    context = result.get("analysis_context") or build_analysis_context(
        pricing_response=pricing.get("pricingDetails") or {},
        selected_question=request.selected_question,
        selected_strategy=pricing.get("selectedStrategy"),
        selected_model=pricing.get("selectedModel"),
        selected_price=pricing.get("selectedPrice"),
        pricing_explanation=pricing.get("explanation") or {},
    )
    best_model = context.get("best_model")

    return CopilotResponse(
        answer=result["answer"],
        selected_model=context.get("selected_model_name") or (best_model.get("model_name") if best_model else None),
        recommended_price=context.get("selected_price") or (best_model.get("prediction") if best_model else None),
        confidence=context.get("confidence", "LOW"),
        recommended_action=context.get("recommended_action", "MANUAL_REVIEW"),
    )


@app.post("/analyze-pricing", response_model=CopilotAnalyzeResponse)
def analyze_pricing(request: CopilotRequest):
    try:
        pricing = build_explained_quote(request.quote_request or {}, request.pricing_api_request or {})
        copilot = run_copilot(pricing, request)
        return CopilotAnalyzeResponse(
            selectedQuestion=request.selected_question,
            pricing=pricing,
            copilot=copilot,
        )
    except Exception as exc:
        print("COPILOT ANALYZE ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/interpret-pricing", response_model=CopilotResponse)
def interpret_pricing(request: CopilotRequest):
    try:
        pricing = {
            "pricingDetails": request.pricing_response or {},
            "selectedStrategy": request.selected_strategy,
            "selectedModel": request.selected_model,
            "selectedPrice": request.selected_price,
            "explanation": request.pricing_explanation or {},
        }
        return run_copilot(pricing, request)
    except Exception as exc:
        print("COPILOT ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
