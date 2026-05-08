from copy import deepcopy
from typing import Any, Dict, List, Optional, TypedDict
import json
import logging
import math
import os
import re
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

try:
    from pricing_copilot.interpreter import build_analysis_context
except ModuleNotFoundError:
    from interpreter import build_analysis_context

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger("pricing_copilot")


class CopilotState(TypedDict):
    pricing_response: Dict[str, Any]
    selected_question: str
    quote_request: Dict[str, Any]
    pricing_api_request: Dict[str, Any]
    selected_strategy: Optional[str]
    selected_model: Optional[str]
    selected_price: Optional[float]
    pricing_explanation: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    analysis_context: Dict[str, Any]
    intent: str
    answer_source: str
    matched_fields: List[str]
    scenario_plan: List[Dict[str, Any]]
    scenario_results: List[Dict[str, Any]]
    answer: str


LLM_PROVIDER = os.getenv("PRICING_COPILOT_PROVIDER", "deepseek").strip().lower()
LLM_API_KEY = ""
LLM_MODEL = ""
LLM_BASE_URL = ""
PRICING_API_BASE_URL = os.getenv("PRICING_API_BASE_URL", "http://localhost:8000").rstrip("/")

if LLM_PROVIDER == "xai":
    LLM_MODEL = os.getenv("PRICING_COPILOT_MODEL", "grok-2-latest")
    LLM_API_KEY = os.getenv("XAI_API_KEY", "")
    LLM_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
elif LLM_PROVIDER == "groq":
    LLM_MODEL = os.getenv("PRICING_COPILOT_MODEL", "llama-3.3-70b-versatile")
    LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
elif LLM_PROVIDER == "gemini":
    LLM_MODEL = os.getenv("PRICING_COPILOT_MODEL", "gemini-1.5-flash")
    LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
else:
    LLM_MODEL = os.getenv("PRICING_COPILOT_MODEL", "deepseek-chat")
    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

llm = None
if LLM_API_KEY:
    if LLM_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=LLM_API_KEY,
            temperature=0.2,
            max_retries=1,
        )
    else:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=45,
            max_retries=2,
        )
else:
    logger.warning("No API key found for provider '%s'. Pricing Copilot will use fallback answers.", LLM_PROVIDER)


FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "production_page": {
        "aliases": ["page", "pages", "production page", "production pages", "nb pages", "nombre de pages"],
        "kind": "numeric",
        "label": "production pages",
        "default_percent": 10.0,
        "default_step": 8.0,
        "min": 1.0,
    },
    "quantity": {
        "aliases": ["quantity", "qty", "volume", "run size", "tirage", "quantite", "quantite"],
        "kind": "numeric",
        "label": "quantity",
        "default_percent": 10.0,
        "default_step": 50.0,
        "min": 1.0,
    },
    "height": {
        "aliases": ["height", "hauteur"],
        "kind": "numeric",
        "label": "height",
        "default_percent": 5.0,
        "default_step": 5.0,
        "min": 1.0,
    },
    "width": {
        "aliases": ["width", "largeur"],
        "kind": "numeric",
        "label": "width",
        "default_percent": 5.0,
        "default_step": 5.0,
        "min": 1.0,
    },
    "thickness": {
        "aliases": ["thickness", "spine thickness", "epaisseur", "epaisseur"],
        "kind": "numeric",
        "label": "thickness",
        "default_percent": 10.0,
        "default_step": 0.2,
        "min": 0.1,
    },
    "shrinkwrap": {
        "aliases": ["shrinkwrap", "shrink wrap"],
        "kind": "boolean",
        "label": "shrinkwrap",
    },
    "has_insert": {
        "aliases": ["insert", "inserts"],
        "kind": "boolean",
        "label": "insert",
    },
    "has_tab": {
        "aliases": ["tab", "tabs"],
        "kind": "boolean",
        "label": "tab",
    },
    "has_coil": {
        "aliases": ["coil", "spiral"],
        "kind": "boolean",
        "label": "coil",
    },
    "has_backcover": {
        "aliases": ["backcover", "back cover"],
        "kind": "boolean",
        "label": "back cover",
    },
    "double_sided_cover": {
        "aliases": ["double sided cover", "double-sided cover"],
        "kind": "boolean",
        "label": "double-sided cover",
    },
}

GENERAL_SENSITIVITY_FIELDS = ["production_page", "quantity", "width", "height", "thickness"]
SCENARIO_KEYWORDS = [
    "if",
    "what if",
    "impact",
    "increase",
    "decrease",
    "change",
    "raise",
    "lower",
    "more",
    "less",
    "augmente",
    "augmenter",
    "baisse",
    "diminue",
]
CLIENT_KEYWORDS = ["client", "customer", "elasticity", "elasticite", "sensitivity"]
FORMULA_KEYWORDS = ["formula", "equation", "linear formula", "formula of the price", "coefficient"]
DRIVER_KEYWORDS = ["driver", "most important feature", "main feature", "drives the price", "drives price", "influences the price", "influence the price"]
ACTION_KEYWORDS = ["what action", "recommended action", "what do you recommend", "recommendation"]
CONFIDENCE_KEYWORDS = ["confidence", "reliability", "how reliable", "why low", "why medium", "why high"]
WHY_PRICE_KEYWORDS = ["why this price", "why this quote", "explain this price", "explain the price", "why this amount", "why this value"]
MODEL_SELECTION_KEYWORDS = ["why was this model selected", "why this model", "why was this strategy selected", "why this strategy", "why linear strategy", "why couple linear"]
APPROVAL_KEYWORDS = ["approve", "approval", "reliable enough to approve", "can i approve", "should i approve"]
CLIENT_PRICE_LEVEL_KEYWORDS = ["high or low for this client", "high for this client", "low for this client", "compare to the client average", "above the client average", "below the client average"]
AMBIGUOUS_REFERENCES = ["this", "that", "it", "them", "those", "these", "same", "again"]


def analyze_pricing(state: CopilotState) -> CopilotState:
    state["analysis_context"] = build_analysis_context(
        pricing_response=state["pricing_response"],
        selected_question=state["selected_question"],
        selected_strategy=state.get("selected_strategy"),
        selected_model=state.get("selected_model"),
        selected_price=state.get("selected_price"),
        pricing_explanation=state.get("pricing_explanation") or {},
    )
    return state


def detect_intent(state: CopilotState) -> CopilotState:
    question = normalize_text(state.get("selected_question", ""))
    history_text = normalize_text(build_history_text(state.get("conversation_history", [])))
    plan = classify_question(question, history_text)
    state["intent"] = plan["intent"]
    state["answer_source"] = plan["answer_source"]
    state["matched_fields"] = plan["matched_fields"]
    return state


def prepare_scenarios(state: CopilotState) -> CopilotState:
    run_driver_probe = state.get("intent") == "driver" and should_probe_driver_via_scenarios(state.get("analysis_context", {}))
    should_rerun = state.get("answer_source") == "pricing_rerun"
    if not should_rerun and not run_driver_probe:
        state["scenario_plan"] = []
        return state

    pricing_api_request = deepcopy(state.get("pricing_api_request") or {})
    features = pricing_api_request.get("features") or {}
    if not features:
        state["scenario_plan"] = []
        return state

    question = normalize_text(state.get("selected_question", ""))
    history_text = normalize_text(build_history_text(state.get("conversation_history", [])))
    target_source = question
    if is_ambiguous_follow_up(question) and not find_target_fields(question):
        target_source = f"{history_text} {question}".strip()

    targets = state.get("matched_fields") or find_target_fields(target_source)
    direction = detect_direction(target_source)
    percent_hint = parse_percent_hint(question) or parse_percent_hint(history_text)
    absolute_hint = parse_absolute_hint(question)

    plan: List[Dict[str, Any]] = []
    if not targets:
        targets = GENERAL_SENSITIVITY_FIELDS

    for target in targets:
        spec = FIELD_SPECS.get(target)
        if not spec:
            continue

        current_value = features.get(target)
        scenario = build_scenario_for_field(
            field_name=target,
            spec=spec,
            current_value=current_value,
            direction=direction,
            percent_hint=percent_hint,
            absolute_hint=absolute_hint,
        )
        if scenario:
            plan.append(scenario)

    state["scenario_plan"] = plan
    return state


def execute_scenarios(state: CopilotState) -> CopilotState:
    if not state.get("scenario_plan"):
        state["scenario_results"] = []
        return state

    context = state.get("analysis_context", {})
    base_price = context.get("selected_price")
    pricing_api_request = deepcopy(state.get("pricing_api_request") or {})
    base_features = pricing_api_request.get("features") or {}
    results: List[Dict[str, Any]] = []

    for scenario in state["scenario_plan"]:
        request_payload = deepcopy(pricing_api_request)
        request_payload.setdefault("features", {})
        request_payload["features"][scenario["field_name"]] = scenario["new_value"]

        scenario_response = call_pricing_api(request_payload)
        if not scenario_response:
            continue

        scenario_context = build_analysis_context(
            pricing_response=scenario_response,
            selected_question=state["selected_question"],
            pricing_explanation={},
        )
        scenario_price = scenario_context.get("selected_price")

        results.append({
            **scenario,
            "pricing_response": scenario_response,
            "analysis_context": scenario_context,
            "scenario_price": scenario_price,
            "base_price": base_price,
            "delta_price": compute_price_delta(base_price, scenario_price),
            "delta_percent": compute_percent_delta(base_price, scenario_price),
            "base_feature_value": base_features.get(scenario["field_name"]),
        })

    state["scenario_results"] = results
    return state


def generate_fallback_answer(state: CopilotState) -> str:
    context = state.get("analysis_context", {})
    intent = state.get("intent")
    scenario_results = state.get("scenario_results", [])

    if intent == "formula":
        return build_formula_answer(context)

    if intent == "driver":
        return build_driver_answer(context, scenario_results)

    if intent == "model_selection":
        return build_model_selection_answer(context)

    if intent == "approval":
        return build_approval_answer(context)

    if intent == "client":
        return build_client_answer(context)

    if intent == "client_price_level":
        return build_client_price_level_answer(context)

    if intent == "action":
        return build_action_answer(context)

    if intent == "confidence":
        return build_confidence_answer(context)

    if scenario_results:
        return build_scenario_answer(context, scenario_results, state.get("selected_question", ""))

    return build_why_price_answer(context)


def build_standard_analysis_answer(context: Dict[str, Any]) -> str:
    lines = [
        "AI ANALYSIS",
        "",
        "Pricing summary",
        f"- Selected price: {format_number(context.get('selected_price'))}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        f"- Confidence: {context.get('confidence')}",
        f"- Recommended action: {context.get('recommended_action')}",
    ]

    formula = context.get("formula")
    if formula:
        lines.extend(["", "Formula", f"- {formula}"])

    top_drivers = context.get("top_drivers") or []
    if top_drivers:
        lines.extend(["", "Key price drivers"])
        for driver in top_drivers[:5]:
            lines.append(f"- {driver.get('name')}: {format_percent(driver.get('importance'))}")

    client_summary = summarize_client_context(context.get("client_features"))
    if client_summary:
        lines.extend(["", "Client context", client_summary])

    return "\n".join(lines)


def build_why_price_answer(context: Dict[str, Any]) -> str:
    lines = [
        "AI ANALYSIS",
        "",
        "Why this price",
        f"- Selected price: {format_number(context.get('selected_price'))}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
    ]

    formula = context.get("formula")
    if formula:
        lines.append("- A linear pricing formula is available for this selected model.")

    strongest_term = get_strongest_formula_term(context.get("formula"))
    if strongest_term:
        lines.append(
            f"- The strongest formula coefficient is {strongest_term['feature']} ({format_signed_coefficient(strongest_term['coefficient'])})."
        )

    top_drivers = context.get("top_drivers") or []
    if top_drivers:
        driver = top_drivers[0]
        lines.append(f"- Top driver in the pricing explanation: {driver.get('name')} ({format_percent(driver.get('importance'))}).")

    lines.extend([
        f"- Confidence: {context.get('confidence')}",
        f"- Recommended action: {context.get('recommended_action')}",
    ])

    client_line = build_client_one_liner(context.get("client_features"))
    if client_line:
        lines.extend(["", "Client context", f"- {client_line}"])

    if formula:
        lines.extend(["", "Formula", f"- {formula}"])

    return "\n".join(lines)


def build_driver_answer(context: Dict[str, Any], scenario_results: List[Dict[str, Any]]) -> str:
    top_drivers = context.get("top_drivers") or []
    if top_drivers:
        strongest = top_drivers[0]
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Top driver",
            f"- The strongest driver returned by the pricing explanation is {strongest.get('name')} ({format_percent(strongest.get('importance'))}).",
            f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
            f"- Selected price: {format_number(context.get('selected_price'))}",
            f"- Confidence: {context.get('confidence')}",
        ])

    strongest_term = get_strongest_formula_term(context.get("formula"))
    if strongest_term:
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Top driver",
            f"- Based on the linear formula, the largest absolute coefficient is {strongest_term['feature']} ({format_signed_coefficient(strongest_term['coefficient'])}).",
            "- That makes it the strongest direct linear driver among the published formula terms.",
            f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
            f"- Selected price: {format_number(context.get('selected_price'))}",
        ])

    if scenario_results:
        strongest_result = max(
            scenario_results,
            key=lambda result: abs(float(result.get("delta_percent") or 0.0)),
        )
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Top driver",
            f"- I estimated the strongest driver by rerunning comparison scenarios.",
            f"- The largest movement came from {strongest_result.get('field_label')} with {format_signed_percent(strongest_result.get('delta_percent'))} on price.",
            f"- Base selected price: {format_number(context.get('selected_price'))}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        ])

    return build_standard_analysis_answer(context)


def build_scenario_answer(
    context: Dict[str, Any],
    scenario_results: List[Dict[str, Any]],
    selected_question: str,
) -> str:
    if not scenario_results:
        return build_why_price_answer(context)

    normalized_question = normalize_text(selected_question)
    if len(scenario_results) == 1:
        result = scenario_results[0]
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Scenario result",
            f"- Question: {selected_question}",
            f"- Changed field: {result.get('field_label')}",
            f"- Old value: {format_value(result.get('old_value'))}",
            f"- New value: {format_value(result.get('new_value'))}",
            f"- Base selected price: {format_number(result.get('base_price'))}",
            f"- New predicted price: {format_number(result.get('scenario_price'))}",
            f"- Price delta: {format_signed_number(result.get('delta_price'))}",
            f"- Price delta percent: {format_signed_percent(result.get('delta_percent'))}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
            f"- Confidence: {context.get('confidence')}",
        ])

    if "most" in normalized_question or "biggest" in normalized_question or "strongest" in normalized_question:
        strongest_result = max(
            scenario_results,
            key=lambda result: abs(float(result.get("delta_percent") or 0.0)),
        )
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Scenario comparison",
            f"- I reran pricing scenarios to answer this question.",
            f"- The biggest impact came from {strongest_result.get('field_label')}.",
            f"- Its simulated price movement was {format_signed_percent(strongest_result.get('delta_percent'))} ({format_signed_number(strongest_result.get('delta_price'))}).",
            f"- Base selected price: {format_number(context.get('selected_price'))}",
            f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        ])

    lines = [
        "AI ANALYSIS",
        "",
        "Scenario comparison",
        f"- I reran pricing because this question depends on a changed configuration.",
    ]
    for result in scenario_results:
        lines.append(
            f"- {result.get('field_label')}: {format_value(result.get('old_value'))} -> {format_value(result.get('new_value'))}, "
            f"price {format_number(result.get('base_price'))} -> {format_number(result.get('scenario_price'))} "
            f"({format_signed_percent(result.get('delta_percent'))})."
        )
    lines.extend([
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        f"- Confidence: {context.get('confidence')}",
    ])
    return "\n".join(lines)


def build_formula_answer(context: Dict[str, Any]) -> str:
    formula = context.get("formula")
    if formula:
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Formula",
            f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
            f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
            f"- Selected price: {format_number(context.get('selected_price'))}",
            f"- Formula: {formula}",
            f"- Confidence: {context.get('confidence')}",
            f"- Recommended action: {context.get('recommended_action')}",
        ])

    explanation_type = context.get("explanation_type") or "n/a"
    return "\n".join([
        "AI ANALYSIS",
        "",
        "Formula",
        f"- No explicit linear formula is available for the selected model {context.get('selected_model_name') or 'n/a'}.",
        f"- Explanation type returned by pricing: {explanation_type}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Confidence: {context.get('confidence')}",
        f"- Recommended action: {context.get('recommended_action')}",
    ])


def build_model_selection_answer(context: Dict[str, Any]) -> str:
    selected_strategy = context.get("selected_strategy") or "n/a"
    selected_model = context.get("selected_model_name") or "n/a"
    r2 = extract_r2(context.get("best_model"))

    reason = "The selected model matched the active pricing selection rule."
    if selected_strategy == "COUPLE_LINEAR":
        reason = "The selected model was chosen because the client-specific linear model is prioritized when it is available and its quality threshold is met."
    elif selected_strategy == "FAMILY_LINEAR":
        reason = "The selected model was chosen because the family linear model was the best available linear option for this quote."
    elif selected_strategy == "GLOBAL":
        reason = "The selected model was chosen because no stronger client or family linear option qualified for this quote."

    lines = [
        "AI ANALYSIS",
        "",
        "Model selection",
        f"- Selected strategy: {selected_strategy}",
        f"- Selected model: {selected_model}",
        f"- {reason}",
    ]
    if r2 is not None:
        lines.append(f"- Model R2: {r2:.4f}")
    if context.get("formula"):
        lines.append("- A linear formula is available for this selected model.")
    lines.append(f"- Selected price: {format_number(context.get('selected_price'))}")
    return "\n".join(lines)


def build_action_answer(context: Dict[str, Any]) -> str:
    return "\n".join([
        "AI ANALYSIS",
        "",
        "Recommended action",
        f"- Action: {context.get('recommended_action')}",
        f"- Confidence: {context.get('confidence')}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        f"- Selected price: {format_number(context.get('selected_price'))}",
    ])


def build_approval_answer(context: Dict[str, Any]) -> str:
    confidence = str(context.get("confidence") or "LOW").upper()
    action = context.get("recommended_action") or "MANUAL_REVIEW"

    decision_line = "This quote is not strong enough for direct approval."
    if confidence == "HIGH" and action == "APPROVE":
        decision_line = "This quote looks reliable enough to approve."
    elif confidence == "MEDIUM":
        decision_line = "This quote is usable, but it should still be reviewed before approval."

    lines = [
        "AI ANALYSIS",
        "",
        "Approval decision",
        f"- {decision_line}",
        f"- Confidence: {confidence}",
        f"- Recommended action: {action}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
        f"- Selected price: {format_number(context.get('selected_price'))}",
    ]
    r2 = extract_r2(context.get("best_model"))
    if r2 is not None:
        lines.append(f"- Model R2: {r2:.4f}")
    return "\n".join(lines)


def build_confidence_answer(context: Dict[str, Any]) -> str:
    r2 = extract_r2(context.get("best_model"))
    lines = [
        "AI ANALYSIS",
        "",
        "Confidence",
        f"- Confidence level: {context.get('confidence')}",
        f"- Selected strategy: {context.get('selected_strategy') or 'n/a'}",
        f"- Selected model: {context.get('selected_model_name') or 'n/a'}",
    ]
    if r2 is not None:
        lines.append(f"- Model R2: {r2:.4f}")
    if context.get("formula"):
        lines.append("- A linear formula is available, which improves interpretability.")
    lines.append(f"- Recommended action: {context.get('recommended_action')}")
    return "\n".join(lines)


def build_client_answer(context: Dict[str, Any]) -> str:
    client_features = context.get("client_features")
    if not isinstance(client_features, dict) or not client_features.get("client_found"):
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Client context",
            "- No client-specific profile was available for this quote.",
        ])

    metadata = client_features.get("metadata") if isinstance(client_features.get("metadata"), dict) else {}
    elasticity = first_number(
        client_features.get("elasticity"),
        metadata.get("client_price_elasticity"),
    )
    avg_price = first_number(
        client_features.get("avg_price_ht"),
        metadata.get("client_avg_price_ht"),
    )
    seniority = first_number(
        client_features.get("seniority_years"),
        metadata.get("client_seniority_years"),
    )
    orders = metadata.get("client_nb_orders")

    lines = [
        "AI ANALYSIS",
        "",
        "Client context",
        f"- Client code: {client_features.get('siren') or 'n/a'}",
        f"- Client found: {bool(client_features.get('client_found'))}",
    ]
    if elasticity is not None:
        lines.append(f"- Price elasticity: {elasticity:.4f}")
    if avg_price is not None:
        lines.append(f"- Average historical price HT: {avg_price:.4f}")
    if seniority is not None:
        lines.append(f"- Seniority in years: {seniority:.2f}")
    if isinstance(orders, (int, float)):
        lines.append(f"- Historical orders: {int(orders)}")
    lines.append(f"- Selected price: {format_number(context.get('selected_price'))}")
    return "\n".join(lines)


def build_client_price_level_answer(context: Dict[str, Any]) -> str:
    client_features = context.get("client_features")
    if not isinstance(client_features, dict) or not client_features.get("client_found"):
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Client comparison",
            "- No client-specific profile was available for this quote, so I cannot say whether the price is high or low for this client.",
        ])

    metadata = client_features.get("metadata") if isinstance(client_features.get("metadata"), dict) else {}
    avg_price = first_number(
        client_features.get("avg_price_ht"),
        metadata.get("client_avg_price_ht"),
    )
    selected_price = first_number(context.get("selected_price"))
    if avg_price is None or selected_price is None:
        return "\n".join([
            "AI ANALYSIS",
            "",
            "Client comparison",
            "- The client profile exists, but the average historical price is missing.",
        ])

    delta = selected_price - avg_price
    delta_percent = (delta / avg_price * 100.0) if avg_price else None
    level = "above"
    if delta < 0:
        level = "below"
    elif abs(delta) < 1e-9:
        level = "equal to"

    lines = [
        "AI ANALYSIS",
        "",
        "Client comparison",
        f"- This quote is {level} the client's historical average price.",
        f"- Selected price: {selected_price:.4f}",
        f"- Client average historical price HT: {avg_price:.4f}",
    ]
    if delta_percent is not None:
        lines.append(f"- Difference vs client average: {delta_percent:+.2f}%")
    elasticity = first_number(
        client_features.get("elasticity"),
        metadata.get("client_price_elasticity"),
    )
    if elasticity is not None:
        lines.append(f"- Client price elasticity: {elasticity:.4f}")
    return "\n".join(lines)


def generate_answer(state: CopilotState) -> CopilotState:
    context = state["analysis_context"]
    deterministic_answer = generate_fallback_answer(state)
    scenario_summary = summarize_scenarios_for_prompt(state.get("scenario_results", []))
    history_summary = summarize_history_for_prompt(state.get("conversation_history", []))
    client_context = summarize_client_context(context.get("client_features"))
    intent = state.get("intent", "analysis")
    answer_source = state.get("answer_source", "base_context")

    if intent in {"formula", "driver", "model_selection", "approval", "client", "client_price_level", "action", "confidence", "scenario"}:
        state["answer"] = deterministic_answer
        return state

    prompt = f"""
You are a Pricing Copilot for a book-printing e-commerce administrator.

Detected intent:
{state.get("intent", "analysis")}

Current admin question:
{state["selected_question"]}

Conversation history:
{history_summary}

Base pricing interpretation context:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}

Client context:
{client_context}

Scenario comparison results:
{scenario_summary}

Instructions:
- Respond in English.
- Keep the tone operational and concise.
- Do not answer with scenario simulations unless scenario results are available or the admin explicitly asked a what-if question.
- Use the base pricing request and explanation whenever the question can be answered from the current quote JSON.
- Trigger a rerun only when the question depends on a modified configuration or a hypothetical change.
- If the intent is formula and a formula is available, show it exactly in a dedicated Formula section.
- If the intent is analysis, explain the current selected price using the selected strategy, selected model, formula, and top drivers.
- If the intent is driver, identify the strongest current driver from the selected explanation.
- If client context is limited, say that explicitly instead of inventing evidence.
- Always mention selected strategy, selected model, selected price, confidence, and recommended action.
- Use short sections and bullet points.
- If the admin asks a precise question, answer that exact point first instead of returning a full generic summary.
- Do not repeat the whole client JSON unless the admin specifically asked for raw client details.
- Answer source selected by the system: {answer_source}
"""

    if llm is None:
        state["answer"] = deterministic_answer
        return state

    try:
        response = llm.invoke(prompt)
        content = (response.content or "").strip()
        if not content:
            raise ValueError("LLM returned an empty response.")
        state["answer"] = content
    except Exception as exc:
        logger.exception("LLM call failed, using fallback answer: %s", exc)
        state["answer"] = deterministic_answer
    return state


def build_graph():
    graph = StateGraph(CopilotState)
    graph.add_node("analyze_pricing", analyze_pricing)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("prepare_scenarios", prepare_scenarios)
    graph.add_node("execute_scenarios", execute_scenarios)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("analyze_pricing")
    graph.add_edge("analyze_pricing", "detect_intent")
    graph.add_conditional_edges(
        "detect_intent",
        lambda state: (
            "prepare_scenarios"
            if state.get("answer_source") == "pricing_rerun"
            or (state.get("intent") == "driver" and should_probe_driver_via_scenarios(state.get("analysis_context", {})))
            else "generate_answer"
        ),
        {
            "prepare_scenarios": "prepare_scenarios",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("prepare_scenarios", "execute_scenarios")
    graph.add_edge("execute_scenarios", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def build_history_text(history: List[Dict[str, str]]) -> str:
    user_turns = [turn.get("content", "") for turn in history[-8:] if turn.get("role") == "user"]
    return " ".join(user_turns)


def summarize_history_for_prompt(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous turns."
    formatted = []
    for turn in history[-10:]:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "").strip()
        if content:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted) if formatted else "No previous turns."


def summarize_client_context(client_features: Any) -> str:
    if not isinstance(client_features, dict):
        return "No client context."
    return json.dumps(client_features, ensure_ascii=False, indent=2, default=str)


def summarize_scenarios_for_prompt(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No scenario reruns were needed."

    lines = []
    for result in results:
        lines.append(
            json.dumps(
                {
                    "label": result.get("label"),
                    "field": result.get("field_name"),
                    "old_value": result.get("old_value"),
                    "new_value": result.get("new_value"),
                    "base_price": result.get("base_price"),
                    "scenario_price": result.get("scenario_price"),
                    "delta_price": result.get("delta_price"),
                    "delta_percent": result.get("delta_percent"),
                    "scenario_confidence": result.get("analysis_context", {}).get("confidence"),
                    "scenario_action": result.get("analysis_context", {}).get("recommended_action"),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def has_scenario_signal(text: str) -> bool:
    return any(keyword in text for keyword in SCENARIO_KEYWORDS)


def should_probe_driver_via_scenarios(context: Dict[str, Any]) -> bool:
    return not (context.get("top_drivers") or get_strongest_formula_term(context.get("formula")))


def classify_question(question: str, history_text: str) -> Dict[str, Any]:
    matched_fields = find_target_fields(question)

    if any(keyword in question for keyword in FORMULA_KEYWORDS):
        return {"intent": "formula", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in DRIVER_KEYWORDS):
        return {"intent": "driver", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in MODEL_SELECTION_KEYWORDS):
        return {"intent": "model_selection", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in APPROVAL_KEYWORDS):
        return {"intent": "approval", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in CLIENT_PRICE_LEVEL_KEYWORDS):
        return {"intent": "client_price_level", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in ACTION_KEYWORDS):
        return {"intent": "action", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in CONFIDENCE_KEYWORDS):
        return {"intent": "confidence", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in WHY_PRICE_KEYWORDS):
        return {"intent": "analysis", "answer_source": "base_context", "matched_fields": matched_fields}

    if any(keyword in question for keyword in CLIENT_KEYWORDS):
        if has_scenario_signal(question) or matched_fields:
            return {"intent": "scenario", "answer_source": "pricing_rerun", "matched_fields": matched_fields}
        return {"intent": "client", "answer_source": "base_context", "matched_fields": matched_fields}

    if has_scenario_signal(question) or matched_fields:
        return {"intent": "scenario", "answer_source": "pricing_rerun", "matched_fields": matched_fields}

    if is_ambiguous_follow_up(question) and (has_scenario_signal(history_text) or find_target_fields(history_text)):
        return {
            "intent": "scenario",
            "answer_source": "pricing_rerun",
            "matched_fields": matched_fields or find_target_fields(history_text),
        }

    return {"intent": "analysis", "answer_source": "base_context", "matched_fields": matched_fields}


def is_ambiguous_follow_up(text: str) -> bool:
    compact = text.strip()
    if len(compact) <= 20:
        return True
    return any(token in compact for token in AMBIGUOUS_REFERENCES)


def find_target_fields(text: str) -> List[str]:
    matches: List[str] = []
    for field_name, spec in FIELD_SPECS.items():
        if any(alias in text for alias in spec.get("aliases", [])):
            matches.append(field_name)
    return matches


def detect_direction(text: str) -> str:
    if any(keyword in text for keyword in ["decrease", "lower", "reduce", "less", "baisse", "diminuer", "diminue"]):
        return "decrease"
    if any(keyword in text for keyword in ["remove", "without", "disable"]):
        return "disable"
    if any(keyword in text for keyword in ["add", "with", "enable"]):
        return "enable"
    return "increase"


def parse_percent_hint(text: str) -> Optional[float]:
    match = re.search(r"(-?\\d+(?:\\.\\d+)?)\\s*%", text)
    return float(match.group(1)) if match else None


def parse_absolute_hint(text: str) -> Optional[float]:
    match = re.search(r"(-?\\d+(?:\\.\\d+)?)\\s*(?:page|pages|mm|cm|unit|units|copy|copies)?", text)
    if not match:
        return None
    value = float(match.group(1))
    return None if value == 0 else value


def build_scenario_for_field(
    field_name: str,
    spec: Dict[str, Any],
    current_value: Any,
    direction: str,
    percent_hint: Optional[float],
    absolute_hint: Optional[float],
) -> Optional[Dict[str, Any]]:
    kind = spec.get("kind")
    if current_value is None:
        return None

    if kind == "boolean":
        old_value = int(current_value)
        if direction in {"disable", "decrease"}:
            new_value = 0
        else:
            new_value = 1
        if new_value == old_value:
            new_value = 1 - old_value
        label = f"{'Enable' if new_value == 1 else 'Disable'} {spec['label']}"
        return {
            "field_name": field_name,
            "field_label": spec["label"],
            "old_value": old_value,
            "new_value": new_value,
            "label": label,
        }

    try:
        numeric_value = float(current_value)
    except (TypeError, ValueError):
        return None

    signed_percent = percent_hint if percent_hint is not None else spec.get("default_percent", 10.0)
    signed_absolute = absolute_hint if absolute_hint is not None else spec.get("default_step", 1.0)
    multiplier = -1.0 if direction == "decrease" else 1.0

    if percent_hint is not None:
        next_value = numeric_value * (1.0 + multiplier * (abs(signed_percent) / 100.0))
    else:
        step = abs(signed_absolute)
        if numeric_value > 0 and spec.get("default_percent"):
            step = max(step, numeric_value * (spec["default_percent"] / 100.0))
        next_value = numeric_value + (multiplier * step)

    minimum = spec.get("min", 0.0)
    next_value = max(minimum, next_value)
    if isinstance(current_value, int):
        next_value = int(round(next_value))
    else:
        next_value = round(next_value, 4)

    if next_value == current_value:
        next_value = current_value + (1 if isinstance(current_value, int) else 0.1)

    action = "Decrease" if direction == "decrease" else "Increase"
    return {
        "field_name": field_name,
        "field_label": spec["label"],
        "old_value": current_value,
        "new_value": next_value,
        "label": f"{action} {spec['label']}",
    }


def call_pricing_api(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = f"{PRICING_API_BASE_URL}/predict"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=45) as response:
            raw_body = response.read().decode("utf-8")
            return json.loads(raw_body) if raw_body else {}
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.exception("Scenario pricing call failed for %s: %s", payload, exc)
        return None


def compute_price_delta(base_price: Optional[float], scenario_price: Optional[float]) -> Optional[float]:
    if base_price is None or scenario_price is None:
        return None
    return round(float(scenario_price) - float(base_price), 6)


def compute_percent_delta(base_price: Optional[float], scenario_price: Optional[float]) -> Optional[float]:
    if base_price in (None, 0) or scenario_price is None:
        return None
    return round(((float(scenario_price) - float(base_price)) / float(base_price)) * 100.0, 4)


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.4f}"


def format_signed_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):+.4f}"


def format_signed_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):+.2f}%"


def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.2f}%"


def format_signed_coefficient(value: float) -> str:
    return f"{float(value):+.4f}"


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.4f}"
    return str(value)


def parse_formula_terms(formula: Any) -> List[Dict[str, Any]]:
    if not isinstance(formula, str) or "=" not in formula:
        return []

    rhs = formula.split("=", 1)[1]
    matches = re.findall(r"([+-]?\s*\d+(?:\.\d+)?)\s*[×x*]\s*([A-Za-z0-9_./-]+)", rhs)
    terms: List[Dict[str, Any]] = []
    for coefficient_text, feature_name in matches:
        normalized = coefficient_text.replace(" ", "")
        try:
            coefficient = float(normalized)
        except ValueError:
            continue
        terms.append({"feature": feature_name, "coefficient": coefficient})
    return terms


def get_strongest_formula_term(formula: Any) -> Optional[Dict[str, Any]]:
    terms = parse_formula_terms(formula)
    if not terms:
        return None
    return max(terms, key=lambda term: abs(float(term["coefficient"])))


def first_number(*values: Any) -> Optional[float]:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_r2(best_model: Any) -> Optional[float]:
    if not isinstance(best_model, dict):
        return None
    metrics = best_model.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("r2")
    return float(value) if isinstance(value, (int, float)) else None


def build_client_one_liner(client_features: Any) -> Optional[str]:
    if not isinstance(client_features, dict) or not client_features.get("client_found"):
        return None
    elasticity = first_number(
        client_features.get("elasticity"),
        (client_features.get("metadata") or {}).get("client_price_elasticity") if isinstance(client_features.get("metadata"), dict) else None,
    )
    average_price = first_number(
        client_features.get("avg_price_ht"),
        (client_features.get("metadata") or {}).get("client_avg_price_ht") if isinstance(client_features.get("metadata"), dict) else None,
    )
    parts = [f"Client {client_features.get('siren') or 'n/a'} profile is available"]
    if elasticity is not None:
        parts.append(f"elasticity {elasticity:.4f}")
    if average_price is not None:
        parts.append(f"average historical price HT {average_price:.4f}")
    return ", ".join(parts) + "."


copilot_graph = build_graph()
