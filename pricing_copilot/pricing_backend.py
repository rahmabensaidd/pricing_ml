from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
from urllib import error, request


PRICING_API_BASE_URL = os.getenv("PRICING_API_BASE_URL", "http://localhost:8000").rstrip("/")
MIN_R2_THRESHOLD = 0.60


def predict_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{PRICING_API_BASE_URL}/predict"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=45) as response:
        raw_body = response.read().decode("utf-8")
        return json.loads(raw_body) if raw_body else {}


def get_client_features(siren: Optional[str]) -> Optional[Dict[str, Any]]:
    normalized = (siren or "").strip()
    if not normalized:
        return None

    url = f"{PRICING_API_BASE_URL}/features/client/{normalized}"
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with request.urlopen(req, timeout=30) as response:
            raw_body = response.read().decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            return normalize_client_features_response(payload)
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def normalize_client_features_response(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    found = payload.get("found")
    features = payload.get("features")
    if found is not True or not isinstance(features, dict):
        return {
            "siren": payload.get("siren"),
            "client_found": False,
        }

    metadata = {
        "client_nb_orders": features.get("client_nb_orders"),
        "client_avg_price_ht": features.get("client_avg_price_ht"),
        "client_price_std_ht": features.get("client_price_std_ht"),
        "client_avg_quantity": features.get("client_avg_quantity"),
        "client_price_volatility": features.get("client_price_volatility"),
        "client_relative_price": features.get("client_relative_price"),
        "client_first_order": features.get("client_first_order"),
        "client_last_order": features.get("client_last_order"),
        "client_seniority_years": features.get("client_seniority_years"),
        "client_recency_days": features.get("client_recency_days"),
        "client_price_elasticity": features.get("client_price_elasticity"),
    }

    return {
        "model_name": "ClientFeatures",
        "model_version": 2,
        "siren": features.get("siren") or payload.get("siren"),
        "client_found": True,
        "elasticity": features.get("client_price_elasticity"),
        "seniority_years": features.get("client_seniority_years"),
        "recency_days": features.get("client_recency_days"),
        "avg_price_ht": features.get("client_avg_price_ht"),
        "n_orders": features.get("client_nb_orders"),
        "price_volatility": features.get("client_price_volatility"),
        "relative_price": features.get("client_relative_price"),
        "metadata": metadata,
    }


def ensure_client_features(pricing_details: Dict[str, Any], siren: Optional[str]) -> Dict[str, Any]:
    if not isinstance(pricing_details, dict):
        return {}

    current = pricing_details.get("client_features")
    if isinstance(current, dict) and current.get("client_found") is True:
        return pricing_details

    fallback = get_client_features(siren)
    if fallback:
        pricing_details["client_features"] = fallback

    return pricing_details


def build_explained_quote(
    quote_request: Optional[Dict[str, Any]],
    pricing_api_request: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not pricing_api_request:
        return unavailable_pricing("Pricing payload is missing")

    try:
        pricing_details = predict_pricing(pricing_api_request)
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError):
        return unavailable_pricing("Pricing service unavailable")

    siren = (quote_request or {}).get("siren")
    pricing_details = ensure_client_features(pricing_details, siren)
    selected = select_best_prediction(pricing_details, siren)

    pricing: Dict[str, Any] = {
        "available": selected is not None,
        "selectedPrice": selected.get("price") if selected else None,
        "selectedModel": selected.get("model_name") if selected else None,
        "selectedStrategy": selected.get("strategy") if selected else None,
        "pricingDetails": pricing_details,
        "message": "Pricing calculated successfully" if selected else "No pricing model available for this configuration",
        "requestId": pricing_details.get("request_id"),
        "timestamp": pricing_details.get("timestamp"),
        "input": pricing_details.get("input"),
        "explanation": build_explanation(pricing_details, selected, siren),
    }
    return pricing


def unavailable_pricing(message: str) -> Dict[str, Any]:
    return {
        "available": False,
        "selectedPrice": None,
        "selectedModel": None,
        "selectedStrategy": None,
        "pricingDetails": {},
        "message": message,
        "requestId": None,
        "timestamp": None,
        "input": None,
        "explanation": build_empty_explanation(),
    }


def build_empty_explanation() -> Dict[str, Any]:
    return {
        "selectedSourceKey": "global_prediction",
        "selectedSourceLabel": "global_prediction",
        "explanationType": None,
        "formula": None,
        "shapAvailable": None,
        "clientContext": "No client-specific context was used.",
        "keyInsights": ["No pricing explanation is available."],
        "topDrivers": [],
        "modelSummaries": [],
    }


def select_best_prediction(response: Dict[str, Any], request_siren: Optional[str]) -> Optional[Dict[str, Any]]:
    siren_known = bool(request_siren and request_siren.strip() and request_siren.strip().upper() != "NONE")

    if siren_known:
        candidate = extract_prediction(response, "couple_linear", "COUPLE_LINEAR", True)
        if candidate:
            return candidate

        candidate = extract_prediction(response, "family_linear", "FAMILY_LINEAR", True)
        if candidate:
            return candidate
    else:
        candidate = extract_prediction(response, "family_linear", "FAMILY_LINEAR", True)
        if candidate:
            return candidate

    return extract_prediction(response, "global_prediction", "GLOBAL", False)


def extract_prediction(
    response: Dict[str, Any],
    key: str,
    strategy: str,
    enforce_r2_threshold: bool,
) -> Optional[Dict[str, Any]]:
    block = response.get(key)
    if not isinstance(block, dict):
        return None

    available = block.get("available")
    prediction = block.get("prediction")
    model_name = block.get("model_name")

    if available is not True or not isinstance(prediction, (int, float)):
        return None

    r2 = extract_r2(block)
    if enforce_r2_threshold and (r2 is None or r2 < MIN_R2_THRESHOLD):
        return None

    return {
        "key": key,
        "strategy": strategy,
        "price": float(prediction),
        "model_name": str(model_name) if model_name is not None else key,
        "r2": r2,
    }


def build_explanation(
    pricing_details: Dict[str, Any],
    selected: Optional[Dict[str, Any]],
    siren: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(pricing_details, dict):
        return build_empty_explanation()

    selected_key = resolve_selected_key(selected.get("strategy") if selected else None)
    selected_block = get_dict(pricing_details.get(selected_key)) or {}
    explanation_type = resolve_explanation_type(selected_block)

    explanation = {
        "selectedSourceKey": selected_key,
        "selectedSourceLabel": read_label(pricing_details, selected_key),
        "explanationType": explanation_type,
        "formula": read_string(selected_block.get("formula")),
        "shapAvailable": read_bool(selected_block.get("shap_available")),
        "clientContext": build_client_context(pricing_details, siren),
        "keyInsights": [],
        "topDrivers": extract_top_drivers(selected_block, 6),
        "modelSummaries": build_model_summaries(pricing_details),
    }
    explanation["keyInsights"] = build_insights(selected, explanation, pricing_details)
    return explanation


def resolve_selected_key(strategy: Optional[str]) -> str:
    normalized = (strategy or "").upper()
    if normalized == "COUPLE_LINEAR":
        return "couple_linear"
    if normalized == "FAMILY_LINEAR":
        return "family_linear"
    return "global_prediction"


def read_label(pricing_details: Dict[str, Any], selected_key: str) -> str:
    selected_block = get_dict(pricing_details.get(selected_key))
    if not selected_block:
        return selected_key

    couple = read_string(selected_block.get("couple"))
    if couple:
        return couple

    family = read_string(selected_block.get("family"))
    if family:
        return family

    model_name = read_string(selected_block.get("model_name"))
    return model_name or selected_key


def resolve_explanation_type(prediction_block: Dict[str, Any]) -> Optional[str]:
    formula = read_string(prediction_block.get("formula"))
    if formula:
        return "FORMULA"

    if read_bool(prediction_block.get("shap_available")) is True:
        return "SHAP"

    raw_importance = prediction_block.get("feature_importance")
    if isinstance(raw_importance, dict):
        return "FEATURE_IMPORTANCE"

    return "MODEL_SUMMARY"


def extract_top_drivers(prediction_block: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    raw_importance = prediction_block.get("feature_importance")
    if not isinstance(raw_importance, dict):
        return []

    sorted_items = sorted(
        (
            (str(key), float(value))
            for key, value in raw_importance.items()
            if isinstance(value, (int, float)) and float(value) > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    return [{"name": name, "importance": value} for name, value in sorted_items[:limit]]


def build_client_context(pricing_details: Dict[str, Any], siren: Optional[str]) -> str:
    client_features = get_dict(pricing_details.get("client_features"))
    if not client_features or client_features.get("client_found") is not True:
        if siren and siren.strip():
            return f"No historical client profile was found for {siren}."
        return "No client-specific context was used."

    metadata = get_dict(client_features.get("metadata")) or {}
    orders = value_as_string(metadata.get("client_nb_orders"))
    average_price = format_number(metadata.get("client_avg_price_ht"))
    seniority = value_as_string(metadata.get("client_seniority_years"))
    recency = value_as_string(metadata.get("client_recency_days"))
    client_code = value_as_string(client_features.get("siren"))

    return (
        f"Client profile found for {client_code} with {orders} historical orders, "
        f"average price HT {average_price}, seniority {seniority} years, recency {recency} days."
    )


def build_insights(
    selected: Optional[Dict[str, Any]],
    explanation: Dict[str, Any],
    pricing_details: Dict[str, Any],
) -> List[str]:
    insights: List[str] = []
    insights.append(
        "Selected strategy: "
        + value_as_string(selected.get("strategy") if selected else None)
        + " via model "
        + value_as_string(selected.get("model_name") if selected else None)
        + " with suggested unit price "
        + format_number(selected.get("price") if selected else None)
        + "."
    )

    selected_block = get_dict(pricing_details.get(explanation.get("selectedSourceKey"))) or {}
    r2 = extract_r2(selected_block)
    if r2 is not None:
        insights.append(f"Model fit (R2): {format_number(r2)}.")

    top_drivers = explanation.get("topDrivers") or []
    if top_drivers:
        drivers = ", ".join(
            f"{driver.get('name')} ({format_number(driver.get('importance'))})"
            for driver in top_drivers[:3]
        )
        insights.append(f"Main drivers from the selected explanation: {drivers}.")

    if explanation.get("formula"):
        insights.append("A linear formula is available for this selected model.")
    elif explanation.get("explanationType") == "SHAP":
        insights.append("The selected model exposes SHAP-style explanations.")
    elif explanation.get("explanationType") == "FEATURE_IMPORTANCE":
        insights.append("The selected model exposes feature-importance style explanations.")

    return insights


def build_model_summaries(pricing_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries = [
        build_model_summary(pricing_details, "global_prediction", "Global model"),
        build_model_summary(pricing_details, "family_linear", "Family linear"),
        build_model_summary(pricing_details, "family_nonlinear", "Family nonlinear"),
        build_model_summary(pricing_details, "couple_linear", "Client-family linear"),
        build_model_summary(pricing_details, "couple_nonlinear", "Client-family nonlinear"),
    ]
    return [summary for summary in summaries if summary.get("modelName")]


def build_model_summary(pricing_details: Dict[str, Any], key: str, label: str) -> Dict[str, Any]:
    block = get_dict(pricing_details.get(key)) or {}
    if not block:
        return {"key": key, "label": label, "modelName": None, "prediction": None, "r2": None, "available": None, "explanationType": None}

    return {
        "key": key,
        "label": label,
        "modelName": read_string(block.get("model_name")),
        "prediction": read_number(block.get("prediction")),
        "r2": extract_r2(block),
        "available": read_bool(block.get("available")),
        "explanationType": resolve_explanation_type(block),
    }


def get_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def read_string(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def read_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def read_number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def extract_r2(block: Dict[str, Any]) -> Optional[float]:
    metrics = get_dict(block.get("metrics"))
    return read_number(metrics.get("r2")) if metrics else None


def value_as_string(value: Any) -> str:
    return "n/a" if value is None else str(value)


def format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return value_as_string(value)
    return f"{float(value):.4f}"
