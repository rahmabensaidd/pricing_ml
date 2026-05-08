from typing import Any, Dict, List, Optional


MODEL_PRIORITY = [
    "couple_linear",
    "family_linear",
    "global_prediction",
    "couple_nonlinear",
    "family_nonlinear",
]

STRATEGY_TO_KEY = {
    "COUPLE_LINEAR": "couple_linear",
    "FAMILY_LINEAR": "family_linear",
    "GLOBAL": "global_prediction",
}


def extract_available_predictions(pricing_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []

    for key in MODEL_PRIORITY:
        model = pricing_response.get(key)
        if model and model.get("available") is True:
            predictions.append({
                "key": key,
                "model_name": model.get("model_name"),
                "prediction": model.get("prediction"),
                "metrics": model.get("metrics", {}),
                "feature_importance": model.get("feature_importance", {}),
                "formula": model.get("formula"),
            })

    return predictions


def select_best_model(
    predictions: List[Dict[str, Any]],
    selected_strategy: Optional[str] = None,
    selected_model: Optional[str] = None,
    selected_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if not predictions:
        return None

    if selected_model:
        for prediction in predictions:
            model_name = prediction.get("model_name")
            if isinstance(model_name, str) and model_name.lower() == selected_model.lower():
                return prediction

    selected_key = STRATEGY_TO_KEY.get((selected_strategy or "").upper())
    if selected_key:
        for prediction in predictions:
            if prediction.get("key") == selected_key:
                return prediction

    if isinstance(selected_price, (int, float)):
        closest = min(
            (
                prediction for prediction in predictions
                if isinstance(prediction.get("prediction"), (int, float))
            ),
            key=lambda prediction: abs(float(prediction["prediction"]) - float(selected_price)),
            default=None,
        )
        if closest is not None:
            return closest

    return predictions[0]


def calculate_confidence(
    predictions: List[Dict[str, Any]],
    best_model: Optional[Dict[str, Any]] = None,
    selected_strategy: Optional[str] = None,
    pricing_explanation: Optional[Dict[str, Any]] = None,
) -> str:
    metrics = best_model.get("metrics", {}) if isinstance(best_model, dict) else {}
    raw_r2 = metrics.get("r2") if isinstance(metrics, dict) else None
    r2 = float(raw_r2) if isinstance(raw_r2, (int, float)) else None

    formula_available = bool((pricing_explanation or {}).get("formula"))
    selected_strategy_upper = (selected_strategy or "").upper()

    if r2 is not None:
        if r2 >= 0.90:
            return "HIGH"
        if r2 >= 0.75:
            return "MEDIUM"
        if r2 >= 0.60 and (formula_available or selected_strategy_upper in {"COUPLE_LINEAR", "FAMILY_LINEAR"}):
            return "MEDIUM"

    prices = [prediction["prediction"] for prediction in predictions if isinstance(prediction.get("prediction"), (int, float))]
    if len(prices) < 2:
        return "LOW"

    avg_price = sum(prices) / len(prices)
    if avg_price == 0:
        return "LOW"

    spread_percent = ((max(prices) - min(prices)) / avg_price) * 100.0
    if spread_percent <= 5:
        return "HIGH"
    if spread_percent <= 15:
        return "MEDIUM"
    return "LOW"


def recommend_action(confidence: str) -> str:
    if confidence == "HIGH":
        return "APPROVE"
    if confidence == "MEDIUM":
        return "REVIEW"
    return "MANUAL_REVIEW"


def normalize_top_drivers(
    pricing_explanation: Optional[Dict[str, Any]],
    best_model: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_drivers = (pricing_explanation or {}).get("topDrivers")
    if isinstance(raw_drivers, list) and raw_drivers:
        normalized: List[Dict[str, Any]] = []
        for driver in raw_drivers:
            if not isinstance(driver, dict):
                continue
            name = driver.get("name")
            importance = driver.get("importance")
            if isinstance(name, str) and isinstance(importance, (int, float)):
                normalized.append({"name": name, "importance": float(importance)})
        if normalized:
            return normalized

    raw_importance = best_model.get("feature_importance") if isinstance(best_model, dict) else None
    if isinstance(raw_importance, dict):
        return [
            {"name": str(name), "importance": float(value)}
            for name, value in sorted(raw_importance.items(), key=lambda item: item[1], reverse=True)
            if isinstance(value, (int, float)) and float(value) > 0
        ][:6]

    return []


def normalize_model_summaries(
    pricing_explanation: Optional[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_summaries = (pricing_explanation or {}).get("modelSummaries")
    if isinstance(raw_summaries, list) and raw_summaries:
        summaries: List[Dict[str, Any]] = []
        for summary in raw_summaries:
            if isinstance(summary, dict):
                summaries.append(summary)
        if summaries:
            return summaries

    return [
        {
            "key": prediction.get("key"),
            "modelName": prediction.get("model_name"),
            "prediction": prediction.get("prediction"),
            "r2": (prediction.get("metrics") or {}).get("r2"),
        }
        for prediction in predictions
    ]


def build_analysis_context(
    pricing_response: Dict[str, Any],
    selected_question: str,
    selected_strategy: Optional[str] = None,
    selected_model: Optional[str] = None,
    selected_price: Optional[float] = None,
    pricing_explanation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    predictions = extract_available_predictions(pricing_response)
    best_model = select_best_model(
        predictions=predictions,
        selected_strategy=selected_strategy,
        selected_model=selected_model,
        selected_price=selected_price,
    )
    confidence = calculate_confidence(
        predictions=predictions,
        best_model=best_model,
        selected_strategy=selected_strategy,
        pricing_explanation=pricing_explanation,
    )
    action = recommend_action(confidence)

    input_block = pricing_response.get("input", {}) if isinstance(pricing_response, dict) else {}
    features = input_block.get("features", {}) if isinstance(input_block, dict) else {}

    formula = None
    explanation_type = None
    if isinstance(pricing_explanation, dict):
        formula = pricing_explanation.get("formula")
        explanation_type = pricing_explanation.get("explanationType")

    if not formula and isinstance(best_model, dict):
        formula = best_model.get("formula")

    resolved_model_name = selected_model
    if not resolved_model_name and isinstance(best_model, dict):
        resolved_model_name = best_model.get("model_name")

    resolved_price = selected_price
    if resolved_price is None and isinstance(best_model, dict):
        resolved_price = best_model.get("prediction")

    return {
        "selected_question": selected_question,
        "features": features,
        "predictions": predictions,
        "best_model": best_model,
        "selected_strategy": selected_strategy,
        "selected_model_name": resolved_model_name,
        "selected_price": resolved_price,
        "formula": formula,
        "explanation_type": explanation_type,
        "top_drivers": normalize_top_drivers(pricing_explanation, best_model),
        "model_summaries": normalize_model_summaries(pricing_explanation, predictions),
        "confidence": confidence,
        "recommended_action": action,
        "client_features": pricing_response.get("client_features"),
        "summary": pricing_response.get("summary", {}),
        "pricing_explanation": pricing_explanation or {},
    }
