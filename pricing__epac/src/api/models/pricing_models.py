from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pricing__epac.src.config.feature_config import ALL_FEATURES

class PricingRequest(BaseModel):
    """Pricing prediction request"""
    siren: Optional[str] = Field(None, description="Client SIREN (optional)")
    binding_type: Optional[str] = Field(None, description="Binding type (optional)")
    features: Dict[str, Any] = Field(..., description="Product technical features")

    class Config:
        json_schema_extra = {
            "example": {
                "siren": "SAV",
                "binding_type": "SS",
                "features": {
                    "quantity": 500,
                    "production_page": 252,
                    "height": 276.22,
                    "thickness": 13.578,
                    "width": 209.55,
                    "security_label": 0,
                    "has_coil": 0,
                    "has_insert": 0,
                    "has_tab": 0,
                    "has_backcover": 1,
                    "perf": 1,
                    "double_sided_cover": 0,
                    "shrinkwrap": 0,
                    "three_hole_drill": 0,
                    "text_paper_type": "BIRCH_W40_TB",
                    "text_color": "4/4",
                    "cover_finish_type": "LAYFLAT-GLOSS",
                    "cover_color": "4/0",
                    "cover_size": "L",
                    "cover_paper_type": "12PT_C1S",
                    "head_and_tail": "NONE",
                    "priority_level": "NORMAL",
                    "coil_type": "NONE",
                    "tab_color": "NONE",
                    "insert_paper_type": "NONE",
                    "case_finish_type": "NONE",
                    "spine_type": "NONE",
                    "label_type": "NONE"
                }
            }
        }


class ModelMetrics(BaseModel):
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None


class PredictionGlobal(BaseModel):
    model_name: str
    model_version: int
    prediction: float
    metrics: ModelMetrics = ModelMetrics()
    features_used: List[str] = ALL_FEATURES
    feature_importance: Dict[str, float] = {}
    available: bool = True
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionFamilyLinear(BaseModel):
    model_name: str
    model_version: int
    family: str
    prediction: float
    formula: Optional[str] = None
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionFamilyNonLinear(BaseModel):
    model_name: str
    model_version: int
    family: str
    prediction: float
    feature_importance: Dict[str, float] = {}
    shap_available: bool = False
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionCoupleLinear(BaseModel):
    model_name: str
    model_version: int
    couple: str
    prediction: float
    formula: Optional[str] = None
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class PredictionCoupleNonLinear(BaseModel):
    model_name: str
    model_version: int
    couple: str
    prediction: float
    feature_importance: Dict[str, float] = {}
    shap_available: bool = False
    metrics: ModelMetrics = ModelMetrics()
    available: bool = False
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class ClientFeaturesInfo(BaseModel):
    model_name: str
    model_version: int
    siren: str
    client_found: bool
    elasticity: Optional[float] = None
    seniority_years: Optional[float] = None
    recency_days: Optional[float] = None
    avg_price_ht: Optional[float] = None
    n_orders: Optional[int] = None
    price_volatility: Optional[float] = None
    relative_price: Optional[float] = None
    metadata: Dict[str, Any] = {}


class PricingResponse(BaseModel):
    request_id: str
    timestamp: str
    input: Dict[str, Any]
    global_prediction: Optional[PredictionGlobal] = None
    family_linear: Optional[PredictionFamilyLinear] = None
    family_nonlinear: Optional[PredictionFamilyNonLinear] = None
    couple_linear: Optional[PredictionCoupleLinear] = None
    couple_nonlinear: Optional[PredictionCoupleNonLinear] = None
    client_features: Optional[ClientFeaturesInfo] = None
    summary: Dict[str, Any] = {}