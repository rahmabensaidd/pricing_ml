"""Model definitions and default hyperparameters for global training."""

from typing import Any, Dict, List

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None


def get_model_configs(config) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = [
        {"name": "RandomForest", "model": RandomForestRegressor, "params": {
            "n_estimators": 200, "max_depth": 15, "min_samples_split": 10,
            "min_samples_leaf": 5, "random_state": config.random_state, "n_jobs": -1}},
        {"name": "XGBoost", "model": xgb.XGBRegressor, "params": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": config.random_state,
            "n_jobs": -1, "verbosity": 0}},
        {"name": "LightGBM", "model": lgb.LGBMRegressor, "params": {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": config.random_state,
            "n_jobs": -1, "verbose": -1}},
        {"name": "GradientBoosting", "model": GradientBoostingRegressor, "params": {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "random_state": config.random_state}},
        {"name": "ExtraTrees", "model": ExtraTreesRegressor, "params": {
            "n_estimators": 200, "max_depth": 15, "min_samples_split": 10,
            "min_samples_leaf": 5, "random_state": config.random_state, "n_jobs": -1}},
        {"name": "AdaBoost", "model": AdaBoostRegressor, "params": {
            "n_estimators": 100, "learning_rate": 0.1, "loss": "linear",
            "random_state": config.random_state}},
        {"name": "Bagging", "model": BaggingRegressor, "params": {
            "n_estimators": 50, "max_samples": 0.8, "max_features": 0.8,
            "random_state": config.random_state, "n_jobs": -1}},
        {"name": "DecisionTree", "model": DecisionTreeRegressor, "params": {
            "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5,
            "random_state": config.random_state}},
        {"name": "SVR_RBF", "model": SVR, "params": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}},
        {"name": "SVR_Poly", "model": SVR, "params": {"kernel": "poly", "degree": 3, "C": 10.0, "epsilon": 0.1, "gamma": "scale"}},
        {"name": "NeuralNetwork", "model": MLPRegressor, "params": {
            "hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam",
            "alpha": 0.001, "batch_size": "auto", "learning_rate": "adaptive",
            "max_iter": 500, "random_state": config.random_state}},
        {"name": "GaussianProcess", "model": GaussianProcessRegressor, "params": {
            "alpha": 1e-6, "normalize_y": True, "n_restarts_optimizer": 5,
            "random_state": config.random_state}},
        {"name": "KernelRidge", "model": KernelRidge, "params": {
            "alpha": 1.0, "kernel": "rbf", "gamma": 0.1}},
    ]

    if CATBOOST_AVAILABLE and CatBoostRegressor is not None:
        models.append({"name": "CatBoost", "model": CatBoostRegressor, "params": {
            "iterations": 300, "depth": 6, "learning_rate": 0.05,
            "loss_function": "RMSE", "eval_metric": "RMSE", "random_seed": config.random_state,
            "verbose": False, "allow_writing_files": False, "task_type": "CPU", "thread_count": 4}})
    return models
