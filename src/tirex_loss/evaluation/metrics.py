"""
Metrics Functions for evaluation

"""

import os
import json
import polars as pl
import numpy as np

from datetime import datetime
from typing import Optional, List
from sklearn.metrics import mean_absolute_error, r2_score


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    sp: int = 1,
    multioutput='raw_values',
    eps: float = 1e-8,
) -> List[float]:
    """
    MASE consistent with Hyndman & Koehler and the sktime implementation
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    if y_train.size <= sp:
        raise ValueError(
            f"Length of y_train must be > sp. Got len={y_train.size}, sp={sp}."
        )

    # Seasonal naive on training: ŷ_t = y_{t-sp}
    y_pred_naive_train = y_train[:-sp]  
    mae_naive = mean_absolute_error(y_train[sp:], y_pred_naive_train, multioutput=multioutput)
   
    # MAE of forecast (on horizon)
    mae_pred = mean_absolute_error(y_true, y_pred, multioutput=multioutput)

    # Avoid division by 0 (flat series => huge MASE)
    denom = max(mae_naive, eps)
    return mae_pred / denom


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput='raw_values',
    eps: float = 1e-8,
) -> List[float]:
   
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    # percentage error
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, eps)  # avoid division by zero
    percentage_error = np.abs(y_true - y_pred) / denominator
   
    output_errors = np.average(np.abs(percentage_error), axis=0)
    if multioutput == "raw_values":
        return output_errors
    return np.average(output_errors)


def score_data(y_true, y_pred, y_train, model_path=None, feature_names=None, log_prefix:Optional[str]=None):
    """score the model

    Args:
        x (_type_): _description_
        y (_type_): _description_
        feature_names (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    score_smape = symmetric_mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    score_mase = mean_absolute_scaled_error(y_true, y_pred, y_train, multioutput='raw_values')
    score_r2 = r2_score(y_true, y_pred, multioutput='raw_values')
   
    if feature_names is None:
        feature_names = list(range(y_true.shape[-1]))

    # build JSON structure
    current_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    results = {
        "log_prefix": log_prefix,
        "timestamp": current_time,
        "global": {
            # you can add global aggregates if needed, e.g. mean over features
            "smape_mean": float(np.mean(score_smape)),
            "mase_mean": float(np.mean(score_mase)),
            "r2_mean": float(np.mean(score_r2)),
        },
        "per_feature": []
    }
    for i, fname in enumerate(feature_names):
        results["per_feature"].append(
            {
                "feature": str(fname),
                "smape": float(score_smape[i]),
                "mase": float(score_mase[i]),
                "r2": float(score_r2[i]),
            }
        )

    print_lines = []
    for i, y_name in enumerate(feature_names):
        print_lines.append(f'Prefix: {log_prefix} | Feature: {y_name} | SMAPE: {score_smape[i]:.4f} | MASE: {score_mase[i]:.4f} | r2: {score_r2[i]:.4f}')
        print(print_lines[-1])
       
    # write JSON file
    if model_path is not None:
        os.makedirs(model_path, exist_ok=True)
        filename = "scores.json"
        full_path = os.path.join(model_path, filename)
        with open(full_path, "w") as f:
            json.dump(results, f, indent=2)

    return score_smape, score_mase, score_r2