"""
Metrics Functions for evaluation

"""

import os
import json
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

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
    denom = np.maximum(mae_naive, eps)
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


def calculate_quantile_reliability(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    quantiles: List[float],
    feature_names: Optional[List[str]] = None,
    plot: bool = False,
    ax: Optional[plt.Axes] = None
) -> np.ndarray:
    """
    Calculates observed frequency of hits and optionally plots a reliability diagram.
    
    y_true: shape (n_samples,) or (n_samples, 1)
    y_pred: shape (n_samples, len(quantiles))
    quantiles: List of quantile levels (e.g., [0.1, 0.2, ..., 0.9])
    plot: Whether to generate a plot.
    ax: Existing matplotlib axis to plot on. If None and plot=True, creates a new figure.
    """
    # Ensure inputs are correct shapes
    y_true = np.asarray(y_true).reshape(y_true.shape[0], -1, 1)
    y_pred = np.asarray(y_pred, dtype=float)
    quantiles = np.asarray(quantiles)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Sample counts must match: {y_true.shape[0]} vs {y_pred.shape[0]}")

    # Calculate observed frequencies
    # True if the actual value fell at or below the predicted quantile
    hits = (y_true <= y_pred)
    observed_frequencies = np.mean(hits, axis=0)
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        
        # Reference Line (Perfect Calibration)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration', alpha=0.7)

        # Plot each feature's reliability line
        num_features = observed_frequencies.shape[0]
        for i in range(num_features):
            label = feature_names[i] if feature_names else f"Feature {i}"
            ax.plot(
                quantiles, 
                observed_frequencies[i], 
                marker='o', 
                linestyle='-', 
                linewidth=1.5, 
                label=label
            )

        ax.set_xlabel('Predicted Quantile (Expected Frequency)')
        ax.set_ylabel('Observed Frequency')
        ax.set_title('Reliability Diagram (Per Feature)')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside if many features
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        # If we created a new figure and legend is outside, adjust layout
        if ax.get_figure():
            plt.tight_layout()

    return observed_frequencies


def calculate_wis_from_quantiles(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    quantiles: list
) -> float:
    """
    Calculates the Weighted Interval Score (WIS) by automatically 
    pairing symmetric quantiles.
    
    y_true: shape (n_samples,)
    y_pred: shape (n_samples, len(quantiles))
    quantiles: list of floats (e.g., [0.1, 0.2, ..., 0.9])
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred)
    quantiles = np.array(quantiles)
    
    n_samples = len(y_true)
    K = len(quantiles)
    
    # 1. Find the Median
    # We look for 0.5. If not exactly 0.5, we take the middle index.
    median_idx = np.argmin(np.abs(quantiles - 0.5)) 
    median_pred = y_pred[:, :, median_idx]
    
    # Start the score with the Absolute Error of the median (weight = 1/2)
    total_wis = 0.5 * np.mean(np.abs(y_true - median_pred))
    
    # 2. Identify symmetric pairs (e.g., 0.1 and 0.9)
    # We only iterate through the first half of the quantiles
    for i in range(median_idx):
        low_q = quantiles[i]
        # Find the matching upper quantile (e.g., if low is 0.1, high is 0.9)
        high_idx = K - 1 - i
        high_q = quantiles[high_idx]
        
        # Alpha is the probability MASS outside the interval
        # e.g., for a 10%-90% interval, alpha is 0.2 (20%)
        alpha = 2 * low_q 
        
        low_vals = y_pred[:, :, i]
        high_vals = y_pred[:, :, high_idx]
        
        # WIS Components for this interval:
        # Sharpness
        sharpness = (high_vals - low_vals)
        
        # Over-prediction (Truth below interval)
        overprediction = (2/alpha) * np.maximum(0, low_vals - y_true)
        
        # Under-prediction (Truth above interval)
        underprediction = (2/alpha) * np.maximum(0, y_true - high_vals)
        
        interval_score = sharpness + overprediction + underprediction
        
        # Add to total weighted by alpha/2
        total_wis += np.mean(interval_score) * (alpha / 2)
        
    # Final normalization by the number of components
    # (Number of intervals + 1 for the median)
    return (total_wis / (median_idx + 0.5)).item()



def score_data(y_true, y_pred, y_train, quantiles,
               model_path=None, feature_names=None, log_prefix:Optional[str]=None,
               log_scores:bool=True,
               multioutput='raw_values'):
    """score the model

    Args:
        y_true (_type_): true values shape (n_samples, n_features)
        y_pred (_type_): predicted values shape (n_samples, n_features, quantiles)
        y_train (_type_): training values shape (n_samples, n_features)
        quantiles (_type_): list of quantiles
        model_path (_type_, optional): path to save results. Defaults to None.
        feature_names (_type_, optional): names of features. Defaults to None.
        log_prefix (_type_, optional): prefix for logging. Defaults to None.
        log_scores (_type_, optional): whether to log scores. Defaults to True.
        multioutput (_type_, optional): how to handle multiple outputs. Defaults to 'raw_values'.

    Returns:
        _type_: _description_
    """

    quantiles = np.array(quantiles)
    mean_idx = np.argmin(np.abs(quantiles - 0.5))
    mean_pred = y_pred[:,:, mean_idx]

    score_smape = symmetric_mean_absolute_percentage_error(y_true, mean_pred, multioutput=multioutput)
    score_mase = mean_absolute_scaled_error(y_true, mean_pred, y_train, multioutput=multioutput)
    score_r2 = r2_score(y_true, mean_pred, multioutput=multioutput)
    score_wis = calculate_wis_from_quantiles(y_true, y_pred, quantiles)

    if log_scores:
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
                "wis_mean": score_wis
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
            print_lines.append(f'Prefix: {log_prefix} | Feature: {y_name} | SMAPE: {score_smape[i]:.4f} | MASE: {score_mase[i]:.4f} | r2: {score_r2[i]:.4f} | WIS: {score_wis:.4f}')
            print(print_lines[-1])
        
        # write JSON file
        if model_path is not None:
            os.makedirs(model_path, exist_ok=True)
            filename = "scores.json"
            full_path = os.path.join(model_path, filename)
            with open(full_path, "w") as f:
                json.dump(results, f, indent=2)

    return score_smape, score_mase, score_r2, score_wis