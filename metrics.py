from __future__ import annotations

"""Evaluation metrics

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import numpy as np
from typing import Dict, Any, List, Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    std = float(np.std(y_true))
    if std < 1e-12:
        return float("inf")
    return rmse(y_true, y_pred) / std


def rmse_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    nt = y_true.shape[0]
    errors = np.zeros(nt)
    for t in range(nt):
        diff = y_true[t].ravel() - y_pred[t].ravel()
        errors[t] = float(np.sqrt(np.mean(diff ** 2)))
    return errors


def correlation_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    nt = y_true.shape[0]
    corrs = np.zeros(nt)
    for t in range(nt):
        a = y_true[t].ravel()
        b = y_pred[t].ravel()
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corrs[t] = 0.0
        else:
            corrs[t] = float(np.corrcoef(a, b)[0, 1])
    return corrs


def relative_energy_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    E_true = float(np.sum(y_true ** 2))
    E_pred = float(np.sum(y_pred ** 2))
    if E_true < 1e-12:
        return float("inf")
    return abs(E_pred - E_true) / E_true


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def valid_prediction_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.3,
    dt: float = 0.2,
) -> float:
    nt = y_true.shape[0]
    ref_std = float(np.std(y_true))
    if ref_std < 1e-12:
        return 0.0

    for t in range(nt):
        diff = y_true[t].ravel() - y_pred[t].ravel()
        nrmse_t = float(np.sqrt(np.mean(diff ** 2))) / ref_std
        if nrmse_t > threshold:
            return t * dt

    return nt * dt


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float = 0.2,
    model_name: str = "unknown",
    train_time: float = 0.0,
) -> Dict[str, Any]:
    nt = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:nt]
    y_pred = y_pred[:nt]

    rmse_steps = rmse_per_step(y_true, y_pred)
    corr_steps = correlation_per_step(y_true, y_pred)

    return {
        "model": model_name,
        "rmse_global": rmse(y_true, y_pred),
        "nrmse_global": nrmse(y_true, y_pred),
        "mae_global": mae(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
        "relative_energy_error": relative_energy_error(y_true, y_pred),
        "valid_prediction_time": valid_prediction_time(y_true, y_pred, threshold=0.3, dt=dt),
        "mean_correlation": float(np.mean(corr_steps)),
        "rmse_per_step": rmse_steps,
        "correlation_per_step": corr_steps,
        "n_steps": nt,
        "train_time_s": train_time,
    }


def compare_models(
    results_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = {
        "models": [],
        "rmse": [],
        "nrmse": [],
        "mae": [],
        "r_squared": [],
        "relative_energy_error": [],
        "valid_prediction_time": [],
        "mean_correlation": [],
        "train_time_s": [],
    }

    for res in results_list:
        summary["models"].append(res["model"])
        summary["rmse"].append(res["rmse_global"])
        summary["nrmse"].append(res["nrmse_global"])
        summary["mae"].append(res["mae_global"])
        summary["r_squared"].append(res["r_squared"])
        summary["relative_energy_error"].append(res["relative_energy_error"])
        summary["valid_prediction_time"].append(res["valid_prediction_time"])
        summary["mean_correlation"].append(res["mean_correlation"])
        summary["train_time_s"].append(res["train_time_s"])

    summary["best_rmse"] = summary["models"][int(np.argmin(summary["rmse"]))]
    summary["best_nrmse"] = summary["models"][int(np.argmin(summary["nrmse"]))]
    summary["best_r_squared"] = summary["models"][int(np.argmax(summary["r_squared"]))]
    summary["best_correlation"] = summary["models"][int(np.argmax(summary["mean_correlation"]))]
    summary["best_valid_time"] = summary["models"][int(np.argmax(summary["valid_prediction_time"]))]
    summary["fastest_training"] = summary["models"][int(np.argmin(summary["train_time_s"]))]

    return summary
