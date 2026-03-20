from __future__ import annotations

"""SINDy model

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import time as time_module


def preprocess_sindy_data(
    a: np.ndarray,
    denoise: bool = True,
    standardize: bool = True,
    sg_window: int = 11,
    sg_polyorder: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    a_out = a.copy()
    scaler = {"mean": None, "std": None, "denoised": denoise, "standardized": standardize}

    if denoise:
        from scipy.signal import savgol_filter
        nt = a_out.shape[0]
        win = min(sg_window, nt - 1)
        if win % 2 == 0:
            win -= 1
        win = max(win, sg_polyorder + 2)
        if win % 2 == 0:
            win += 1
        for j in range(a_out.shape[1]):
            a_out[:, j] = savgol_filter(a_out[:, j], win, sg_polyorder)

    if standardize:
        mean = a_out.mean(axis=0)
        std = a_out.std(axis=0)
        std[std < 1e-12] = 1.0
        a_out = (a_out - mean) / std
        scaler["mean"] = mean
        scaler["std"] = std

    return a_out, scaler


def inverse_transform_sindy(a: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    if scaler.get("standardized") and scaler["mean"] is not None:
        return a * scaler["std"] + scaler["mean"]
    return a


def fit_sindy(
    a_train: np.ndarray,
    dt: float = 0.2,
    threshold: float = 0.005,
    poly_degree: int = 2,
    max_iter: int = 50,
    alpha: float = 0.1,
    smooth_derivatives: bool = True,
) -> Dict[str, Any]:
    t_start = time_module.time()

    import pysindy as ps

    r = a_train.shape[1]

    feature_names = [f"a{i}" for i in range(r)]

    optimizer = ps.STLSQ(
        threshold=threshold,
        max_iter=max_iter,
        alpha=alpha,
    )
    library = ps.PolynomialLibrary(degree=poly_degree, include_bias=True)

    if smooth_derivatives:
        differentiation_method = ps.SmoothedFiniteDifference()
    else:
        differentiation_method = ps.FiniteDifference()

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        differentiation_method=differentiation_method,
        discrete_time=False,
    )

    model.fit(a_train, t=dt)

    coefficients = model.coefficients()
    try:
        feature_lib_names = model.get_feature_names()
    except Exception:
        feature_lib_names = feature_names
    n_terms = int(np.count_nonzero(coefficients))

    t_end = time_module.time()

    return {
        "model": model,
        "coefficients": coefficients,
        "feature_names": feature_lib_names,
        "n_terms": n_terms,
        "threshold": threshold,
        "poly_degree": poly_degree,
        "dt": dt,
        "r": r,
        "alpha": alpha,
        "train_time": t_end - t_start,
    }


def _check_stability(
    sindy_result: Dict[str, Any],
    a0: np.ndarray,
    n_check: int = 500,
    dt: float = 0.2,
    max_ratio: float = 10.0,
) -> bool:
    try:
        a_pred = predict_sindy_iterative(sindy_result, a0, n_check, dt)
        max_pred = np.max(np.abs(a_pred))
        max_ref = max(1.0, np.max(np.abs(a0)))
        return max_pred < max_ratio * max_ref
    except Exception:
        return False


def fit_sindy_with_search(
    a_train: np.ndarray,
    a_val: np.ndarray,
    dt: float = 0.2,
    sindy_r: int = 8,
    preprocess: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from metrics import r_squared, correlation_per_step

    if preprocess:
        a_train_p, scaler = preprocess_sindy_data(a_train, denoise=True, standardize=True)
        a_val_p, _ = preprocess_sindy_data(a_val, denoise=False, standardize=True,
                                            sg_window=11, sg_polyorder=3)
        if scaler["mean"] is not None:
            a_val_p = (a_val - scaler["mean"]) / scaler["std"]
    else:
        a_train_p = a_train
        a_val_p = a_val
        scaler = {"mean": None, "std": None, "denoised": False, "standardized": False}

    if sindy_r <= 10:
        degrees = [2, 3]
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        alphas = [0.01, 0.05, 0.1, 0.5]
    else:
        degrees = [1, 2]
        thresholds = [0.005, 0.01, 0.02, 0.05]
        alphas = [0.05, 0.1, 0.5]

    n_val_use = min(200, a_val_p.shape[0])
    best_result = None
    best_score = -np.inf
    best_config = {}

    for degree in degrees:
        for threshold in thresholds:
            for alpha in alphas:
                try:
                    sindy = fit_sindy(
                        a_train_p, dt=dt,
                        threshold=threshold,
                        poly_degree=degree,
                        alpha=alpha,
                        smooth_derivatives=True,
                    )

                    if sindy["n_terms"] == 0:
                        continue

                    if not _check_stability(sindy, a_val_p[0], n_check=200, dt=dt):
                        continue

                    scores = []
                    vp = predict_sindy_iterative(sindy, a0=a_val_p[0], n_steps=n_val_use, dt=dt)
                    max_abs = max(1.0, float(np.max(np.abs(a_train_p))) * 3.0)
                    vp = np.clip(np.nan_to_num(vp, nan=0.0), -max_abs, max_abs)

                    horizons = [h for h in (10, 50, 100, n_val_use) if h <= n_val_use]
                    for h in horizons:
                        yt, yp = a_val_p[:h], vp[:h]
                        r2 = float(r_squared(yt, yp))
                        corr = float(np.mean(correlation_per_step(yt, yp)))
                        scores.append(0.7 * r2 + 0.3 * corr)

                    score = float(np.mean(scores)) if scores else -999.0

                    if score > best_score:
                        best_score = score
                        best_result = sindy
                        best_config = {
                            "degree": degree, "threshold": threshold,
                            "alpha": alpha, "score": score,
                        }

                except Exception:
                    continue

    if best_result is None:
        for fb_thresh in [0.005, 0.01, 0.001]:
            try:
                fb_sindy = fit_sindy(a_train, dt=dt, threshold=fb_thresh,
                                     poly_degree=2, alpha=0.1, smooth_derivatives=True)
                if fb_sindy["n_terms"] > 0:
                    best_result = fb_sindy
                    scaler = {"mean": None, "std": None, "denoised": False, "standardized": False}
                    best_config = {"degree": 2, "threshold": fb_thresh,
                                   "alpha": 0.1, "score": -999.0, "fallback": True}
                    break
            except Exception:
                continue
        if best_result is None:
            best_result = fit_sindy(a_train_p, dt=dt, threshold=0.005, poly_degree=2, alpha=0.1)
            best_config = {"degree": 2, "threshold": 0.005, "alpha": 0.1, "score": -999.0}

    best_result["search_config"] = best_config
    return best_result, scaler


def predict_sindy(
    sindy_result: Dict[str, Any],
    a0: np.ndarray,
    n_steps: int,
    dt: Optional[float] = None,
) -> np.ndarray:
    model = sindy_result["model"]
    dt_sindy = dt if dt is not None else sindy_result["dt"]
    r = a0.shape[0]

    t_span = np.arange(n_steps + 1) * dt_sindy

    try:
        a_sim = model.simulate(a0, t_span, integrator="odeint",
                               integrator_kws={"mxstep": 5000})
        if a_sim.shape[0] >= n_steps + 1:
            return a_sim[:n_steps]
        else:
            a_pred = np.zeros((n_steps, r))
            a_pred[:a_sim.shape[0]] = a_sim[:n_steps]
            a_pred[a_sim.shape[0]:] = a_sim[-1]
            return a_pred
    except Exception as e:
        print(f"[SINDy] Integration failed: {e}")
        a_pred = np.tile(a0, (n_steps, 1))
        return a_pred


def predict_sindy_iterative(
    sindy_result: Dict[str, Any],
    a0: np.ndarray,
    n_steps: int,
    dt: Optional[float] = None,
) -> np.ndarray:
    model = sindy_result["model"]
    dt_sindy = dt if dt is not None else sindy_result["dt"]
    r = a0.shape[0]
    a_pred = np.zeros((n_steps, r))
    a_current = a0.astype(float).copy()
    max_val = 10.0 * np.max(np.abs(a0))

    def rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(state[np.newaxis, :])[0], dtype=float)

    for t in range(n_steps):
        a_pred[t] = a_current
        try:
            da = rhs(a_current)
            a_current = a_current + dt_sindy * da
            a_current = np.clip(a_current, -max_val, max_val)
            if not np.all(np.isfinite(a_current)):
                a_pred[t + 1:] = a_pred[t]
                break
        except Exception:
            a_pred[t + 1:] = a_pred[t]
            break

    return a_pred


def predict_sindy_rolling_restart(
    sindy_result: Dict[str, Any],
    a_true: np.ndarray,
    n_steps: int,
    dt: Optional[float] = None,
    restart_interval: int = 50,
) -> np.ndarray:
    model = sindy_result["model"]
    dt_sindy = dt if dt is not None else sindy_result["dt"]
    r = a_true.shape[1]
    a_pred = np.zeros((n_steps, r))
    max_val = 10.0 * np.max(np.abs(a_true))

    def rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(state[np.newaxis, :])[0], dtype=float)

    t = 0
    while t < n_steps:
        a_current = a_true[t].astype(float).copy()
        segment_len = min(restart_interval, n_steps - t)

        for s in range(segment_len):
            a_pred[t + s] = a_current
            if t + s + 1 < n_steps:
                try:
                    da = rhs(a_current)
                    a_current = a_current + dt_sindy * da
                    a_current = np.clip(a_current, -max_val, max_val)
                    if not np.all(np.isfinite(a_current)):
                        a_current = a_true[t + s].astype(float).copy()
                except Exception:
                    a_current = a_true[t + s].astype(float).copy()

        t += segment_len

    return a_pred
