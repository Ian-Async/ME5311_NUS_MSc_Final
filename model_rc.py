from __future__ import annotations

"""Reservoir Computing models: ESN and NG-RC / NVAR

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import time as time_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def _initialize_reservoir(
    n_input: int,
    n_reservoir: int,
    spectral_radius: float,
    input_scaling: float,
    connectivity: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)

    W_in = rng.uniform(-input_scaling, input_scaling, size=(n_reservoir, n_input))

    W = rng.randn(n_reservoir, n_reservoir)
    mask = rng.rand(n_reservoir, n_reservoir) < connectivity
    W *= mask

    eigenvalues = np.linalg.eigvals(W)
    current_sr = np.max(np.abs(eigenvalues))
    if current_sr > 1e-10:
        W = W * (spectral_radius / current_sr)

    return W_in, W


def _run_reservoir(
    u: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    leak_rate: float = 1.0,
    noise_level: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    nt, n_input = u.shape
    n_reservoir = W.shape[0]
    states = np.zeros((nt, n_reservoir))
    x = np.zeros(n_reservoir)

    for t in range(nt):
        pre_activation = W_in @ u[t] + W @ x
        if noise_level > 0 and rng is not None:
            pre_activation += noise_level * rng.randn(n_reservoir)
        x_new = np.tanh(pre_activation)
        x = (1.0 - leak_rate) * x + leak_rate * x_new
        states[t] = x

    return states


def _train_readout(
    states: np.ndarray,
    targets: np.ndarray,
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    nt = states.shape[0]
    S = np.hstack([states, np.ones((nt, 1))])
    StS = S.T @ S + ridge_alpha * np.eye(S.shape[1])
    W_out = np.linalg.solve(StS, S.T @ targets).T
    return W_out


def _train_readout_multistep(
    u_train: np.ndarray,
    targets: np.ndarray,
    W_in: np.ndarray,
    W: np.ndarray,
    W_out_init: np.ndarray,
    leak_rate: float,
    ridge_alpha: float,
    n_multistep: int = 5,
    weight_decay: float = 0.8,
) -> np.ndarray:
    nt = u_train.shape[0]
    n_reservoir = W.shape[0]
    r = targets.shape[1]

    states = _run_reservoir(u_train, W_in, W, leak_rate=leak_rate)

    max_t = nt - n_multistep
    if max_t < 100:
        return _train_readout(states, targets, ridge_alpha)

    weights = np.array([weight_decay ** k for k in range(n_multistep)])
    weights /= weights.sum()

    weighted_targets = np.zeros_like(targets[:max_t])
    for k in range(n_multistep):
        weighted_targets += weights[k] * targets[k:max_t + k]

    S = np.hstack([states[:max_t], np.ones((max_t, 1))])
    StS = S.T @ S + ridge_alpha * np.eye(S.shape[1])
    W_out = np.linalg.solve(StS, S.T @ weighted_targets).T
    return W_out


def _build_windowed_input(a: np.ndarray, window: int) -> np.ndarray:
    nt, r = a.shape
    n_samples = nt - window + 1
    u = np.zeros((n_samples, window * r))
    for i in range(n_samples):
        u[i] = a[i:i + window].flatten()
    return u


def augment_features(
    a: np.ndarray,
    add_diff: bool = True,
    add_delay: bool = False,
    delay: int = 1,
) -> np.ndarray:
    nt, r = a.shape
    features = [a]
    trim = 0

    if add_diff:
        diff = np.zeros_like(a)
        diff[1:] = a[1:] - a[:-1]
        features.append(diff)

    if add_delay and delay > 0:
        trim = max(trim, delay)
        delayed = np.zeros_like(a)
        delayed[delay:] = a[:-delay]
        features.append(delayed)

    result = np.hstack(features)
    if trim > 0:
        result = result[trim:]
    return result


def fit_esn(
    a_train: np.ndarray,
    a_val: Optional[np.ndarray] = None,
    n_reservoir: int = 500,
    spectral_radius: float = 0.9,
    input_scaling: float = 1.0,
    ridge_alpha: float = 1e-2,
    leak_rate: float = 1.0,
    connectivity: float = 0.1,
    noise_level: float = 1e-4,
    washout: int = 100,
    window: int = 1,
    seed: int = 42,
    use_augmented: bool = False,
    multistep_loss: bool = False,
    output_damping: float = 0.9,
) -> Dict[str, Any]:
    t_start = time_module.time()

    nt, r = a_train.shape
    rng = np.random.RandomState(seed)

    if use_augmented:
        a_input = augment_features(a_train, add_diff=True, add_delay=False)
        n_input_features = a_input.shape[1]
    else:
        a_input = a_train
        n_input_features = r

    n_input = window * n_input_features

    W_in, W = _initialize_reservoir(
        n_input=n_input, n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        connectivity=connectivity,
        seed=seed,
    )

    if window > 1:
        u_windowed = _build_windowed_input(a_input, window)
        u_train = u_windowed[:-1]
        y_train = a_train[window:]
    else:
        u_train = a_input[:-1]
        y_train = a_train[1:]

    states = _run_reservoir(u_train, W_in, W, leak_rate=leak_rate,
                            noise_level=noise_level, rng=rng)

    states_use = states[washout:]
    targets_use = y_train[washout:]

    if multistep_loss:
        W_out_init = _train_readout(states_use, targets_use, ridge_alpha=ridge_alpha)
        W_out = _train_readout_multistep(
            u_train[washout:], targets_use, W_in, W, W_out_init,
            leak_rate=leak_rate, ridge_alpha=ridge_alpha,
            n_multistep=5, weight_decay=0.8,
        )
    else:
        W_out = _train_readout(states_use, targets_use, ridge_alpha=ridge_alpha)

    S_aug = np.hstack([states_use, np.ones((states_use.shape[0], 1))])
    y_hat = S_aug @ W_out.T
    train_rmse = float(np.sqrt(np.mean((targets_use - y_hat) ** 2)))

    t_end = time_module.time()

    return {
        "W_in": W_in,
        "W": W,
        "W_out": W_out,
        "n_reservoir": n_reservoir,
        "spectral_radius": spectral_radius,
        "input_scaling": input_scaling,
        "ridge_alpha": ridge_alpha,
        "leak_rate": leak_rate,
        "connectivity": connectivity,
        "r": r,
        "window": window,
        "washout": washout,
        "train_rmse": train_rmse,
        "train_time": t_end - t_start,
        "seed": seed,
        "use_augmented": use_augmented,
        "output_damping": float(np.clip(output_damping, 0.0, 1.0)),
    }


def predict_esn(
    esn_result: Dict[str, Any],
    a0: np.ndarray,
    n_steps: int,
    warmup: Optional[np.ndarray] = None,
) -> np.ndarray:
    W_in = esn_result["W_in"]
    W = esn_result["W"]
    W_out = esn_result["W_out"]
    leak_rate = esn_result["leak_rate"]
    r = esn_result["r"]
    n_reservoir = esn_result["n_reservoir"]
    window = esn_result.get("window", 1)
    use_augmented = esn_result.get("use_augmented", False)
    output_damping = float(np.clip(esn_result.get("output_damping", 0.9), 0.0, 1.0))

    x = np.zeros(n_reservoir)

    if window > 1:
        if warmup is not None and warmup.shape[0] >= window:
            if use_augmented:
                warmup_aug = augment_features(warmup, add_diff=True, add_delay=False)
                win_buffer = warmup_aug[-(window - 1):].tolist() + [
                    augment_features(
                        np.vstack([warmup[-1:], a0[np.newaxis, :]]),
                        add_diff=True, add_delay=False
                    )[-1].tolist()
                ]
            else:
                win_buffer = warmup[-(window - 1):].tolist() + [a0.copy()]
        else:
            if use_augmented:
                aug0 = np.zeros(r * 2)
                aug0[:r] = a0
                win_buffer = [aug0.copy()] * window
            else:
                win_buffer = [a0.copy()] * window
    else:
        win_buffer = None

    if warmup is not None:
        if use_augmented:
            warmup_aug = augment_features(warmup, add_diff=True, add_delay=False)
        else:
            warmup_aug = warmup

        if window > 1:
            warmup_windowed = _build_windowed_input(warmup_aug, window)
            for t in range(warmup_windowed.shape[0]):
                pre = W_in @ warmup_windowed[t] + W @ x
                x_new = np.tanh(pre)
                x = (1.0 - leak_rate) * x + leak_rate * x_new
        else:
            for t in range(warmup_aug.shape[0]):
                pre = W_in @ warmup_aug[t] + W @ x
                x_new = np.tanh(pre)
                x = (1.0 - leak_rate) * x + leak_rate * x_new

    a_pred = np.zeros((n_steps, r))
    u_current = a0.copy()
    prev_u = a0.copy()
    max_val = max(10.0, 5.0 * np.max(np.abs(a0)))

    for t in range(n_steps):
        a_pred[t] = u_current

        if use_augmented:
            diff = u_current - prev_u
            u_aug = np.concatenate([u_current, diff])
        else:
            u_aug = u_current

        if window > 1:
            u_input = np.array(win_buffer).flatten()
        else:
            u_input = u_aug

        pre = W_in @ u_input + W @ x
        x_new = np.tanh(pre)
        x = (1.0 - leak_rate) * x + leak_rate * x_new

        s_aug = np.append(x, 1.0)
        u_next = W_out @ s_aug
        u_next = output_damping * u_next + (1.0 - output_damping) * u_current

        u_next = np.clip(u_next, -max_val, max_val)
        if not np.all(np.isfinite(u_next)):
            a_pred[t + 1:] = u_current
            break

        prev_u = u_current.copy()
        u_current = u_next

        if window > 1:
            if use_augmented:
                diff_next = u_next - prev_u
                u_aug_next = np.concatenate([u_next, diff_next])
                win_buffer.pop(0)
                win_buffer.append(u_aug_next.copy())
            else:
                win_buffer.pop(0)
                win_buffer.append(u_next.copy())

    return a_pred


def fit_esn_with_validation(
    a_train: np.ndarray,
    a_val: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    seed: int = 42,
    n_val_rollout: int = 800,
    window: int = 1,
    n_coarse: int = 150,
    n_fine_top: int = 10,
    use_augmented: bool = False,
    multistep_loss: bool = False,
    default_leak_rate: float = 0.3,
    default_output_damping: float = 0.9,
) -> Dict[str, Any]:
    from metrics import r_squared, correlation_per_step

    r = a_train.shape[1]
    rng = np.random.RandomState(seed)

    if use_augmented:
        reservoir_choices = [200, 300, 500]
        n_coarse = min(n_coarse, 80)
    else:
        reservoir_choices = [300, 500, 800, 1000]

    search_space = {
        "n_reservoir": reservoir_choices,
        "spectral_radius": (0.7, 1.2),
        "input_scaling": (0.01, 2.0),
        "ridge_alpha": (1e-8, 1.0),
        "leak_rate": (0.1, 1.0),
        "washout": [50, 100, 200],
    }

    n_val_use = min(n_val_rollout, a_val.shape[0])

    def _sample_params() -> Dict[str, Any]:
        return {
            "n_reservoir": rng.choice(search_space["n_reservoir"]),
            "spectral_radius": rng.uniform(*search_space["spectral_radius"]),
            "input_scaling": rng.uniform(*search_space["input_scaling"]),
            "ridge_alpha": 10 ** rng.uniform(np.log10(search_space["ridge_alpha"][0]),
                                              np.log10(search_space["ridge_alpha"][1])),
            "leak_rate": rng.uniform(*search_space["leak_rate"]),
            "washout": rng.choice(search_space["washout"]),
            "output_damping": rng.uniform(0.75, 0.98),
        }

    def _evaluate_config(params: Dict[str, Any], s: int) -> Tuple[float, Optional[Dict]]:
        try:
            result = fit_esn(
                a_train,
                n_reservoir=int(params["n_reservoir"]),
                spectral_radius=float(params["spectral_radius"]),
                input_scaling=float(params["input_scaling"]),
                ridge_alpha=float(params["ridge_alpha"]),
                leak_rate=float(params["leak_rate"]),
                washout=int(params["washout"]),
                window=window,
                seed=s,
                use_augmented=use_augmented,
                multistep_loss=multistep_loss,
                output_damping=float(params.get("output_damping", default_output_damping)),
            )

            warmup_len = min(100, a_train.shape[0] // 2)
            warmup_seq = a_train[-warmup_len:]
            a_val_pred = predict_esn(result, a0=a_val[0], n_steps=n_val_use, warmup=warmup_seq)

            max_h = min(n_val_use, a_val_pred.shape[0])
            horizons = [h for h in (20, 50, 100, 200, 500, max_h) if h <= max_h]

            scores = []
            for h in horizons:
                yt = a_val[:h]
                yp = a_val_pred[:h]
                r2 = float(r_squared(yt, yp))
                corr = float(np.mean(correlation_per_step(yt, yp)))
                if h <= 100:
                    w = 0.10
                elif h <= 200:
                    w = 0.15
                elif h <= 500:
                    w = 0.30
                else:
                    w = 0.45
                scores.append(w * (0.7 * r2 + 0.3 * corr))
            score = float(np.sum(scores))

            pred_norm = np.max(np.abs(a_val_pred))
            train_norm = np.max(np.abs(a_train))
            if pred_norm > 10 * train_norm:
                score -= 1.0

            return score, result
        except Exception:
            return -999.0, None

    coarse_results: List[Tuple[float, Dict, Dict]] = []

    for i in range(n_coarse):
        params = _sample_params()
        score, result = _evaluate_config(params, seed + i)
        if result is not None:
            coarse_results.append((score, params, result))

    if not coarse_results:
        result = fit_esn(a_train, window=window, seed=seed,
                        use_augmented=use_augmented, multistep_loss=multistep_loss,
                        leak_rate=default_leak_rate,
                        output_damping=default_output_damping)
        result["val_score"] = -999.0
        result["val_params"] = {}
        result["search_phase"] = "fallback"
        return result

    coarse_results.sort(key=lambda x: x[0], reverse=True)

    top_configs = coarse_results[:n_fine_top]
    fine_results: List[Tuple[float, Dict, Dict]] = list(coarse_results[:n_fine_top])

    for rank, (base_score, base_params, _) in enumerate(top_configs):
        for _ in range(5):
            perturbed = {}
            for key, val in base_params.items():
                if key == "n_reservoir":
                    perturbed[key] = rng.choice(search_space["n_reservoir"])
                elif key == "washout":
                    perturbed[key] = rng.choice(search_space["washout"])
                elif key == "ridge_alpha":
                    log_val = np.log10(val)
                    perturbed[key] = 10 ** (log_val + rng.uniform(-0.5, 0.5))
                else:
                    factor = rng.uniform(0.8, 1.2)
                    perturbed[key] = val * factor

            perturbed["spectral_radius"] = np.clip(perturbed["spectral_radius"], 0.5, 1.5)
            perturbed["input_scaling"] = np.clip(perturbed["input_scaling"], 0.001, 5.0)
            perturbed["ridge_alpha"] = np.clip(perturbed["ridge_alpha"], 1e-10, 10.0)
            perturbed["leak_rate"] = np.clip(perturbed["leak_rate"], 0.05, 1.0)
            perturbed["output_damping"] = np.clip(perturbed.get("output_damping", default_output_damping), 0.6, 0.995)

            score, result = _evaluate_config(perturbed, seed + n_coarse + rank * 5 + _)
            if result is not None:
                fine_results.append((score, perturbed, result))

    fine_results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, best_result = fine_results[0]

    best_result["val_score"] = best_score
    best_result["val_params"] = best_params
    best_result["search_phase"] = "fine"
    best_result["n_configs_tried"] = n_coarse + n_fine_top * 5

    return best_result

def _build_delay_state(data: np.ndarray, t_idx: int, delay: int) -> np.ndarray:
    r = data.shape[1]
    z = np.zeros(r * delay, dtype=float)
    for k in range(delay):
        z[k * r:(k + 1) * r] = data[t_idx - k]
    return z


def fit_nvar(
    a_train: np.ndarray,
    delay: int = 8,
    degree: int = 2,
    ridge_alpha: float = 1e-4,
) -> Dict[str, Any]:
    t_start = time_module.time()

    n, r = a_train.shape
    delay = int(max(2, delay))
    if n <= delay + 1:
        raise ValueError("Not enough samples for NVAR with given delay")

    n_rows = n - delay
    X_lin = np.zeros((n_rows, r * delay), dtype=float)
    Y = np.zeros((n_rows, r), dtype=float)

    for i in range(n_rows):
        t = delay - 1 + i
        X_lin[i] = _build_delay_state(a_train, t, delay)
        Y[i] = a_train[t + 1]

    poly = PolynomialFeatures(degree=int(max(1, degree)), include_bias=True)
    X = poly.fit_transform(X_lin)

    reg_eye = np.eye(X.shape[1], dtype=float)
    W = np.linalg.solve(X.T @ X + float(ridge_alpha) * reg_eye, X.T @ Y)

    train_pred = X @ W
    train_rmse = float(np.sqrt(np.mean((Y - train_pred) ** 2)))

    t_end = time_module.time()
    return {
        "W": W,
        "poly": poly,
        "delay": delay,
        "degree": int(degree),
        "ridge_alpha": float(ridge_alpha),
        "r": int(r),
        "n_features": int(X.shape[1]),
        "train_rmse": train_rmse,
        "train_time": t_end - t_start,
        "method": "nvar",
    }


def predict_nvar_iterative(
    nvar: Dict[str, Any],
    n_steps: int,
    history: np.ndarray,
) -> np.ndarray:
    delay = int(nvar["delay"])
    r = int(nvar["r"])

    hist = np.array(history, dtype=float, copy=True)
    if hist.ndim != 2 or hist.shape[1] != r:
        raise ValueError("history must be shaped (delay, r)")
    if hist.shape[0] < delay:
        pad = np.repeat(hist[:1], repeats=delay - hist.shape[0], axis=0)
        hist = np.vstack([pad, hist])
    elif hist.shape[0] > delay:
        hist = hist[-delay:]

    pred = np.zeros((n_steps, r), dtype=float)
    window = hist.copy()

    W = nvar["W"]
    poly = nvar["poly"]

    for t in range(n_steps):
        pred[t] = window[-1]
        z = np.zeros(r * delay, dtype=float)
        for k in range(delay):
            z[k * r:(k + 1) * r] = window[-1 - k]
        feat = poly.transform(z.reshape(1, -1))
        x_next = (feat @ W)[0]
        window = np.vstack([window[1:], x_next[None, :]])

    return pred


def fit_nvar_with_validation(
    a_train: np.ndarray,
    a_val: np.ndarray,
    delay_candidates: Optional[List[int]] = None,
    degree_candidates: Optional[List[int]] = None,
    ridge_candidates: Optional[List[float]] = None,
    n_val_rollout: int = 1000,
) -> Dict[str, Any]:
    from metrics import correlation_per_step, r_squared

    n_val_use = min(n_val_rollout, a_val.shape[0])

    if delay_candidates is None:
        delay_candidates = [2, 4, 8]
    if degree_candidates is None:
        degree_candidates = [1]
    if ridge_candidates is None:
        ridge_candidates = [1e-3, 1e-2]

    best = None
    best_score = -np.inf

    for delay in delay_candidates:
        for degree in degree_candidates:
            for ridge in ridge_candidates:
                try:
                    model = fit_nvar(a_train, delay=delay, degree=degree, ridge_alpha=ridge)
                    hist = np.vstack([a_train[-(delay - 1):], a_val[0:1]])
                    y_pred = predict_nvar_iterative(model, n_steps=n_val_use, history=hist)

                    max_h = min(n_val_use, y_pred.shape[0])
                    horizons = [h for h in (20, 200, 600, max_h) if h <= max_h]

                    def _fast_corr(yt: np.ndarray, yp: np.ndarray) -> float:
                        if yt.shape[0] <= 120:
                            return float(np.mean(correlation_per_step(yt, yp)))
                        stride = max(1, yt.shape[0] // 120)
                        return float(np.mean(correlation_per_step(yt[::stride], yp[::stride])))

                    scores = []
                    for h in horizons:
                        yt, yp = a_val[:h], y_pred[:h]
                        r2 = float(r_squared(yt, yp))
                        corr = _fast_corr(yt, yp)
                        if h <= 20:
                            w = 0.10
                        elif h <= 200:
                            w = 0.30
                        else:
                            w = 0.60
                        scores.append(w * (0.7 * r2 + 0.3 * corr))

                    pred_norm = float(np.max(np.abs(y_pred)))
                    train_norm = float(np.max(np.abs(a_train)))
                    penalty = -0.5 if pred_norm > 5 * max(1.0, train_norm) else 0.0
                    score = float(np.sum(scores)) + penalty

                    if score > best_score:
                        best_score = score
                        model["val_score"] = score
                        model["val_delay"] = int(delay)
                        model["val_degree"] = int(degree)
                        model["val_ridge_alpha"] = float(ridge)
                        best = model
                except Exception:
                    continue

    if best is None:
        best = fit_nvar(a_train, delay=4, degree=2, ridge_alpha=1e-4)
        best["val_score"] = -999.0
        best["val_delay"] = 4
        best["val_degree"] = 2
        best["val_ridge_alpha"] = 1e-4

    return best
