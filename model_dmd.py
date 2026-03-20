from __future__ import annotations

"""Dynamic Mode Decomposition (DMD) model

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import numpy as np
from typing import Dict, Any, Optional
import time as time_module


def _build_hankel_state(data: np.ndarray, t_idx: int, delay: int) -> np.ndarray:
    r = data.shape[1]
    z = np.zeros(r * delay, dtype=float)
    for k in range(delay):
        z[k * r:(k + 1) * r] = data[t_idx - k]
    return z


def fit_hankel_dmd(
    a_train: np.ndarray,
    dt: float = 0.2,
    delay: int = 8,
    r_dmd: Optional[int] = None,
    damping: float = 0.999,
) -> Dict[str, Any]:
    t_start = time_module.time()

    n, r = a_train.shape
    delay = int(max(2, delay))
    if n <= delay + 1:
        raise ValueError("Not enough samples for Hankel-DMD with given delay")

    cols = n - delay
    X = np.zeros((r * delay, cols), dtype=float)
    Y = np.zeros((r * delay, cols), dtype=float)

    for j in range(cols):
        t = delay - 1 + j
        X[:, j] = _build_hankel_state(a_train, t, delay)
        Y[:, j] = _build_hankel_state(a_train, t + 1, delay)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if r_dmd is not None:
        r_keep = int(min(max(2, r_dmd), len(S)))
        U = U[:, :r_keep]
        S = S[:r_keep]
        Vt = Vt[:r_keep, :]

    S_inv = np.diag(1.0 / (S + 1e-12))
    A_tilde = U.T @ Y @ Vt.T @ S_inv

    eigenvalues, W = np.linalg.eig(A_tilde)
    Phi = Y @ Vt.T @ S_inv @ W

    eigenvalues_raw = eigenvalues.copy()
    magnitudes = np.abs(eigenvalues)
    unstable_mask = magnitudes > 1.0
    n_clipped = int(np.sum(unstable_mask))
    if np.any(unstable_mask):
        eigenvalues[unstable_mask] = (
            eigenvalues[unstable_mask] / magnitudes[unstable_mask] * damping
        )

    A_tilde_stable = (Phi @ np.diag(eigenvalues) @ np.linalg.pinv(Phi)).real
    omega = np.log(eigenvalues + 1e-15) / dt

    t_end = time_module.time()
    return {
        "A_tilde": A_tilde_stable,
        "eigenvalues": eigenvalues,
        "eigenvalues_raw": eigenvalues_raw,
        "omega": omega,
        "Phi": Phi,
        "dt": dt,
        "delay": delay,
        "state_dim": int(r * delay),
        "r_dmd": int(Phi.shape[1]),
        "n_clipped": n_clipped,
        "train_time": t_end - t_start,
        "method": "hankel-dmd",
        "orig_rank": int(r),
    }


def predict_hankel_dmd_iterative(
    dmd: Dict[str, Any],
    n_steps: int,
    history: np.ndarray,
) -> np.ndarray:
    A_tilde = dmd["A_tilde"]
    delay = int(dmd["delay"])
    r = int(dmd["orig_rank"])

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

    for t in range(n_steps):
        pred[t] = window[-1]
        z = np.zeros(r * delay, dtype=float)
        for k in range(delay):
            z[k * r:(k + 1) * r] = window[-1 - k]
        z_next = A_tilde @ z
        x_next = z_next[:r].real
        window = np.vstack([window[1:], x_next[None, :]])

    return pred
