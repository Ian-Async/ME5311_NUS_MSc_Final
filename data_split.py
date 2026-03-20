from __future__ import annotations

"""Data splitting and POD projection utilities

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


def temporal_train_val_test_split(
    vectors: np.ndarray,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> Dict[str, Any]:
    assert train_ratio + val_ratio + test_ratio <= 1.0 + 1e-6, \
        "Ratios must sum to <= 1.0"
    nt = vectors.shape[0]

    n_test = int(nt * test_ratio)
    n_val = int(nt * val_ratio)
    n_train = int(nt * train_ratio)

    test = vectors[nt - n_test:]
    val = vectors[nt - n_test - n_val:nt - n_test]
    train = vectors[:n_train]

    return {
        "train": train,
        "val": val,
        "test": test,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "nt": nt,
    }


def project_to_pod_coefficients(
    vectors: np.ndarray,
    U: np.ndarray,
    mean_flat: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    nt = vectors.shape[0]
    ny, nx = vectors.shape[1], vectors.shape[2]
    n_dof = ny * nx * 2
    r = U.shape[1]

    X = vectors.reshape(nt, n_dof)

    if center:
        X = X - mean_flat[np.newaxis, :]

    a = X @ U

    return a


def reconstruct_from_pod_coefficients(
    a: np.ndarray,
    U: np.ndarray,
    mean_flat: np.ndarray,
    ny: int,
    nx: int,
    center: bool = True,
) -> np.ndarray:
    nt = a.shape[0]
    X_rec = a @ U.T

    if center:
        X_rec = X_rec + mean_flat[np.newaxis, :]

    vectors_rec = X_rec.reshape(nt, ny, nx, 2)
    return vectors_rec


def prepare_pod_reduced_data(
    split: Dict[str, Any],
    r: int = 20,
    energy_threshold: Optional[float] = None,
    center: bool = True,
    method: str = "randomized",
    seed: int = 0,
) -> Dict[str, Any]:
    from pod_svd import build_snapshot_matrix, compute_svd, select_rank_by_energy

    train = split["train"]
    nt_train, ny, nx, _ = train.shape
    n_dof = ny * nx * 2

    X_train, mean_flat = build_snapshot_matrix(train, center=center)

    if energy_threshold is not None:
        r = select_rank_by_energy(
            X_train, energy_threshold=energy_threshold,
            r_max=200, method=method, seed=seed,
        )

    svd = compute_svd(X_train, r=r, method=method, seed=seed)

    U = svd["U"]
    s = svd["s"]
    energy = svd["energy"]
    energy_cum = svd["energy_cum"]

    a_train = project_to_pod_coefficients(split["train"], U, mean_flat, center=center)
    a_val = project_to_pod_coefficients(split["val"], U, mean_flat, center=center)
    a_test = project_to_pod_coefficients(split["test"], U, mean_flat, center=center)

    return {
        "a_train": a_train,
        "a_val": a_val,
        "a_test": a_test,
        "U": U,
        "s": s,
        "mean_flat": mean_flat,
        "energy": energy,
        "energy_cum": energy_cum,
        "r": int(svd["r"]),
        "ny": ny,
        "nx": nx,
        "n_dof": n_dof,
        "center": center,
    }
