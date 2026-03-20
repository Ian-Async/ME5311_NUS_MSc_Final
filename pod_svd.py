from __future__ import annotations

"""POD/SVD operators

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import numpy as np


def build_snapshot_matrix(vectors: np.ndarray, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    nt, ny, nx, ncomp = vectors.shape
    if ncomp != 2:
        raise ValueError(f"Expected 2 components, got {ncomp}")

    X = vectors.reshape(nt, ny * nx * 2).T

    if center:
        mean_field_flat = X.mean(axis=1)
        X = X - mean_field_flat[:, None]
    else:
        mean_field_flat = np.zeros((ny * nx * 2,), dtype=X.dtype)

    return X, mean_field_flat


def compute_svd(
    X: np.ndarray,
    r: int = 20,
    method: str = "randomized",
    seed: int = 0,
) -> dict:
    if r < 1:
        raise ValueError("r must be >= 1")

    d, nt = X.shape
    r_eff = min(int(r), d, nt)

    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or Inf")

    total_energy = float(np.sum(X.astype(np.float64) ** 2))
    if total_energy <= 0.0:
        raise ValueError("Total energy is non-positive; check input data")

    if method not in {"randomized", "full"}:
        raise ValueError("method must be 'randomized' or 'full'")

    if method == "full":
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        U_r = U[:, :r_eff]
        s_r = s[:r_eff]
        Vt_r = Vt[:r_eff, :]
    else:
        rng = np.random.RandomState(seed)
        oversample = min(10, max(2, min(d, nt) - r_eff))
        l = r_eff + oversample
        omega = rng.standard_normal(size=(nt, l))
        Y = X @ omega
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ X
        Uhat, s, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat
        U_r = U[:, :r_eff]
        s_r = s[:r_eff]
        Vt_r = Vt[:r_eff, :]

    energy_r = (s_r**2) / total_energy
    energy_cum_r = np.cumsum(energy_r)

    return {
        "U": U_r,
        "s": s_r,
        "Vt": Vt_r,
        "energy": energy_r,
        "energy_cum": energy_cum_r,
        "energy_total": total_energy,
        "r": int(r_eff),
    }


def select_rank_by_energy(
    X: np.ndarray,
    energy_threshold: float = 0.95,
    r_max: int = 100,
    method: str = "randomized",
    seed: int = 0,
) -> int:
    svd = compute_svd(X, r=r_max, method=method, seed=seed)
    cum = svd["energy_cum"]
    for i, e in enumerate(cum):
        if e >= energy_threshold:
            return i + 1
    return len(cum)


def reconstruct_mode_field(U_col: np.ndarray, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    mode = U_col.reshape(ny, nx, 2)
    mode_x = mode[:, :, 0]
    mode_y = mode[:, :, 1]
    return mode_x, mode_y


def reconstruct_multiple_modes(
    U: np.ndarray, ny: int, nx: int, n_modes: int = 4
) -> list[dict]:
    n_modes = min(n_modes, U.shape[1])
    modes = []
    for i in range(n_modes):
        mx, my = reconstruct_mode_field(U[:, i], ny, nx)
        modes.append({
            "ux": mx,
            "uy": my,
            "umag": np.sqrt(mx**2 + my**2),
            "mode_idx": i,
        })
    return modes


def compute_pod(
    vectors: np.ndarray,
    r: int = 10,
    center: bool = True,
    method: str = "randomized",
    seed: int = 0,
) -> dict:
    nt, ny, nx, _ = vectors.shape
    X, mean_flat = build_snapshot_matrix(vectors, center=center)
    svd = compute_svd(X, r=r, method=method, seed=seed)

    return {
        "ny": ny,
        "nx": nx,
        "nt": nt,
        "center": center,
        "mean_flat": mean_flat,
        "centered_mean_field": mean_flat,
        **svd,
    }
