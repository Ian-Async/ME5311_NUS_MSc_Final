from __future__ import annotations

"""Data I/O utilities

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def load_vectors(
    npy_path: str | Path | None = None,
    mmap: bool = True,
    dtype: str | np.dtype | None = "float32",
) -> np.ndarray:
    root = get_project_root()
    if npy_path is None:
        npy_path = root / "data" / "vector_64.npy"
    else:
        npy_path = Path(npy_path)
        if not npy_path.is_absolute():
            npy_path = (root / npy_path).resolve()

    if not npy_path.exists():
        raise FileNotFoundError(f"Data file not found: {npy_path}")

    mmap_mode = "r" if mmap else None
    vectors = np.load(str(npy_path), mmap_mode=mmap_mode)

    if vectors.ndim != 4 or vectors.shape[-1] != 2 or vectors.shape[1:3] != (64, 64):
        raise ValueError(
            "Unexpected data shape: "
            f"{vectors.shape}, expected (nt, 64, 64, 2)"
        )

    if dtype is None:
        return vectors

    if isinstance(dtype, str):
        v = dtype.strip().lower()
        if v in {"none", "null"}:
            return vectors
        if v == "float32":
            dtype = np.dtype(np.float32)
        elif v == "float64":
            dtype = np.dtype(np.float64)
        else:
            raise ValueError("dtype must be one of: None, float32, float64")
    else:
        dtype = np.dtype(dtype)

    if vectors.dtype != dtype:
        vectors = vectors.astype(dtype, copy=True)

    return vectors


def get_data_info(vectors: np.ndarray) -> Tuple[int, int, int, int, np.dtype]:
    nt, ny, nx, comps = vectors.shape
    return int(nt), int(ny), int(nx), int(comps), vectors.dtype


def quick_inspect(vectors: np.ndarray, t: int = 0, sample_frames: int = 10) -> None:
    nt = vectors.shape[0]
    sample_frames = max(1, int(sample_frames))
    if nt <= sample_frames:
        idx = np.arange(nt)
    else:
        idx = np.linspace(0, nt - 1, sample_frames, dtype=int)

    sample = vectors[idx]

    print("Loaded data with shape:", vectors.shape)
    print("dtype:", vectors.dtype)
    print(
        "min/max/mean/std (sampled):",
        float(sample.min()),
        float(sample.max()),
        float(sample.mean()),
        float(sample.std()),
    )
    if 0 <= t < nt:
        print(f"Sample block vectors[{t}, 0:3, 0:3, :]=")
        print(vectors[t, 0:3, 0:3, :])


if __name__ == "__main__":
    v = load_vectors()
    quick_inspect(v, t=0)
