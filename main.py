from __future__ import annotations

"""End-to-end analysis pipeline

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from load_data import load_vectors, get_project_root
from data_split import (
    temporal_train_val_test_split,
    prepare_pod_reduced_data,
    reconstruct_from_pod_coefficients,
)
from model_dmd import (
    fit_hankel_dmd,
    predict_hankel_dmd_iterative,
)
from model_rc import fit_nvar_with_validation, predict_nvar_iterative
from model_sindy import (
    fit_sindy, predict_sindy, predict_sindy_iterative,
    predict_sindy_rolling_restart,
    fit_sindy_with_search, preprocess_sindy_data, inverse_transform_sindy,
)
from model_rc import fit_esn, predict_esn, fit_esn_with_validation
from metrics import compute_all_metrics, compare_models, rmse, nrmse, r_squared, rmse_per_step, correlation_per_step
from visualization import plot_report_figure


MODELS_ORDERED = ["Hankel-DMD", "SINDy", "HybridRC"]


def _cleanup_png_outputs(outputs_root: Path, keep: Path) -> None:
    outputs_root = outputs_root.resolve()
    keep = keep.resolve()
    if not outputs_root.exists():
        return
    for png in outputs_root.rglob("*.png"):
        if png.resolve() != keep:
            try:
                png.unlink()
            except Exception:
                pass


def _enforce_three_output_files(outputs_root: Path, keep_files: list[Path]) -> None:
    outputs_root = outputs_root.resolve()
    keep = {p.resolve() for p in keep_files}
    if not outputs_root.exists():
        return

    for p in sorted(outputs_root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        rp = p.resolve()
        if p.is_file() and rp not in keep:
            try:
                p.unlink()
            except Exception:
                pass
        elif p.is_dir():
            try:
                p.rmdir()
            except Exception:
                pass

    for child in outputs_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _run_metadata() -> dict[str, Any]:
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    packages: dict[str, str | None] = {"numpy": np.__version__}
    for name in ("matplotlib", "scipy", "sklearn", "pysindy"):
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", None)
        except Exception:
            packages[name] = None
    return {
        "timestamp": ts,
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ME5311 Project 2: Prediction Pipeline")
    p.add_argument("--data", default="data/vector_64.npy")
    p.add_argument("--out", default="outputs")
    p.add_argument("--r", type=int, default=31)
    p.add_argument("--energy-threshold", type=float, default=0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--sindy-threshold", type=float, default=0.05)
    p.add_argument("--sindy-degree", type=int, default=1)
    p.add_argument("--sindy-alpha", type=float, default=0.1)
    p.add_argument("--sindy-rank", type=int, default=8)
    p.add_argument("--sindy-search", action="store_true")
    p.add_argument("--sindy-preprocess", action="store_true")
    p.add_argument("--esn-reservoir", type=int, default=800)
    p.add_argument("--esn-spectral-radius", type=float, default=0.7)
    p.add_argument("--esn-input-scaling", type=float, default=0.3)
    p.add_argument("--esn-ridge", type=float, default=1e-1)
    p.add_argument("--esn-window", type=int, default=10)
    p.add_argument("--esn-leak-rate", type=float, default=0.3)
    p.add_argument("--esn-output-damping", type=float, default=0.9)
    p.add_argument("--esn-search", action="store_true")
    p.add_argument("--esn-augmented", action="store_true")
    p.add_argument("--esn-multistep", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-collect-sensitivity", action="store_true")
    p.add_argument("--internal-child", action="store_true")
    p.add_argument("--skip-figure", action="store_true")
    return p


def _evaluate_at_horizons(
    a_true: np.ndarray,
    a_pred: np.ndarray,
    horizons: List[int],
    dt: float,
    model_name: str,
    train_time: float = 0.0,
) -> Dict[str, Any]:
    results = {"model": model_name, "train_time_s": train_time, "horizons": {}}
    n_max = min(a_true.shape[0], a_pred.shape[0])
    for h in horizons:
        n = min(h, n_max)
        if n < 1:
            continue
        yt, yp = a_true[:n], a_pred[:n]
        cs = correlation_per_step(yt, yp)
        results["horizons"][h] = {
            "rmse": float(rmse(yt, yp)),
            "nrmse": float(nrmse(yt, yp)),
            "r_squared": float(r_squared(yt, yp)),
            "mean_corr": float(np.mean(cs)),
        }
    return results


def _rolling_origin_evaluate(
    a_true: np.ndarray,
    predict_fn,
    horizons: List[int],
    dt: float,
    n_origins: int = 5,
) -> Dict[int, Dict[str, float]]:
    n_total = a_true.shape[0]
    max_h = max(horizons)

    available = n_total - max_h
    if available < n_origins:
        n_origins = max(1, available)
    origins = np.linspace(0, max(0, available - 1), n_origins, dtype=int)

    results = {h: {"rmse": [], "r_squared": [], "mean_corr": []} for h in horizons}

    for origin in origins:
        a0 = a_true[origin]
        remaining = n_total - origin
        n_pred = min(max_h, remaining)
        a_pred = predict_fn(a0, n_pred)

        for h in horizons:
            n = min(h, n_pred, remaining)
            if n < 2:
                continue
            yt = a_true[origin:origin + n]
            yp = a_pred[:n]
            results[h]["rmse"].append(float(rmse(yt, yp)))
            results[h]["r_squared"].append(float(r_squared(yt, yp)))
            cs = correlation_per_step(yt, yp)
            results[h]["mean_corr"].append(float(np.mean(cs)))

    aggregated = {}
    for h in horizons:
        if results[h]["rmse"]:
            aggregated[h] = {
                "rmse": float(np.mean(results[h]["rmse"])),
                "rmse_std": float(np.std(results[h]["rmse"])),
                "r_squared": float(np.mean(results[h]["r_squared"])),
                "r_squared_std": float(np.std(results[h]["r_squared"])),
                "mean_corr": float(np.mean(results[h]["mean_corr"])),
                "mean_corr_std": float(np.std(results[h]["mean_corr"])),
            }
    return aggregated


def _sanitize_prediction(a_pred: np.ndarray, reference: np.ndarray, clip_std: float = 6.0) -> np.ndarray:
    max_abs = max(1.0, float(np.max(np.abs(reference))) * clip_std)
    a_clean = np.array(a_pred, dtype=float, copy=True)
    a_clean = np.nan_to_num(a_clean, nan=0.0, posinf=max_abs, neginf=-max_abs)
    return np.clip(a_clean, -max_abs, max_abs)


def _score_validation_rollout(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    max_h = min(y_true.shape[0], y_pred.shape[0])
    horizons = [h for h in (20, 200, 600, max_h) if h <= max_h]
    if not horizons:
        horizons = [max_h]

    def _fast_corr(yt: np.ndarray, yp: np.ndarray) -> float:
        if yt.shape[0] <= 120:
            return float(np.mean(correlation_per_step(yt, yp)))
        stride = max(1, yt.shape[0] // 120)
        return float(np.mean(correlation_per_step(yt[::stride], yp[::stride])))

    scores = []
    for h in horizons:
        yt, yp = y_true[:h], y_pred[:h]
        r2 = float(r_squared(yt, yp))
        corr = _fast_corr(yt, yp)
        if h <= 100:
            w = 0.10
        elif h <= 200:
            w = 0.15
        elif h <= 500:
            w = 0.30
        else:
            w = 0.45
        scores.append(w * (0.7 * r2 + 0.3 * corr))
    return float(np.mean(scores))


def _tune_shrink_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    candidates = [1.00, 0.95, 0.90, 0.85, 0.80]
    best_alpha = 1.0
    best_score = -np.inf
    for a in candidates:
        score = _score_validation_rollout(y_true, a * y_pred)
        if score > best_score:
            best_score = score
            best_alpha = float(a)
    return best_alpha


def _build_history_window(reference: np.ndarray, current_index: int, delay: int) -> np.ndarray:
    delay = int(max(2, delay))
    if reference.shape[0] == 0:
        raise ValueError("reference must be non-empty")

    start = max(0, current_index - delay + 1)
    hist = reference[start:current_index + 1]
    if hist.shape[0] < delay:
        pad = np.repeat(hist[:1], repeats=delay - hist.shape[0], axis=0)
        hist = np.vstack([pad, hist])
    return hist


def _run_child_and_load_results(
    root: Path,
    args,
    out_dir: Path,
    r_value: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    logger: logging.Logger,
) -> Dict[str, Any] | None:
    out_abs = out_dir.resolve()
    out_abs.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((root / "main.py").resolve()),
        "--internal-child",
        "--skip-figure",
        "--no-collect-sensitivity",
        "--out", str(out_abs),
        "--data", str(args.data),
        "--r", str(int(r_value)),
        "--dt", str(float(args.dt)),
        "--train-ratio", str(float(train_ratio)),
        "--val-ratio", str(float(val_ratio)),
        "--test-ratio", str(float(test_ratio)),
        "--seed", str(int(args.seed)),
        "--n-repeats", "1",
        "--sindy-threshold", str(float(args.sindy_threshold)),
        "--sindy-degree", str(int(args.sindy_degree)),
        "--sindy-alpha", str(float(args.sindy_alpha)),
        "--sindy-rank", str(int(args.sindy_rank)),
        "--esn-reservoir", str(int(args.esn_reservoir)),
        "--esn-spectral-radius", str(float(args.esn_spectral_radius)),
        "--esn-input-scaling", str(float(args.esn_input_scaling)),
        "--esn-ridge", str(float(args.esn_ridge)),
        "--esn-window", str(int(args.esn_window)),
    ]
    for flag in (
        "sindy_search", "sindy_preprocess", "esn_search", "esn_augmented",
        "esn_multistep",
    ):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))

    logger.info("  child run: r=%d, split=%.2f/%.2f/%.2f", r_value, train_ratio, val_ratio, test_ratio)
    completed = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        logger.warning("  child run failed (r=%d)", r_value)
        return None

    result_file = out_abs / "results.json"
    if not result_file.exists():
        logger.warning("  child run missing results: %s", result_file)
        return None
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("  failed to load child results %s: %s", result_file, e)
        return None


def _collect_sensitivity_results(root: Path, args, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Collecting sensitivity results (embedded in outputs/results.json)...")
    legacy_tmp = (root / "outputs" / "_tmp_sensitivity").resolve()
    if legacy_tmp.exists():
        shutil.rmtree(legacy_tmp, ignore_errors=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="me5311_sensitivity_"))
    try:
        rank_results: Dict[str, Any] = {}
        for rr in (4, 10, 31):
            res = _run_child_and_load_results(
                root=root,
                args=args,
                out_dir=tmp_root / f"r{rr}",
                r_value=rr,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                logger=logger,
            )
            if res is not None:
                rank_results[str(rr)] = {
                    "pod": res.get("pod", {}),
                    "models": res.get("models", {}),
                    "horizons": res.get("horizons", {}),
                    "rolling_origin": res.get("rolling_origin", {}),
                    "physical_space": res.get("physical_space", {}),
                }

        train_results: Dict[str, Any] = {}
        for frac in (0.20, 0.40, 0.60):
            res = _run_child_and_load_results(
                root=root,
                args=args,
                out_dir=tmp_root / f"train{int(frac*100)}",
                r_value=31,
                train_ratio=frac,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                logger=logger,
            )
            if res is not None:
                train_results[f"{frac:.2f}"] = {
                    "pod": res.get("pod", {}),
                    "models": res.get("models", {}),
                    "horizons": res.get("horizons", {}),
                    "rolling_origin": res.get("rolling_origin", {}),
                    "physical_space": res.get("physical_space", {}),
                }

        return {
            "rank": rank_results,
            "train_fraction": train_results,
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("me5311_p2")
    np.random.seed(args.seed)

    root = get_project_root()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (root / data_path).resolve()
    if args.internal_child:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = (root / out_dir).resolve()
    else:
        out_dir = root / "outputs"
        if Path(args.out).resolve() != out_dir.resolve():
            logger.info("Ignoring --out=%s to enforce single output folder: %s", args.out, out_dir)
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = args.dt

    logger.info("Loading data: %s", data_path)
    vectors = load_vectors(npy_path=data_path, mmap=False, dtype="float32")
    nt, ny, nx, ncomp = vectors.shape
    logger.info("Shape: %s, dtype: %s", vectors.shape, vectors.dtype)

    split = temporal_train_val_test_split(
        vectors, args.train_ratio, args.val_ratio, args.test_ratio,
    )
    logger.info("Train=%d, Val=%d, Test=%d",
                split["n_train"], split["n_val"], split["n_test"])

    r = args.r
    energy_threshold = args.energy_threshold if (args.energy_threshold and args.energy_threshold > 0) else None
    pod_data = prepare_pod_reduced_data(
        split, r=r, energy_threshold=energy_threshold,
        center=True, method="randomized", seed=args.seed,
    )
    r = pod_data["r"]
    cum_e = float(pod_data["energy_cum"][-1])
    logger.info("POD r=%d captures %.2f%% of training energy", r, cum_e * 100)

    a_train = pod_data["a_train"]
    a_val = pod_data["a_val"]
    a_test = pod_data["a_test"]
    U = pod_data["U"]
    mean_flat = pod_data["mean_flat"]

    n_test = a_test.shape[0]
    n_val = a_val.shape[0]
    horizons = [20, 200, 2000]
    rolling_horizons = [20, 200, 2000]

    all_preds = {}
    all_metrics = {}
    all_horizons = {}
    all_rolling = {}

    logger.info("--- Hankel-DMD ---")
    n_val_sel_dmd = min(n_val, 1000)
    delay_candidates = [12, 16, 20]
    damping_candidates = [0.995, 0.999]
    best_hankel = None
    best_delay = 16
    best_damping = 0.999
    dmd_alpha = 1.0
    hankel_val_score = -np.inf

    for dly in delay_candidates:
        for dmp in damping_candidates:
            try:
                hankel_try = fit_hankel_dmd(
                    a_train,
                    dt=dt,
                    delay=dly,
                    damping=dmp,
                )
                hankel_hist_val_try = np.vstack([a_train[-(dly - 1):], a_val[0:1]])
                a_hankel_val_try = predict_hankel_dmd_iterative(
                    hankel_try,
                    n_steps=n_val_sel_dmd,
                    history=hankel_hist_val_try,
                )
                alpha_try = _tune_shrink_factor(a_val[:n_val_sel_dmd], a_hankel_val_try)
                score_try = _score_validation_rollout(a_val[:n_val_sel_dmd], alpha_try * a_hankel_val_try)
                if score_try > hankel_val_score:
                    hankel_val_score = float(score_try)
                    dmd_alpha = float(alpha_try)
                    best_hankel = hankel_try
                    best_delay = int(dly)
                    best_damping = float(dmp)
            except Exception as e:
                logger.warning("  DMD candidate delay=%d damping=%.3f failed: %s", dly, dmp, e)

    if best_hankel is None:
        hankel = fit_hankel_dmd(a_train, dt=dt, delay=16, damping=0.999)
        best_delay = 16
        best_damping = 0.999
        hankel_hist_val = np.vstack([a_train[-(best_delay - 1):], a_val[0:1]])
        a_hankel_val = predict_hankel_dmd_iterative(
            hankel,
            n_steps=n_val_sel_dmd,
            history=hankel_hist_val,
        )
        dmd_alpha = _tune_shrink_factor(a_val[:n_val_sel_dmd], a_hankel_val)
        hankel_val_score = _score_validation_rollout(a_val[:n_val_sel_dmd], dmd_alpha * a_hankel_val)
    else:
        hankel = best_hankel

    hankel_delay = best_delay
    logger.info(
        "  Selected delay=%d, damping=%.3f, shrink alpha=%.3f, val_score=%.4f",
        hankel_delay,
        best_damping,
        dmd_alpha,
        hankel_val_score,
    )

    hankel_hist_test = np.vstack([a_val[-(hankel_delay - 1):], a_test[0:1]])
    a_dmd = dmd_alpha * predict_hankel_dmd_iterative(hankel, n_steps=n_test, history=hankel_hist_test)

    hankel["variant"] = "hankel-dmd"
    hankel["validation_score_hankel"] = float(hankel_val_score)
    hankel["selected_delay"] = int(best_delay)
    hankel["selected_damping"] = float(best_damping)

    all_preds["Hankel-DMD"] = a_dmd
    all_metrics["Hankel-DMD"] = compute_all_metrics(
        a_test, a_dmd, dt=dt, model_name="Hankel-DMD", train_time=hankel["train_time"]
    )
    all_horizons["Hankel-DMD"] = _evaluate_at_horizons(
        a_test, a_dmd, horizons, dt, "Hankel-DMD", hankel["train_time"]
    )

    def _hankel_predict_fn(a0, n_steps):
        test_ref = np.vstack([a_val, a_test])
        try:
            matched = np.where(np.all(np.isclose(a_test, a0, atol=1e-10), axis=1))[0]
            origin = int(matched[0]) if matched.size > 0 else 0
        except Exception:
            origin = 0
        current_index = a_val.shape[0] + origin
        hist = _build_history_window(test_ref, current_index, hankel_delay)
        pred = predict_hankel_dmd_iterative(hankel, n_steps=n_steps, history=hist)
        return dmd_alpha * pred

    all_rolling["Hankel-DMD"] = _rolling_origin_evaluate(a_test, _hankel_predict_fn, rolling_horizons, dt)

    for h, m in all_horizons["Hankel-DMD"]["horizons"].items():
        logger.info("  h=%d: RMSE=%.3f, R2=%.4f, corr=%.4f", h, m["rmse"], m["r_squared"], m["mean_corr"])
    logger.info("  Rolling-origin (mean):")
    for h, m in all_rolling["Hankel-DMD"].items():
        logger.info("    h=%d: R2=%.4f+/-%.4f, corr=%.4f+/-%.4f",
                    h, m["r_squared"], m["r_squared_std"], m["mean_corr"], m["mean_corr_std"])

    logger.info("--- SINDy ---")
    sindy_ok = True
    sindy_degree = args.sindy_degree
    sindy_threshold = args.sindy_threshold
    sindy_r = min(args.sindy_rank if args.sindy_rank > 0 else r, r)
    rank_candidates = sorted(set([
        max(2, sindy_r - 2),
        sindy_r,
        min(r, sindy_r + 2),
        min(r, 4),
    ]))
    if sindy_r < r:
        logger.info("  SINDy initial rank=%d (of %d POD modes)", sindy_r, r)
    logger.info("  SINDy rank candidates: %s", rank_candidates)

    sindy_scaler = {"mean": None, "std": None, "denoised": False, "standardized": False}
    sindy_alpha = 1.0
    best_fn = predict_sindy_iterative
    best_mode = "iterative"
    best_rank_score = -np.inf

    try:
        best_package = None
        n_val_sel = min(n_val, 1000)

        for rk in rank_candidates:
            a_train_s = a_train[:, :rk]
            a_val_s = a_val[:, :rk]
            a_test_s = a_test[:, :rk]

            threshold_candidates = sorted(set([sindy_threshold, 0.005, 0.01, 0.02, 0.05]))
            local_best_bundle = None
            local_score = -np.inf

            for thr in threshold_candidates:
                local_scaler = {"mean": None, "std": None, "denoised": False, "standardized": False}

                if args.sindy_search and rk == sindy_r:
                    logger.info("  Running SINDy hyperparameter search at rank=%d...", rk)
                    local_sindy, local_scaler = fit_sindy_with_search(
                        a_train_s, a_val_s, dt=dt, sindy_r=rk,
                        preprocess=args.sindy_preprocess,
                    )
                else:
                    if args.sindy_preprocess:
                        a_train_sp, local_scaler = preprocess_sindy_data(a_train_s)
                    else:
                        a_train_sp = a_train_s
                    local_sindy = fit_sindy(
                        a_train_sp, dt=dt,
                        threshold=float(thr),
                        poly_degree=sindy_degree,
                        alpha=args.sindy_alpha,
                        smooth_derivatives=True,
                    )

                if local_scaler.get("standardized") and local_scaler["mean"] is not None:
                    a_val_sp = (a_val_s - local_scaler["mean"]) / local_scaler["std"]
                    a_test_sp = (a_test_s - local_scaler["mean"]) / local_scaler["std"]
                else:
                    a_val_sp = a_val_s
                    a_test_sp = a_test_s

                local_candidates = {}
                mode_predictors = {
                    "iterative": lambda model, a0, n_steps, dt_step: predict_sindy_iterative(model, a0=a0, n_steps=n_steps, dt=dt_step),
                    "iterative_damped": lambda model, a0, n_steps, dt_step: 0.97 * predict_sindy_iterative(
                        model, a0=a0, n_steps=n_steps, dt=dt_step
                    ),
                    "iterative_damped_strong": lambda model, a0, n_steps, dt_step: 0.94 * predict_sindy_iterative(
                        model, a0=a0, n_steps=n_steps, dt=dt_step
                    ),
                }
                if rk >= 12:
                    mode_predictors["global"] = (
                        lambda model, a0, n_steps, dt_step: predict_sindy(
                            model, a0=a0, n_steps=n_steps, dt=dt_step
                        )
                    )

                for mode_name, predict_fn in mode_predictors.items():
                    vp = predict_fn(local_sindy, a_val_sp[0], n_val_sel, dt)
                    vp_orig = inverse_transform_sindy(vp, local_scaler) if local_scaler.get("standardized") else vp
                    vp_orig = _sanitize_prediction(vp_orig, a_train_s)
                    alpha = _tune_shrink_factor(a_val_s[:n_val_sel], vp_orig)
                    vp_scaled = alpha * vp_orig
                    score = _score_validation_rollout(a_val_s[:n_val_sel], vp_scaled)

                    h_long = min(2000, n_val_sel)
                    if h_long >= 200:
                        r2_long = float(r_squared(a_val_s[:h_long], vp_scaled[:h_long]))
                        corr_long = float(np.mean(correlation_per_step(a_val_s[:h_long], vp_scaled[:h_long])))
                        long_score = 0.7 * r2_long + 0.3 * corr_long
                        score += 0.20 * long_score
                        if r2_long < 0:
                            score -= min(0.8, 0.25 + 0.20 * abs(r2_long))

                    local_candidates[mode_name] = {
                        "predict_fn": predict_fn,
                        "alpha": alpha,
                        "score": score,
                    }

                mode_best = max(local_candidates, key=lambda k: local_candidates[k]["score"])
                score_best = float(local_candidates[mode_best]["score"])
                if score_best > local_score:
                    local_score = score_best
                    local_best_bundle = {
                        "thr": float(thr),
                        "mode": mode_best,
                        "predict_fn": local_candidates[mode_best]["predict_fn"],
                        "alpha": float(local_candidates[mode_best]["alpha"]),
                        "sindy": local_sindy,
                        "scaler": local_scaler,
                        "a_train_s": a_train_s,
                        "a_test_sp": a_test_sp,
                    }

            logger.info("  rank=%d: thr=%.3f, mode=%s, score=%.4f",
                        rk, local_best_bundle["thr"], local_best_bundle["mode"], local_score)

            if local_score > best_rank_score:
                best_rank_score = local_score
                best_package = {
                    "rank": rk,
                    "threshold": local_best_bundle["thr"],
                    "sindy": local_best_bundle["sindy"],
                    "scaler": local_best_bundle["scaler"],
                    "a_train_s": local_best_bundle["a_train_s"],
                    "a_test_sp": local_best_bundle["a_test_sp"],
                    "best_mode": local_best_bundle["mode"],
                    "best_fn": local_best_bundle["predict_fn"],
                    "alpha": local_best_bundle["alpha"],
                }

        if best_package is None:
            raise RuntimeError("No valid SINDy model found across rank candidates")

        sindy_r = int(best_package["rank"])
        sindy = best_package["sindy"]
        sindy_scaler = best_package["scaler"]
        a_train_s = best_package["a_train_s"]
        a_test_sp = best_package["a_test_sp"]
        best_mode = best_package["best_mode"]
        best_fn = best_package["best_fn"]
        sindy_alpha = float(best_package["alpha"])

        logger.info("  Selected rank=%d, mode=%s (score=%.4f, alpha=%.3f)",
                    sindy_r, best_mode, best_rank_score, sindy_alpha)
        logger.info("  Selected SINDy threshold=%.3f", float(best_package["threshold"]))
        logger.info("  Fitted in %.3fs, %d active terms", sindy["train_time"], sindy["n_terms"])

        a_sindy_sub = best_fn(sindy, a_test_sp[0], n_test, dt)
        if sindy_scaler.get("standardized"):
            a_sindy_sub = inverse_transform_sindy(a_sindy_sub, sindy_scaler)
        a_sindy_sub = _sanitize_prediction(a_sindy_sub, a_train_s, clip_std=3.0)
        a_sindy_sub = sindy_alpha * a_sindy_sub

        if sindy_r < r:
            a_sindy = np.zeros((n_test, r))
            a_sindy[:, :sindy_r] = a_sindy_sub
            a_sindy[:, sindy_r:] = a_test[0, sindy_r:]
        else:
            a_sindy = a_sindy_sub

        sindy_metrics = compute_all_metrics(a_test, a_sindy, dt=dt, model_name="SINDy", train_time=sindy["train_time"])
        sindy_metrics["sindy_rank"] = sindy_r
        sindy_metrics["forecast_mode"] = best_mode

    except Exception as e:
        logger.warning("  SINDy failed: %s", e)
        sindy_ok = False
        a_sindy = np.tile(a_test[0], (n_test, 1))
        sindy_metrics = compute_all_metrics(a_test, a_sindy, dt=dt, model_name="SINDy", train_time=0.0)

    all_preds["SINDy"] = a_sindy
    all_metrics["SINDy"] = sindy_metrics
    all_horizons["SINDy"] = _evaluate_at_horizons(a_test, a_sindy, horizons, dt, "SINDy", sindy_metrics["train_time_s"])

    def _sindy_predict_fn(a0, n_steps):
        if sindy_scaler.get("standardized") and sindy_scaler["mean"] is not None:
            a0_p = (a0[:sindy_r] - sindy_scaler["mean"]) / sindy_scaler["std"]
        else:
            a0_p = a0[:sindy_r]
        pred_sub = best_fn(sindy, a0_p, n_steps, dt)
        if sindy_scaler.get("standardized"):
            pred_sub = inverse_transform_sindy(pred_sub, sindy_scaler)
        pred_sub = _sanitize_prediction(pred_sub, a_train_s, clip_std=3.0)
        pred_sub = sindy_alpha * pred_sub
        if sindy_r < r:
            pred_full = np.zeros((n_steps, r))
            pred_full[:, :sindy_r] = pred_sub
            pred_full[:, sindy_r:] = a0[sindy_r:]
            return pred_full
        return pred_sub

    if sindy_ok:
        all_rolling["SINDy"] = _rolling_origin_evaluate(a_test, _sindy_predict_fn, rolling_horizons, dt)
    else:
        all_rolling["SINDy"] = {}

    for h, m in all_horizons["SINDy"]["horizons"].items():
        logger.info("  h=%d: RMSE=%.3f, R2=%.4f, corr=%.4f", h, m["rmse"], m["r_squared"], m["mean_corr"])
    if all_rolling.get("SINDy"):
        logger.info("  Rolling-origin (mean):")
        for h, m in all_rolling["SINDy"].items():
            logger.info("    h=%d: R2=%.4f+/-%.4f, corr=%.4f+/-%.4f",
                        h, m["r_squared"], m["r_squared_std"], m["mean_corr"], m["mean_corr_std"])

    logger.info("--- RC (ESN, window=%d) ---", args.esn_window)

    n_repeats = args.n_repeats
    rc_repeat_results = []

    for rep in range(n_repeats):
        rep_seed = args.seed + rep * 1000
        np.random.seed(rep_seed)

        if args.esn_search:
            if rep == 0:
                logger.info("  Running hyperparameter search (repeat %d/%d)...", rep + 1, n_repeats)
            esn = fit_esn_with_validation(
                a_train, a_val, seed=rep_seed, window=args.esn_window,
                use_augmented=args.esn_augmented,
                multistep_loss=args.esn_multistep,
                default_leak_rate=args.esn_leak_rate,
                default_output_damping=args.esn_output_damping,
            )
        else:
            local_candidates = [
                {
                    "n_reservoir": args.esn_reservoir,
                    "spectral_radius": args.esn_spectral_radius,
                    "input_scaling": args.esn_input_scaling,
                    "ridge_alpha": args.esn_ridge,
                    "leak_rate": args.esn_leak_rate,
                    "output_damping": args.esn_output_damping,
                    "multistep_loss": args.esn_multistep,
                },
                {
                    "n_reservoir": args.esn_reservoir,
                    "spectral_radius": max(0.55, args.esn_spectral_radius - 0.10),
                    "input_scaling": max(0.05, args.esn_input_scaling * 0.75),
                    "ridge_alpha": args.esn_ridge * 2.0,
                    "leak_rate": max(0.10, args.esn_leak_rate - 0.10),
                    "output_damping": min(0.98, args.esn_output_damping + 0.04),
                    "multistep_loss": True,
                },
                {
                    "n_reservoir": args.esn_reservoir,
                    "spectral_radius": min(0.95, args.esn_spectral_radius + 0.08),
                    "input_scaling": min(1.20, args.esn_input_scaling * 1.20),
                    "ridge_alpha": max(1e-4, args.esn_ridge * 0.6),
                    "leak_rate": min(0.65, args.esn_leak_rate + 0.10),
                    "output_damping": max(0.75, args.esn_output_damping - 0.05),
                    "multistep_loss": args.esn_multistep,
                },
                {
                    "n_reservoir": args.esn_reservoir + 200,
                    "spectral_radius": max(0.55, args.esn_spectral_radius - 0.08),
                    "input_scaling": max(0.05, args.esn_input_scaling * 0.85),
                    "ridge_alpha": args.esn_ridge * 3.0,
                    "leak_rate": max(0.10, args.esn_leak_rate - 0.15),
                    "output_damping": min(0.99, args.esn_output_damping + 0.06),
                    "multistep_loss": True,
                },
                {
                    "n_reservoir": max(300, args.esn_reservoir - 200),
                    "spectral_radius": max(0.50, args.esn_spectral_radius - 0.15),
                    "input_scaling": max(0.03, args.esn_input_scaling * 0.60),
                    "ridge_alpha": args.esn_ridge * 5.0,
                    "leak_rate": max(0.08, args.esn_leak_rate - 0.18),
                    "output_damping": min(0.995, args.esn_output_damping + 0.08),
                    "multistep_loss": True,
                },
            ]

            warmup_val_len = min(300, a_train.shape[0])
            warmup_val = a_train[-warmup_val_len:]
            n_val_sel_rc = min(n_val, 2000)
            best_local = None
            best_local_score = -np.inf

            for idx, cfg in enumerate(local_candidates):
                esn_try = fit_esn(
                    a_train,
                    n_reservoir=int(cfg["n_reservoir"]),
                    spectral_radius=float(cfg["spectral_radius"]),
                    input_scaling=float(cfg["input_scaling"]),
                    ridge_alpha=float(cfg["ridge_alpha"]),
                    leak_rate=float(cfg["leak_rate"]),
                    window=args.esn_window,
                    seed=rep_seed + idx,
                    use_augmented=args.esn_augmented,
                    multistep_loss=bool(cfg.get("multistep_loss", args.esn_multistep)),
                    output_damping=float(cfg["output_damping"]),
                )

                val_pred = predict_esn(esn_try, a0=a_val[0], n_steps=n_val_sel_rc, warmup=warmup_val)
                alpha_try = _tune_shrink_factor(a_val[:n_val_sel_rc], val_pred)
                val_pred_scaled = alpha_try * val_pred
                score_try = _score_validation_rollout(a_val[:n_val_sel_rc], val_pred_scaled)
                pred_norm = float(np.max(np.abs(val_pred_scaled)))
                ref_norm = float(np.max(np.abs(a_train)))
                if pred_norm > 8.0 * max(1.0, ref_norm):
                    score_try -= 0.5

                if score_try > best_local_score:
                    best_local_score = score_try
                    best_local = (esn_try, alpha_try)

            esn, local_alpha = best_local
            esn["shrink_alpha"] = local_alpha
            if rep == 0:
                logger.info("  Local ESN refinement score=%.4f", best_local_score)

        warmup_len = min(200, a_val.shape[0])
        warmup_seq = a_val[-warmup_len:]
        n_val_sel_rc = min(n_val, 2000)
        a_rc_val = predict_esn(esn, a0=a_val[0], n_steps=n_val_sel_rc, warmup=warmup_seq)
        rc_alpha = _tune_shrink_factor(a_val[:n_val_sel_rc], a_rc_val)
        if rep == 0:
            logger.info("  RC validation shrink alpha=%.3f", rc_alpha)

        a_rc = rc_alpha * predict_esn(esn, a0=a_test[0], n_steps=n_test, warmup=warmup_seq)
        esn["shrink_alpha"] = rc_alpha

        rc_metrics = compute_all_metrics(a_test, a_rc, dt=dt, model_name="RC", train_time=esn["train_time"])
        rc_horizons = _evaluate_at_horizons(a_test, a_rc, horizons, dt, "RC", esn["train_time"])

        rc_repeat_results.append({
            "a_pred": a_rc,
            "metrics": rc_metrics,
            "horizons": rc_horizons,
            "esn": esn,
        })

        if n_repeats > 1:
            logger.info("  Repeat %d/%d: R2=%.4f, VPT=%.1f",
                        rep + 1, n_repeats,
                        rc_metrics["r_squared"],
                        rc_metrics["valid_prediction_time"])

    rc_r2s = [rr["metrics"]["r_squared"] for rr in rc_repeat_results]
    rc_vpts = [rr["metrics"]["valid_prediction_time"] for rr in rc_repeat_results]

    best_rc_idx = int(np.argmax(rc_r2s))
    best_rc = rc_repeat_results[best_rc_idx]

    a_rc = best_rc["a_pred"]
    esn = best_rc["esn"]

    logger.info("  Fitted in %.3fs, reservoir=%d, SR=%.2f, window=%d, train_rmse=%.6f",
                esn["train_time"], esn["n_reservoir"], esn["spectral_radius"],
                esn.get("window", 1), esn["train_rmse"])

    if n_repeats > 1:
        logger.info("  RC over %d repeats: R2=%.4f+/-%.4f, VPT=%.1f+/-%.1f",
                    n_repeats,
                    float(np.mean(rc_r2s)), float(np.std(rc_r2s)),
                    float(np.mean(rc_vpts)), float(np.std(rc_vpts)))

    rc_val_score = _score_validation_rollout(a_val[:n_val_sel_rc], esn.get("shrink_alpha", 1.0) * a_rc_val)

    logger.info("--- HybridRC (RC vs NVAR) ---")
    nvar = fit_nvar_with_validation(
        a_train,
        a_val,
        n_val_rollout=min(n_val, 1000),
    )
    nvar_delay = int(nvar.get("val_delay", nvar.get("delay", 4)))
    nvar_hist_val = np.vstack([a_train[-(nvar_delay - 1):], a_val[0:1]])
    a_nvar_val = predict_nvar_iterative(nvar, n_steps=min(n_val, 1000), history=nvar_hist_val)
    nvar_alpha = _tune_shrink_factor(a_val[:a_nvar_val.shape[0]], a_nvar_val)
    nvar_val_score = _score_validation_rollout(a_val[:a_nvar_val.shape[0]], nvar_alpha * a_nvar_val)
    logger.info(
        "  NVAR delay=%d, degree=%s, alpha=%.3f, val_score=%.4f",
        nvar_delay,
        nvar.get("val_degree", nvar.get("degree")),
        nvar_alpha,
        nvar_val_score,
    )

    best_family_name = "HybridRC"
    if nvar_val_score > rc_val_score:
        logger.info("  Selected backend: NVAR (better than RC on validation)")
        nvar_hist_test = np.vstack([a_val[-(nvar_delay - 1):], a_test[0:1]])
        a_best_family = nvar_alpha * predict_nvar_iterative(nvar, n_steps=n_test, history=nvar_hist_test)
        family_metrics = compute_all_metrics(
            a_test, a_best_family, dt=dt, model_name=best_family_name, train_time=nvar["train_time"]
        )
        family_metrics["selected_backend"] = "NVAR"
        family_metrics["backend_val_score"] = float(nvar_val_score)
        family_metrics["delay"] = int(nvar_delay)
        family_metrics["degree"] = int(nvar.get("val_degree", nvar.get("degree", 2)))
        family_metrics["ridge_alpha"] = float(nvar.get("val_ridge_alpha", nvar.get("ridge_alpha", 1e-4)))
        family_metrics["shrink_alpha"] = float(nvar_alpha)

        def _family_predict_fn(a0, n_steps):
            test_ref = np.vstack([a_val, a_test])
            try:
                matched = np.where(np.all(np.isclose(a_test, a0, atol=1e-10), axis=1))[0]
                origin = int(matched[0]) if matched.size > 0 else 0
            except Exception:
                origin = 0
            current_index = a_val.shape[0] + origin
            hist = _build_history_window(test_ref, current_index, nvar_delay)
            return nvar_alpha * predict_nvar_iterative(nvar, n_steps=n_steps, history=hist)

    else:
        logger.info("  Selected backend: RC (better than NVAR on validation)")
        a_best_family = a_rc
        family_metrics = best_rc["metrics"]
        family_metrics["model"] = best_family_name
        family_metrics["selected_backend"] = "RC"
        family_metrics["backend_val_score"] = float(rc_val_score)
        family_metrics["shrink_alpha"] = float(esn.get("shrink_alpha", 1.0))
        family_metrics["n_reservoir"] = esn.get("n_reservoir")
        family_metrics["spectral_radius"] = esn.get("spectral_radius")
        family_metrics["window"] = esn.get("window", 1)

        def _family_predict_fn(a0, n_steps):
            return esn.get("shrink_alpha", 1.0) * predict_esn(esn, a0=a0, n_steps=n_steps, warmup=warmup_seq)

    all_preds[best_family_name] = a_best_family
    all_metrics[best_family_name] = family_metrics
    all_horizons[best_family_name] = _evaluate_at_horizons(
        a_test, a_best_family, horizons, dt, best_family_name, family_metrics["train_time_s"]
    )
    all_rolling[best_family_name] = _rolling_origin_evaluate(a_test, _family_predict_fn, rolling_horizons, dt)

    for h, m in all_horizons[best_family_name]["horizons"].items():
        logger.info("  h=%d: RMSE=%.3f, R2=%.4f, corr=%.4f", h, m["rmse"], m["r_squared"], m["mean_corr"])
    logger.info("  Rolling-origin (mean):")
    for h, m in all_rolling[best_family_name].items():
        logger.info("    h=%d: R2=%.4f+/-%.4f, corr=%.4f+/-%.4f",
                    h, m["r_squared"], m["r_squared_std"], m["mean_corr"], m["mean_corr_std"])

    metrics_list = [all_metrics[m] for m in MODELS_ORDERED]
    comparison = compare_models(metrics_list)

    logger.info("=== Full test comparison (%d steps = %.0f t.u.) ===", n_test, n_test * dt)
    for i, name in enumerate(comparison["models"]):
        logger.info("  %-6s RMSE=%.3f R2=%.4f corr=%.4f VPT=%.1f train=%.2fs",
                     name, comparison["rmse"][i], comparison["r_squared"][i],
                     comparison["mean_correlation"][i],
                     comparison["valid_prediction_time"][i], comparison["train_time_s"][i])

    t_snap = min(50, n_test - 1)
    test_vec_true = split["test"][t_snap]
    vel_mag_true = np.sqrt(test_vec_true[:, :, 0]**2 + test_vec_true[:, :, 1]**2)

    predictions = {}
    for mname in MODELS_ORDERED:
        predictions[mname] = {"a_pred": all_preds[mname], "metrics": all_metrics[mname]}

    field_comparisons = {"true": vel_mag_true}
    for mname in MODELS_ORDERED:
        ap = all_preds[mname]
        if ap.shape[0] > t_snap:
            vec_p = reconstruct_from_pod_coefficients(
                ap[t_snap:t_snap + 1], U, mean_flat, ny, nx, center=True,
            )
            field_comparisons[mname] = np.sqrt(vec_p[0, :, :, 0]**2 + vec_p[0, :, :, 1]**2)
    predictions["_fields"] = field_comparisons

    logger.info("=== Physical-space evaluation ===")
    test_flat = split["test"].reshape(n_test, -1).astype(np.float64)
    ref_mean = mean_flat.astype(np.float64)[np.newaxis, :]
    std_phys = float(np.std(test_flat - ref_mean))
    phys_horizons = [h for h in horizons if h <= n_test]
    phys_results = {}

    for mname in MODELS_ORDERED:
        phys_results[mname] = {}
        ap = all_preds[mname]
        n_avail = min(n_test, ap.shape[0])
        for h in phys_horizons:
            n = min(h, n_avail)
            true_h = test_flat[:n]
            pred_h = ap[:n].astype(np.float64) @ U.T + ref_mean
            diff = true_h - pred_h
            rmse_p = float(np.sqrt(np.mean(diff ** 2)))
            nrmse_p = rmse_p / std_phys if std_phys > 1e-12 else float("inf")
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((true_h - ref_mean) ** 2))
            r2_p = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            phys_results[mname][h] = {"rmse": rmse_p, "nrmse": nrmse_p, "r_squared": r2_p}
            logger.info("  %s h=%d phys: RMSE=%.3f, NRMSE=%.3f, R2=%.4f",
                         mname, h, rmse_p, nrmse_p, r2_p)
    predictions["_phys"] = phys_results

    lines = []
    lines.append("=" * 72)
    lines.append("ME5311 Project 2  —  Prediction Results Summary")
    lines.append("=" * 72)
    lines.append(f"\nDATA & SETUP")
    lines.append(f"  Shape: {tuple(vectors.shape)}, dt={dt}")
    lines.append(f"  Train/Val/Test: {split['n_train']}/{split['n_val']}/{split['n_test']}")
    lines.append(f"  POD rank r={r}, energy: {cum_e*100:.2f}%")
    lines.append(f"  Hankel-DMD delay={hankel_delay}")
    lines.append(f"  SINDy rank={sindy_r}")
    lines.append(f"  HybridRC backend={all_metrics['HybridRC'].get('selected_backend', 'RC')}")
    if n_repeats > 1:
        lines.append(f"  RC repeats: {n_repeats} (report mean +/- std)")

    lines.append(f"\n{'-'*72}")
    lines.append("PREDICTION AT MULTIPLE HORIZONS")
    lines.append("-" * 72)
    for mname in MODELS_ORDERED:
        mh = all_horizons[mname]
        lines.append(f"\n  {mname}:")
        lines.append(f"  {'Horizon':>8} {'RMSE':>10} {'NRMSE':>10} {'R2':>10} {'Corr':>10}")
        for h, m in mh["horizons"].items():
            lines.append(f"  {h:>8} {m['rmse']:10.4f} {m['nrmse']:10.4f} "
                         f"{m['r_squared']:10.4f} {m['mean_corr']:10.4f}")

    lines.append(f"\n{'-'*72}")
    lines.append("ROLLING-ORIGIN MULTI-STEP EVALUATION (mean +/- std)")
    lines.append("-" * 72)
    for mname in MODELS_ORDERED:
        ro = all_rolling.get(mname, {})
        if ro:
            lines.append(f"\n  {mname}:")
            lines.append(f"  {'Horizon':>8} {'R2':>14} {'Corr':>14}")
            for h, m in ro.items():
                lines.append(f"  {h:>8} {m['r_squared']:7.4f}+/-{m['r_squared_std']:.4f}"
                             f" {m['mean_corr']:7.4f}+/-{m['mean_corr_std']:.4f}")

    if n_repeats > 1:
        lines.append(f"\n{'-'*72}")
        lines.append(f"RC REPEATED RUNS ({n_repeats} seeds)")
        lines.append("-" * 72)
        lines.append(f"  R2:  {np.mean(rc_r2s):.4f} +/- {np.std(rc_r2s):.4f}")
        lines.append(f"  VPT: {np.mean(rc_vpts):.1f} +/- {np.std(rc_vpts):.1f}")

    lines.append(f"\n{'-'*72}")
    lines.append("OVERALL COMPARISON")
    lines.append("-" * 72)
    lines.append(f"  {'Model':<12} {'RMSE':>10} {'NRMSE':>8} {'R2':>10} "
                 f"{'Corr':>8} {'VPT':>8} {'Time(s)':>8}")
    for i, name in enumerate(comparison["models"]):
        lines.append(
            f"  {name:<12} {comparison['rmse'][i]:10.4f} {comparison['nrmse'][i]:8.4f} "
            f"{comparison['r_squared'][i]:10.4f} {comparison['mean_correlation'][i]:8.4f} "
            f"{comparison['valid_prediction_time'][i]:8.1f} {comparison['train_time_s'][i]:8.2f}")

    lines.append(f"\nBest VPT: {comparison['best_valid_time']}")
    lines.append(f"Best R2: {comparison['best_r_squared']}")
    lines.append(f"Fastest: {comparison['fastest_training']}")
    lines.append("=" * 72)

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary: %s", out_dir / "summary.txt")

    rj = {
        "schema_version": 4,
        "project": "ME5311 Project 2",
        "run": _run_metadata(),
        "data": {
            "shape": list(vectors.shape), "dtype": str(vectors.dtype), "dt": dt,
            "n_train": split["n_train"], "n_val": split["n_val"], "n_test": split["n_test"],
        },
        "pod": {
            "r": r,
            "energy": [float(v) for v in pod_data["energy"]],
            "energy_cum": [float(v) for v in pod_data["energy_cum"]],
        },
        "models": {},
        "horizons": {},
        "rolling_origin": {},
        "physical_space": {},
    }
    for mname in MODELS_ORDERED:
        m = all_metrics[mname]
        rj["models"][mname] = {
            "rmse": m["rmse_global"], "nrmse": m["nrmse_global"],
            "mae": m["mae_global"], "r_squared": m["r_squared"],
            "relative_energy_error": m["relative_energy_error"],
            "valid_prediction_time": m["valid_prediction_time"],
            "mean_correlation": m["mean_correlation"],
            "train_time_s": m["train_time_s"],
        }
        rj["horizons"][mname] = {str(h): v for h, v in all_horizons[mname]["horizons"].items()}
        rj["rolling_origin"][mname] = {str(h): v for h, v in all_rolling.get(mname, {}).items()}
        rj["physical_space"][mname] = {str(h): v for h, v in phys_results.get(mname, {}).items()}

    rj["models"]["Hankel-DMD"]["method"] = hankel.get("method", "hankel-dmd")
    rj["models"]["Hankel-DMD"]["delay"] = int(hankel_delay)
    rj["models"]["Hankel-DMD"]["state_dim"] = int(hankel.get("state_dim", r * hankel_delay))
    rj["models"]["Hankel-DMD"]["r_dmd"] = int(hankel.get("r_dmd", r))
    rj["models"]["Hankel-DMD"]["shrink_alpha"] = float(dmd_alpha)
    rj["models"]["Hankel-DMD"]["validation_score"] = float(hankel.get("validation_score_hankel", 0.0))
    rj["models"]["SINDy"]["sindy_rank"] = sindy_r
    rj["models"]["SINDy"]["forecast_mode"] = sindy_metrics.get("forecast_mode")
    rj["models"]["SINDy"]["shrink_alpha"] = float(sindy_alpha)
    if sindy_ok and hasattr(sindy, 'get'):
        rj["models"]["SINDy"]["search_config"] = sindy.get("search_config")
    rj["models"]["HybridRC"]["selected_backend"] = all_metrics["HybridRC"].get("selected_backend")
    rj["models"]["HybridRC"]["backend_val_score"] = float(all_metrics["HybridRC"].get("backend_val_score", 0.0))
    rj["models"]["HybridRC"]["shrink_alpha"] = float(all_metrics["HybridRC"].get("shrink_alpha", 1.0))
    if all_metrics["HybridRC"].get("selected_backend") == "RC":
        rj["models"]["HybridRC"]["n_reservoir"] = esn.get("n_reservoir")
        rj["models"]["HybridRC"]["spectral_radius"] = esn.get("spectral_radius")
        rj["models"]["HybridRC"]["window"] = esn.get("window", 1)
        rj["models"]["HybridRC"]["val_score"] = esn.get("val_score")
    else:
        rj["models"]["HybridRC"]["delay"] = int(nvar_delay)
        rj["models"]["HybridRC"]["degree"] = int(nvar.get("val_degree", nvar.get("degree", 2)))
        rj["models"]["HybridRC"]["ridge_alpha"] = float(nvar.get("val_ridge_alpha", nvar.get("ridge_alpha", 1e-4)))
    if n_repeats > 1:
        rj["models"]["HybridRC"]["n_repeats"] = n_repeats
        rj["models"]["HybridRC"]["rc_r_squared_mean"] = float(np.mean(rc_r2s))
        rj["models"]["HybridRC"]["rc_r_squared_std"] = float(np.std(rc_r2s))
        rj["models"]["HybridRC"]["rc_vpt_mean"] = float(np.mean(rc_vpts))
        rj["models"]["HybridRC"]["rc_vpt_std"] = float(np.std(rc_vpts))

    if (not args.internal_child) and (not args.no_collect_sensitivity):
        sensitivity = _collect_sensitivity_results(root, args, logger)
        rj["sensitivity"] = sensitivity
        rank_keys = sorted(list(sensitivity.get("rank", {}).keys()), key=lambda x: int(x))
        frac_keys = sorted(list(sensitivity.get("train_fraction", {}).keys()), key=lambda x: float(x))
        lines.append("\nSENSITIVITY RESULTS (embedded in JSON)")
        lines.append("-" * 72)
        lines.append(f"  Rank sensitivity: {', '.join(rank_keys) if rank_keys else 'none'}")
        lines.append(f"  Train-fraction sensitivity: {', '.join(frac_keys) if frac_keys else 'none'}")
        (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(rj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    logger.info("Results JSON: %s", results_path)

    single_fig_path = root / "outputs" / "figure.png"
    if not args.skip_figure:
        logger.info("Generating single report figure: %s", single_fig_path)
        _cleanup_png_outputs(root / "outputs", keep=single_fig_path)
        try:
            plot_report_figure(out_path=single_fig_path, primary_r=31)
        except Exception as e:
            logger.warning("Figure generation failed, continue cleanup: %s", e)

    summary_path = out_dir / "summary.txt"
    if not args.internal_child:
        _enforce_three_output_files(
            root / "outputs",
            keep_files=[single_fig_path, results_path, summary_path],
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
