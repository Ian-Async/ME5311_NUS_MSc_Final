from __future__ import annotations

"""Visualization routines

Author: Cao Yiyang
Student ID: A0329403J
Affiliation: National University of Singapore (NUS)
Coursework: ME5311 Project 2
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODELS_ORDERED = ["Hankel-DMD", "SINDy", "HybridRC"]
COLORS = {"Hankel-DMD": "#3B6FB6", "SINDy": "#D07C2C", "HybridRC": "#3E8F6A"}
MARKERS = {"Hankel-DMD": "o", "SINDy": "s", "HybridRC": "^"}
RANKS = [4, 10, 31]
RANK_COLORS = {4: "#8F2F3A", 10: "#6D5AA8", 31: "#6D4C41"}
RANK_MARKERS = {4: "o", 10: "s", 31: "^"}
PRIMARY_R = 31
TRAIN_FRACS = [0.20, 0.40, 0.60]
ROOT = Path(__file__).parent


def _apply_journal_axis_style(ax: plt.Axes, with_grid: bool = True) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
    ax.tick_params(direction="in", length=2, width=0.5)
    if with_grid:
        ax.grid(True, color="#D9D9D9", alpha=0.45, linewidth=0.5)


def plot_project2_figure(
    pod_data: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    predictions: Dict[str, Dict[str, np.ndarray]],
    a_test_true: np.ndarray,
    dt: float = 0.2,
    out_path: str | Path = "outputs/figure.png",
    n_pod_modes_show: int = 4,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "legend.title_fontsize": 5,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "figure.edgecolor": "none",
        "savefig.edgecolor": "none",
    })

    fig = plt.figure(figsize=(7.60, 9.2), facecolor="white")
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0.0)
    gs = fig.add_gridspec(3, 2, wspace=0.22, hspace=0.42,
                          left=0.0738, right=0.9425, top=0.960, bottom=0.045)

    ax_a = fig.add_subplot(gs[0, 0])
    energy = np.asarray(pod_data.get("energy", []), dtype=float)
    energy_cum = np.asarray(pod_data.get("energy_cum", []), dtype=float)
    r_modes = np.arange(1, len(energy) + 1)

    if energy.size > 0:
        ax_a.bar(r_modes, energy, color="steelblue", alpha=0.7, width=0.6)
    if energy_cum.size > 0:
        line_cum, = ax_a.plot(r_modes, energy_cum, "o-", color="darkorange", markersize=2.5,
                              linewidth=1.0, label="Cumulative energy")
        ax_a.axhline(0.85, color="green", ls="--", lw=0.7, alpha=0.7)

    r_used = pod_data.get("r", len(energy))
    r_used = int(np.clip(r_used, 1, max(1, len(energy))))
    ax_a.axvline(r_used, color="crimson", ls="-.", lw=0.8, alpha=0.75)

    ax_a.set_xlabel("Mode index $r$", fontsize=7)
    ax_a.set_ylabel("Energy fraction", fontsize=7)
    ax_a.set_title("(a) POD energy spectrum (training data)", fontsize=8,
                    fontweight="bold", y=1.02, pad=4)
    if energy_cum.size > 0:
        ax_a.legend(handles=[line_cum], labels=["Cumulative energy"],
                    loc="lower right", fontsize=5.2,
                    frameon=True, framealpha=0.85, fancybox=True)
    ax_a.set_xlim(0.3, len(energy) + 0.7)
    ax_a.set_ylim(0.0, min(1.02, max(0.9, float(np.max(energy_cum)) + 0.08 if energy_cum.size else 1.0)))
    ax_a.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_a.grid(True, alpha=0.2, linewidth=0.5)

    ax_b = fig.add_subplot(gs[0, 1])
    max_show_b = min(500, a_test_true.shape[0])

    for res in all_metrics:
        name = res["model"]
        rmse_steps = np.array(res["rmse_per_step"])
        n_show = min(max_show_b, len(rmse_steps))
        t_axis = np.arange(n_show) * dt
        ax_b.plot(t_axis, rmse_steps[:n_show], "-", lw=0.9,
                  color=COLORS.get(name, "gray"),
                  label=name, alpha=0.85)

    ax_b.set_xlabel("Prediction horizon (time units)", fontsize=7)
    ax_b.set_ylabel("RMSE (POD coefficients)", fontsize=7)
    ax_b.set_title("(b) Prediction error vs. time", fontsize=8,
                    fontweight="bold", pad=4)
    ax_b.legend(loc="upper left", fontsize=5, frameon=True, fancybox=True,
                framealpha=0.8)
    ax_b.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_b.grid(True, alpha=0.2, linewidth=0.5)

    ax_c = fig.add_subplot(gs[1, 0])
    n_show_c = min(200, a_test_true.shape[0])
    t_axis_c = np.arange(n_show_c) * dt

    ax_c.plot(t_axis_c, a_test_true[:n_show_c, 0], "k-", lw=1.0,
              label="True $a_1(t)$", alpha=0.8)

    for name in MODELS_ORDERED:
        pred_dict = predictions.get(name, {})
        a_pred = pred_dict.get("a_pred")
        if a_pred is not None and a_pred.shape[0] >= n_show_c:
            ax_c.plot(t_axis_c, a_pred[:n_show_c, 0], "--", lw=0.8,
                      color=COLORS.get(name, "gray"),
                      label=f"{name}", alpha=0.8)

    ax_c.set_xlabel("Time (time units)", fontsize=7)
    ax_c.set_ylabel("$a_1(t)$", fontsize=7)
    ax_c.set_title("(c) Leading POD coefficient: true vs. predicted",
                    fontsize=8, fontweight="bold", pad=4)
    ax_c.legend(loc="lower left", fontsize=4.8,
                frameon=True, fancybox=True, framealpha=0.85, ncol=3)
    ax_c.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_c.grid(True, alpha=0.2, linewidth=0.5)

    ax_d_outer = fig.add_subplot(gs[1, 1])
    ax_d_outer.set_axis_off()
    ax_d_outer.set_title("(d) Velocity-magnitude snapshots at $t_{test}=50\\Delta t$",
                         fontsize=8, fontweight="bold", y=1.06, pad=4)

    gs_d = gs[1, 1].subgridspec(3, 3,
                                height_ratios=[1.0, 1.0, 0.08],
                                hspace=0.18, wspace=0.12)

    field_data = predictions.get("_fields", {})
    true_field = field_data.get("true")
    if true_field is not None:
        vmin, vmax = float(true_field.min()), float(true_field.max())
    else:
        vmin, vmax = None, None

    ax_ref = fig.add_subplot(gs_d[0, :])
    im = None
    if true_field is not None:
        im = ax_ref.imshow(true_field, origin="lower", cmap="inferno",
                           aspect="equal", interpolation="none",
                           vmin=vmin, vmax=vmax)
        ax_ref.set_title("Reference", fontsize=6, pad=2)
        x_ticks = [0, true_field.shape[1] // 2, true_field.shape[1] - 1]
        y_ticks = [0, true_field.shape[0] // 2, true_field.shape[0] - 1]
        ax_ref.set_xticks(x_ticks)
        ax_ref.set_yticks(y_ticks)
    else:
        ax_ref.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax_ref.transAxes, fontsize=6)
        ax_ref.set_title("Reference", fontsize=6, pad=2)
        ax_ref.set_xticks([])
        ax_ref.set_yticks([])
    ax_ref.tick_params(labelsize=4, direction="in", length=1.5, pad=1)
    ax_ref.set_xlabel("x", fontsize=5, labelpad=1)
    ax_ref.set_ylabel("y", fontsize=5, labelpad=1)

    for idx, mname in enumerate(MODELS_ORDERED):
        ax_field = fig.add_subplot(gs_d[1, idx])
        field = field_data.get(mname)
        if field is not None:
            im = ax_field.imshow(field, origin="lower", cmap="inferno",
                                 aspect="equal", interpolation="none",
                                 vmin=vmin, vmax=vmax)
            x_ticks = [0, field.shape[1] // 2, field.shape[1] - 1]
            y_ticks = [0, field.shape[0] // 2, field.shape[0] - 1]
            ax_field.set_xticks(x_ticks)
            ax_field.set_yticks(y_ticks)
        else:
            ax_field.text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax_field.transAxes, fontsize=6)
            ax_field.set_xticks([])
            ax_field.set_yticks([])
        ax_field.set_title(mname, fontsize=6, pad=2)
        ax_field.tick_params(labelsize=4, direction="in", length=1.5, pad=1)
        ax_field.set_xlabel("x", fontsize=5, labelpad=1)
        ax_field.set_ylabel("y", fontsize=5, labelpad=1)

    if im is not None:
        cax = fig.add_subplot(gs_d[2, :])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Velocity magnitude", fontsize=5, labelpad=1)
        cb.ax.tick_params(labelsize=4, length=1.5, pad=1)

    ax_e = fig.add_subplot(gs[2, 0])
    max_show_e = min(500, a_test_true.shape[0])

    for res in all_metrics:
        name = res["model"]
        corr_steps = np.array(res["correlation_per_step"])
        n_show = min(max_show_e, len(corr_steps))
        t_axis_e = np.arange(n_show) * dt
        ax_e.plot(t_axis_e, corr_steps[:n_show], "-", lw=0.9,
                  color=COLORS.get(name, "gray"),
                  label=name, alpha=0.85)

    ax_e.axhline(0.9, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax_e.text(max_show_e * dt * 0.98, 0.91, "0.9", fontsize=4.5,
              color="gray", ha="right")
    ax_e.set_xlabel("Prediction horizon (time units)", fontsize=7)
    ax_e.set_ylabel("Correlation", fontsize=7)
    ax_e.set_title("(e) Prediction correlation vs. time", fontsize=8,
                    fontweight="bold", pad=4)
    ax_e.legend(loc="lower left", fontsize=5, frameon=True, fancybox=True,
                framealpha=0.8)
    ax_e.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_e.grid(True, alpha=0.2, linewidth=0.5)
    ax_e.set_ylim(-0.3, 1.05)

    ax_f = fig.add_subplot(gs[2, 1])
    from metrics import r_squared as r2_fn

    horizons_line = [10, 20, 50, 100, 200, 500]

    all_r2_vals = []
    for model_name in MODELS_ORDERED:
        r2_vals = []
        h_valid = []
        for h in horizons_line:
            pred_dict = predictions.get(model_name, {})
            a_pred = pred_dict.get("a_pred")
            if a_pred is not None:
                n = min(h, a_test_true.shape[0], a_pred.shape[0])
                if n >= 2:
                    r2 = float(r2_fn(a_test_true[:n], a_pred[:n]))
                    r2_vals.append(r2)
                    h_valid.append(h * dt)
                    all_r2_vals.append(r2)
        if h_valid:
            ax_f.plot(h_valid, r2_vals, "-" + MARKERS.get(model_name, "o"),
                      color=COLORS.get(model_name, "gray"), markersize=4,
                      linewidth=1.0, label=model_name, alpha=0.85,
                      markeredgewidth=0.4, markeredgecolor="white")

    ax_f.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_f.axhline(0.9, color="gray", lw=0.4, ls=":", alpha=0.4)
    ax_f.set_xlabel("Prediction horizon (time units)", fontsize=7)
    ax_f.set_ylabel("$R^2$", fontsize=7)
    ax_f.set_title("(f) $R^2$ vs. prediction horizon", fontsize=8,
                    fontweight="bold", pad=4)
    ax_f.legend(loc="upper right", fontsize=5, frameon=True, fancybox=True,
                framealpha=0.85)
    ax_f.tick_params(direction="in", length=2, width=0.5, labelsize=6)
    ax_f.grid(True, alpha=0.2, linewidth=0.5)
    r2_floor = min(-0.3, min(all_r2_vals) - 0.1) if all_r2_vals else -0.3
    ax_f.set_ylim(max(r2_floor, -2.0), 1.1)

    fig.savefig(out_path, dpi=600,
                edgecolor="none", facecolor="white")
    plt.close(fig)
    print(f"[Visualization] Saved Project 2 figure: {out_path}")


def _load_rank_data() -> dict:
    rank_data = {}
    for r in RANKS:
        p = ROOT / "outputs" / f"r{r}" / "results.json"
        if p.exists():
            with open(p) as f:
                rank_data[r] = json.load(f)
        else:
            print(f"[warn] {p} not found, skipping r={r}")
    return rank_data


def _load_train_sensitivity() -> dict:
    train_data = {}
    for frac in TRAIN_FRACS:
        pct = int(frac * 100)
        p = ROOT / "outputs" / f"train{pct}" / "results.json"
        if p.exists():
            with open(p) as f:
                train_data[frac] = json.load(f)
    return train_data


def _load_rank_result(r: int) -> dict | None:
    p = ROOT / f"outputs/r{r}" / "results.json"
    if not p.exists():
        print(f"[warn] {p} not found, skipping r={r}")
        return None
    with open(p) as f:
        return json.load(f)


def _get_r2(data: dict, model: str, horizon: int) -> float | None:
    try:
        return data["horizons"][model][str(horizon)]["r_squared"]
    except (KeyError, TypeError):
        return None


def _get_corr(data: dict, model: str, horizon: int) -> float | None:
    try:
        return data["horizons"][model][str(horizon)]["mean_corr"]
    except (KeyError, TypeError):
        return None


def _get_nrmse(data: dict, model: str, horizon: int) -> float | None:
    try:
        return data["horizons"][model][str(horizon)]["nrmse"]
    except (KeyError, TypeError):
        return None


def _format_time_label(seconds: float) -> str:
    s = float(seconds)
    if s <= 0:
        return "0s"
    if s < 0.1:
        return f"{s * 1000:.1f}ms"
    return f"{s:.2f}s"


def plot_report_figure(
    out_path: str | Path | None = None,
    primary_r: int = PRIMARY_R,
) -> None:
    """Generate the 6-panel report figure.

    Panels:
      (a) POD energy spectrum with vertical lines at r=4, 10, 31
    (b) Main-rank prediction error evolution (Q2/Q3)
    (c) Main-rank accuracy-cost comparison (Q1/Q6)
    (d) Rank sensitivity at r=4/10/31 (Q5/Q6)
    (e) Training-fraction sensitivity (Q5/Q6)
    (f) Summary table at h=20/200/2000 (Q2/Q4)
    """
    rank_data = _load_rank_data()
    train_data = _load_train_sensitivity()

    primary = rank_data.get(primary_r)
    if primary is None:
        if rank_data:
            fallback_r = max(rank_data.keys())
            primary = rank_data[fallback_r]
            print(f"[warn] r={primary_r} not found, fallback to r={fallback_r}")
        else:
            p = ROOT / "outputs" / "results.json"
            if p.exists():
                with open(p) as f:
                    primary = json.load(f)
                print("[warn] rank folders not complete, using outputs/results.json")
            else:
                print("[error] No results found. Run main.py first.")
                return

    embedded = primary.get("sensitivity", {}) if isinstance(primary, dict) else {}
    if (not rank_data) and embedded.get("rank"):
        for rk, rv in embedded.get("rank", {}).items():
            try:
                rank_data[int(rk)] = rv
            except Exception:
                continue

    if (not train_data) and embedded.get("train_fraction"):
        for fk, fv in embedded.get("train_fraction", {}).items():
            try:
                train_data[float(fk)] = fv
            except Exception:
                continue

    dt = primary["data"]["dt"]
    r_val = primary["pod"]["r"]

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.2,
        "legend.handlelength": 1.2,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#BFBFBF",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "#D9D9D9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.60, 9.2), facecolor="white")
    gs = fig.add_gridspec(3, 2, wspace=0.28, hspace=0.46,
                          left=0.08, right=0.96, top=0.955, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    spectrum_src = rank_data.get(31, primary)
    energy = spectrum_src["pod"]["energy"]
    energy_cum = spectrum_src["pod"]["energy_cum"]
    modes = np.arange(1, len(energy) + 1)

    ax_a.bar(modes, energy, color="steelblue", alpha=0.65, width=0.6,
             label="Individual mode")
    ax_a.plot(modes, energy_cum, "o-", color="darkorange", markersize=2.2,
              linewidth=1.0, label="Cumulative")

    energy_cum_arr = np.asarray(energy_cum, dtype=float)
    label_y = {4: 0.55, 10: 0.45, 31: 0.35}
    for r_mark in RANKS:
        rd = rank_data.get(r_mark)
        if rd is not None:
            cum_at_r = rd["pod"]["energy_cum"][-1]
        elif r_mark <= len(energy_cum_arr):
            cum_at_r = float(energy_cum_arr[r_mark - 1])
        else:
            continue
        ax_a.axvline(r_mark, color=RANK_COLORS[r_mark], ls="--", lw=0.8, alpha=0.75)
        if r_mark == 31:
            x_text = r_mark - 0.6
            ha = "right"
        else:
            x_text = r_mark + 0.5
            ha = "left"
        ax_a.text(x_text, label_y.get(r_mark, 0.4),
                  f"r={r_mark}\n({cum_at_r*100:.1f}%)",
                  color=RANK_COLORS[r_mark], fontsize=4.5, va="center", ha=ha)

    ax_a.set_xlabel("Mode index $r$")
    ax_a.set_ylabel("Energy fraction")
    ax_a.set_title("(a) POD energy spectrum",
                   fontsize=8, fontweight="bold", y=1.02, pad=4)
    ax_a.legend(loc="right", fontsize=5, frameon=True, framealpha=0.85)
    ax_a.set_xlim(0.3, len(energy) + 0.7)
    ax_a.set_ylim(0, min(1.02, max(0.9, energy_cum[-1] + 0.08)))
    _apply_journal_axis_style(ax_a, with_grid=True)

    ax_b = fig.add_subplot(gs[0, 1])
    horizons_line = [20, 200, 2000]

    for model in MODELS_ORDERED:
        hor_data = primary["horizons"].get(model, {})
        h_vals, rmse_vals = [], []
        for h in horizons_line:
            entry = hor_data.get(str(h))
            if entry and "rmse" in entry:
                h_vals.append(h * dt)
                rmse_vals.append(entry["rmse"])
        if h_vals:
            ax_b.plot(h_vals, rmse_vals, "-" + MARKERS[model],
                      color=COLORS[model], markersize=4, linewidth=1.1,
                      label=model, alpha=0.85,
                      markeredgewidth=0.4, markeredgecolor="white")

    ax_b.set_xlabel("Prediction horizon (time units)")
    ax_b.set_ylabel("RMSE (POD coefficients)")
    ax_b.set_title(f"(b) Main-rank error evolution (r={r_val})",
                   fontsize=8, fontweight="bold", pad=4)
    ax_b.legend(loc="upper left", fontsize=5.2, frameon=True, framealpha=0.85)
    ax_b.set_xscale("log")
    _apply_journal_axis_style(ax_b, with_grid=True)

    ax_c = fig.add_subplot(gs[1, 0])
    x = np.arange(len(MODELS_ORDERED))
    bar_w = 0.32

    vpts = [primary["models"][m]["valid_prediction_time"] for m in MODELS_ORDERED]
    train_times = [primary["models"][m]["train_time_s"] for m in MODELS_ORDERED]
    model_colors = [COLORS[m] for m in MODELS_ORDERED]

    bars_vpt = ax_c.bar(
        x, vpts, bar_w,
        color=model_colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    for bar, val in zip(bars_vpt, vpts):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                  f"{val:.1f}", ha="center", fontsize=5.2, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.75))

    ax_c2 = ax_c.twinx()
    line_time, = ax_c2.plot(
        x,
        train_times,
        color="#202020",
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        marker="D",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label="Train time",
        zorder=6,
    )

    positive_times = [t for t in train_times if t > 0]
    y2_min = 1e-4
    if positive_times:
        y2_min = max(min(positive_times) * 0.6, 1e-4)
        y2_max = max(positive_times) * 2.0
        ax_c2.set_yscale("log")
        ax_c2.set_ylim(y2_min, y2_max)

    sindy_idx = MODELS_ORDERED.index("SINDy") if "SINDy" in MODELS_ORDERED else -1
    for i, (xi, val) in enumerate(zip(x, train_times)):
        if i == sindy_idx and val > 0:
            y_txt = max(val / 1.28, y2_min * 1.05)
            va = "top"
        else:
            y_txt = val * 1.22 if val > 0 else 1e-4
            va = "bottom"
        ax_c2.text(xi, y_txt,
                   _format_time_label(val), ha="center", va=va, fontsize=4.8, color="#202020",
                   bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(MODELS_ORDERED, fontsize=6)
    ax_c.set_ylabel("Valid prediction time (t.u.)", fontsize=6.5)
    ax_c2.set_ylabel("Training time (s, log)", fontsize=6.5, color="#202020")
    ax_c2.tick_params(axis="y", labelcolor="#202020", labelsize=5)
    ax_c.set_title(f"(c) Main-rank accuracy and computational cost",
                   fontsize=8, fontweight="bold", pad=4)

    ax_c.set_ylim(0, max(vpts) * 1.22 if vpts else 1.0)

    line_time_legend = Line2D(
        [0], [0],
        color="#202020",
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        marker="D",
        markersize=2.4,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label="Train time",
    )
    legend_handles = [
        Patch(facecolor="#BDBDBD", edgecolor="black", linewidth=0.4, label="VPT (bars)"),
        line_time_legend,
    ]
    ax_c.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=4.8,
        frameon=True,
        framealpha=0.9,
    )

    _apply_journal_axis_style(ax_c, with_grid=True)
    _apply_journal_axis_style(ax_c2, with_grid=False)
    ax_c.grid(True, alpha=0.30, linewidth=0.45, axis="y")

    ax_d = fig.add_subplot(gs[1, 1])
    key_horizons = [20, 200, 2000]
    line_styles = {20: "-", 200: "--", 2000: ":"}

    available_ranks = [r for r in RANKS if r in rank_data and "horizons" in rank_data[r]]
    if len(available_ranks) >= 2:
        for model in MODELS_ORDERED:
            for h in key_horizons:
                rs_avail, vals = [], []
                for r in available_ranks:
                    d = rank_data[r]
                    entry = d["horizons"].get(model, {}).get(str(h))
                    if entry and "r_squared" in entry:
                        rs_avail.append(r)
                        vals.append(entry["r_squared"])
                if rs_avail:
                    ax_d.plot(rs_avail, vals,
                              line_styles[h] + "o",
                              color=COLORS[model], markersize=4, linewidth=1.0,
                              markeredgewidth=0.4, markeredgecolor="white")

        handles_model = [Line2D([0], [0], color=COLORS[m], lw=1.5, label=m)
                         for m in MODELS_ORDERED]
        handles_h = [Line2D([0], [0], color="gray", ls=line_styles[h], lw=1.0,
                            label=f"h={h} ({h*dt:.0f} t.u.)")
                     for h in key_horizons]
        ax_d.legend(handles=handles_model + handles_h, loc="lower right",
                    fontsize=4.5, frameon=True, framealpha=0.85, ncol=2)
        ax_d.set_xlabel("POD rank $r$")
        ax_d.set_ylabel("$R^2$ (POD coefficients)")
        ax_d.set_xticks(available_ranks)
        ax_d.set_title("(d) Sensitivity to POD rank (r=4/10/31)",
                       fontsize=8, fontweight="bold", pad=4)
    else:
        for model in MODELS_ORDERED:
            hor_data = primary["horizons"].get(model, {})
            h_vals, r2_vals = [], []
            for h in key_horizons:
                entry = hor_data.get(str(h))
                if entry and "r_squared" in entry:
                    h_vals.append(h * dt)
                    r2_vals.append(entry["r_squared"])
            if h_vals:
                ax_d.plot(h_vals, r2_vals, "-" + MARKERS[model],
                          color=COLORS[model], markersize=4, linewidth=1.1,
                          label=model, alpha=0.85,
                          markeredgewidth=0.4, markeredgecolor="white")

        ax_d.set_xlabel("Prediction horizon (time units)")
        ax_d.set_ylabel("$R^2$ (POD coefficients)")
        ax_d.set_title(f"(d) $R^2$ at key horizons (r={r_val})",
                       fontsize=8, fontweight="bold", pad=4)
        ax_d.legend(loc="upper right", fontsize=5.2, frameon=True, framealpha=0.85)

    ax_d.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_d.set_ylim(-1.5, 1.1)
    _apply_journal_axis_style(ax_d, with_grid=True)

    ax_e = fig.add_subplot(gs[2, 0])
    eval_horizons_q5 = [20, 200, 2000]
    ls_q5 = {20: "-", 200: "--", 2000: ":"}

    available_fracs = sorted([f for f in TRAIN_FRACS if f in train_data])

    if available_fracs:
        for model in MODELS_ORDERED:
            for h in eval_horizons_q5:
                fracs_ok, r2s = [], []
                for frac in available_fracs:
                    d = train_data[frac]
                    entry = d["horizons"].get(model, {}).get(str(h))
                    if entry and "r_squared" in entry:
                        fracs_ok.append(frac * 100)
                        r2s.append(entry["r_squared"])
                if fracs_ok:
                    ax_e.plot(fracs_ok, r2s,
                              ls_q5[h] + MARKERS[model],
                              color=COLORS[model], markersize=4, linewidth=1.0,
                              markeredgewidth=0.4, markeredgecolor="white")

        handles_m = [Line2D([0], [0], color=COLORS[m], lw=1.5, label=m)
                     for m in MODELS_ORDERED]
        handles_hq5 = [Line2D([0], [0], color="gray", ls=ls_q5[h], lw=1.0,
                              label=f"h={h}")
                       for h in eval_horizons_q5]
        ax_e.legend(handles=handles_m + handles_hq5, loc="lower right",
                    fontsize=4.5, frameon=True, framealpha=0.85, ncol=2)
        ax_e.set_xlabel("Training data fraction (%)")
        ax_e.set_ylabel("$R^2$ (POD coefficients)")
        ax_e.set_title(f"(e) Sensitivity to training fraction (main r={r_val})",
                       fontsize=8, fontweight="bold", pad=4)
        ax_e.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax_e.set_ylim(-2.5, 1.1)
    else:
        for model in MODELS_ORDERED:
            ro = primary.get("rolling_origin", {}).get(model, {})
            h_vals, r2_vals, r2_stds = [], [], []
            if ro:
                for h_str in sorted(ro.keys(), key=lambda x: int(x)):
                    h = int(h_str)
                    entry = ro.get(h_str)
                    if entry is None:
                        continue
                    if "r_squared" in entry:
                        h_vals.append(h)
                        r2_vals.append(entry["r_squared"])
                        r2_stds.append(entry.get("r_squared_std", 0.0))
            else:
                hor = primary.get("horizons", {}).get(model, {})
                for h in eval_horizons_q5:
                    entry = hor.get(str(h))
                    if entry is not None and "r_squared" in entry:
                        h_vals.append(h)
                        r2_vals.append(entry["r_squared"])
                        r2_stds.append(0.0)
            if h_vals:
                ax_e.errorbar(
                    h_vals, r2_vals, yerr=r2_stds,
                    fmt="-" + MARKERS[model],
                    color=COLORS[model], markersize=4,
                    linewidth=1.0, elinewidth=0.7, capsize=2,
                    label=model,
                )

        ax_e.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax_e.set_xlabel("Prediction horizon h (steps)")
        ax_e.set_ylabel("Rolling-origin $R^2$ (mean ± std)")
        ax_e.set_title(f"(e) Rolling-origin stability (r={r_val})",
                       fontsize=8, fontweight="bold", pad=4)
        ax_e.set_ylim(-2.5, 1.1)
        ax_e.legend(loc="lower right", fontsize=5.0, frameon=True, framealpha=0.85)

    _apply_journal_axis_style(ax_e, with_grid=True)

    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")

    col_labels = ["Model", "h=20\n$R^2$", "h=200\n$R^2$", "h=2000\n$R^2$",
                  "VPT\n(t.u.)", "Train\n(s)"]
    table_data = []
    cell_colors = []
    for model in MODELS_ORDERED:
        row = [model]
        row_colors = [(*matplotlib.colors.to_rgba(COLORS[model])[:3], 0.15)]
        hor = primary["horizons"].get(model, {})
        for h_key in ["20", "200", "2000"]:
            val = hor.get(h_key, {}).get("r_squared", 0)
            row.append(f"{val:.3f}")
            if val > 0.5:
                bg = (0.85, 0.95, 0.85, 1.0)
            elif val < 0:
                bg = (1.0, 0.92, 0.92, 1.0)
            else:
                bg = (1.0, 1.0, 0.9, 1.0)
            row_colors.append(bg)
        vpt = primary["models"][model]["valid_prediction_time"]
        row.append(f"{vpt:.1f}")
        row_colors.append((0.95, 0.95, 0.95, 1.0))
        train_t = primary["models"][model]["train_time_s"]
        row.append(f"{train_t:.1f}")
        row_colors.append((0.95, 0.95, 0.95, 1.0))
        table_data.append(row)
        cell_colors.append(row_colors)

    table = ax_f.table(cellText=table_data, colLabels=col_labels,
                       cellColours=cell_colors,
                       colColours=[(0.90, 0.90, 0.90, 1.0)] * len(col_labels),
                       colWidths=[0.24, 0.13, 0.14, 0.15, 0.14, 0.12],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(1.12, 1.95)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        cell.set_linewidth(0.35)
        cell.PAD = 0.08
    ax_f.set_title(f"(f) Summary at r={r_val}",
                   fontsize=8, fontweight="bold", pad=12)

    out_fig = Path(out_path) if out_path is not None else (ROOT / "outputs" / "figure.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=600, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[Report] Saved report figure: {out_fig}")


def plot_rank_comparison() -> None:
    datasets = {r: _load_rank_result(r) for r in RANKS}
    available = [r for r in RANKS if datasets[r] is not None]
    if not available:
        print("No results found. Run main.py for r=4, r=10, r=31 first.")
        return

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.60, 9.0), facecolor="white")
    gs = fig.add_gridspec(3, 2, wspace=0.30, hspace=0.48,
                          left=0.09, right=0.97, top=0.95, bottom=0.06)

    horizons_all = [10, 20, 50, 100, 200, 500]
    dt = 0.2

    ax_a = fig.add_subplot(gs[0, 0])
    for r in available:
        d = datasets[r]
        r2s = [_get_r2(d, "Hankel-DMD", h) for h in horizons_all]
        valid = [(h * dt, v) for h, v in zip(horizons_all, r2s) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax_a.plot(xs, ys, "-o", color=RANK_COLORS[r], markersize=4,
                      linewidth=1.1, label=f"r={r}", markeredgewidth=0.5,
                      markeredgecolor="white")
    ax_a.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_a.axhline(0.9, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax_a.set_xlabel("Prediction horizon (time units)", fontsize=7)
    ax_a.set_ylabel("$R^2$", fontsize=7)
    ax_a.set_title("(a) Hankel-DMD: $R^2$ vs. horizon for different POD ranks",
                   fontsize=8, fontweight="bold", pad=4)
    ax_a.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.85)
    ax_a.tick_params(direction="in", length=2, width=0.5)
    ax_a.grid(True, alpha=0.2, linewidth=0.5)
    ax_a.set_ylim(-1.1, 1.1)

    ax_b = fig.add_subplot(gs[0, 1])
    for r in available:
        d = datasets[r]
        r2s = [_get_r2(d, "HybridRC", h) for h in horizons_all]
        valid = [(h * dt, v) for h, v in zip(horizons_all, r2s) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax_b.plot(xs, ys, "-s", color=RANK_COLORS[r], markersize=4,
                      linewidth=1.1, label=f"r={r}", markeredgewidth=0.5,
                      markeredgecolor="white")
    ax_b.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_b.set_xlabel("Prediction horizon (time units)", fontsize=7)
    ax_b.set_ylabel("$R^2$", fontsize=7)
    ax_b.set_title("(b) HybridRC: $R^2$ vs. horizon for different POD ranks",
                   fontsize=8, fontweight="bold", pad=4)
    ax_b.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.85)
    ax_b.tick_params(direction="in", length=2, width=0.5)
    ax_b.grid(True, alpha=0.2, linewidth=0.5)
    ax_b.set_ylim(-1.1, 1.1)

    ax_c = fig.add_subplot(gs[1, 0])
    key_horizons = [10, 50, 100]
    key_styles = ["-", "--", ":"]
    for model in MODELS_ORDERED:
        for h, ls in zip(key_horizons, key_styles):
            r2s = []
            rs_avail = []
            for r in available:
                v = _get_r2(datasets[r], model, h)
                if v is not None:
                    r2s.append(v)
                    rs_avail.append(r)
            if rs_avail:
                ax_c.plot(rs_avail, r2s, ls + "o", color=COLORS[model],
                          markersize=4, linewidth=1.0,
                          label=f"{model} h={h}" if ls == "-" else f"  h={h}",
                          markeredgewidth=0.5, markeredgecolor="white")
    ax_c.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_c.set_xlabel("POD rank $r$", fontsize=7)
    ax_c.set_ylabel("$R^2$", fontsize=7)
    ax_c.set_title("(c) $R^2$ vs. POD rank at key horizons",
                   fontsize=8, fontweight="bold", pad=4)
    ax_c.set_xticks(available)
    ax_c.legend(loc="upper right", fontsize=4.5, frameon=True, framealpha=0.85, ncol=2)
    ax_c.tick_params(direction="in", length=2, width=0.5)
    ax_c.grid(True, alpha=0.2, linewidth=0.5)

    ax_d = fig.add_subplot(gs[1, 1])
    for model in MODELS_ORDERED:
        for h, ls in zip(key_horizons, key_styles):
            corrs = []
            rs_avail = []
            for r in available:
                v = _get_corr(datasets[r], model, h)
                if v is not None:
                    corrs.append(v)
                    rs_avail.append(r)
            if rs_avail:
                ax_d.plot(rs_avail, corrs, ls + "o", color=COLORS[model],
                          markersize=4, linewidth=1.0,
                          label=f"{model} h={h}",
                          markeredgewidth=0.5, markeredgecolor="white")
    ax_d.axhline(0.9, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax_d.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax_d.set_xlabel("POD rank $r$", fontsize=7)
    ax_d.set_ylabel("Pearson correlation", fontsize=7)
    ax_d.set_title("(d) Correlation vs. POD rank at key horizons",
                   fontsize=8, fontweight="bold", pad=4)
    ax_d.set_xticks(available)
    ax_d.legend(loc="lower left", fontsize=4.5, frameon=True, framealpha=0.85, ncol=2)
    ax_d.tick_params(direction="in", length=2, width=0.5)
    ax_d.grid(True, alpha=0.2, linewidth=0.5)
    ax_d.set_ylim(-0.3, 1.05)

    ax_e = fig.add_subplot(gs[2, 0])
    x = np.arange(len(available))
    bar_w = 0.22
    for i_m, model in enumerate(MODELS_ORDERED):
        vpts = []
        for r in available:
            try:
                vpt = datasets[r]["models"][model]["valid_prediction_time"]
                vpts.append(float(vpt))
            except (KeyError, TypeError):
                vpts.append(0.0)
        offset = (i_m - 1) * bar_w
        bars = ax_e.bar(x + offset, vpts, bar_w, color=COLORS[model],
                        alpha=0.8, label=model)
        for bar, val in zip(bars, vpts):
            if val > 0:
                ax_e.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.2,
                          f"{val:.1f}", ha="center", va="bottom",
                          fontsize=4.5, fontweight="bold")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([f"r={r}" for r in available], fontsize=6)
    ax_e.set_ylabel("Valid prediction time (t.u.)", fontsize=7)
    ax_e.set_title("(e) Valid prediction time by model and rank",
                   fontsize=8, fontweight="bold", pad=4)
    ax_e.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.85)
    ax_e.tick_params(direction="in", length=2, width=0.5)
    ax_e.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    ax_f = fig.add_subplot(gs[2, 1])
    for r in available:
        d = datasets[r]
        ec = d["pod"]["energy_cum"]
        ax_f.plot(range(1, len(ec) + 1), [v * 100 for v in ec],
                  "-o", color=RANK_COLORS[r], markersize=3,
                  linewidth=1.0, label=f"r={r} ({ec[-1]*100:.1f}%)",
                  markeredgewidth=0.4, markeredgecolor="white")
    ax_f.axhline(95, color="gray", ls="--", lw=0.7, alpha=0.7)
    ax_f.axhline(85, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax_f.text(0.98, 95.5, "95%", fontsize=4.5, color="gray",
              ha="right", transform=ax_f.get_yaxis_transform())
    ax_f.text(0.98, 85.5, "85%", fontsize=4.5, color="gray",
              ha="right", transform=ax_f.get_yaxis_transform())
    ax_f.set_xlabel("Mode index $r$", fontsize=7)
    ax_f.set_ylabel("Cumulative energy (%)", fontsize=7)
    ax_f.set_title("(f) POD cumulative energy (all ranks)",
                   fontsize=8, fontweight="bold", pad=4)
    ax_f.legend(loc="lower right", fontsize=5.5, frameon=True, framealpha=0.85)
    ax_f.tick_params(direction="in", length=2, width=0.5)
    ax_f.grid(True, alpha=0.2, linewidth=0.5)

    out_fig = ROOT / "outputs" / "rank_comparison.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=600, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[Visualization] Saved rank comparison figure: {out_fig}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rank":
        plot_rank_comparison()
    else:
        plot_report_figure()
