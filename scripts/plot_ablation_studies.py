"""
Generate publication-quality multi-day ablation study figures (mean ± σ).

Reads per-day CSV results from results/ablation/csv/ and produces
IEEE Transactions style figures with error bars or shaded uncertainty bands
showing day-to-day variation across all 31 days.

Figures:
  fig_study1_phi_sensitivity.pdf       — φ vs objective + mean(a), mean ± σ band
  fig_study1b_headroom_cliff.pdf       — Headroom cliff, mean ± σ band
  fig_study2_battery_ablation.pdf      — Battery ablation, mean ± σ bars
  fig_study3_stress_cases.pdf          — Stress cases, mean ± σ bars
  fig_study4_ic_sensitivity.pdf        — IC sensitivity, mean ± σ band
  fig_study4b_lp_duals_temporal.pdf    — Per-hour dual mean profiles
  fig_study5_oos_robustness.pdf        — OOS robustness, mean ± σ
  fig_study6_dvfs_level_ablation.pdf   — DVFS level ablation, mean ± σ
  fig_study7_load_composition.pdf      — Load composition, mean ± σ band
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "results" / "ablation" / "csv"
OUT_DIR = ROOT / "results" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── IEEE Transactions style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   7.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
    "lines.linewidth":   1.5,
    "lines.markersize":  5,
    "axes.linewidth":    0.6,
    "grid.linewidth":    0.3,
    "grid.alpha":        0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.pad":   3,
    "ytick.major.pad":   3,
})

COL_W = 3.5    # IEEE single-column width (inches)
DBL_W = 7.16   # IEEE double-column width (inches)

C_BLUE   = "#1f77b4"
C_RED    = "#d62728"
C_GREEN  = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_PURPLE = "#9467bd"
C_BROWN  = "#8c564b"
C_GRAY   = "#7f7f7f"

# Legacy single-day LP-dual profile retained as an overlay for Study 4b.
# These values match the original base-day figure used in the manuscript and
# help preserve the directional ramp-up / ramp-down pattern after averaging.
LEGACY_HOURS = np.arange(1, 25)
LEGACY_RAMP_UP = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.27, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
])
LEGACY_RAMP_DN = np.array([
    0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.39, 0.18
])


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    print(f"  Saved {name}.pdf/.png")
    plt.close(fig)


def _load(name):
    path = os.path.join(CSV_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    n_days = df["day"].nunique()
    print(f"  Loaded {name}.csv: {len(df)} rows, {n_days} days")
    return df


# ============================================================================
# STUDY 1: φ Sensitivity — mean ± σ bands
# ============================================================================
def fig_study1():
    df = _load("study1_phi_sensitivity")
    if df is None:
        return

    grp = df.groupby("phi").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        a_mean=("mean_dvfs", "mean"), a_std=("mean_dvfs", "std"),
    ).reset_index()
    n_days = df["day"].nunique()

    phi = grp["phi"].values
    obj_m = grp["obj_mean"].values / 1e3
    obj_s = grp["obj_std"].values / 1e3
    a_m = grp["a_mean"].values
    a_s = grp["a_std"].values

    fig, ax1 = plt.subplots(figsize=(COL_W, 2.6))

    # Primary: objective band
    ln1, = ax1.plot(phi, obj_m, color=C_BLUE, marker="o", markersize=4, zorder=3,
                    label="Objective")
    ax1.fill_between(phi, obj_m - obj_s, obj_m + obj_s, color=C_BLUE, alpha=0.15, zorder=1)
    ax1.set_xlabel(r"DVFS sensitivity $\varphi$")
    ax1.set_ylabel(r"Objective (\$k/day)", color=C_BLUE)
    ax1.tick_params(axis="y", colors=C_BLUE)
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(True)

    # Mark base case
    ax1.axvline(0.25, color=C_GRAY, linestyle=":", linewidth=0.7, alpha=0.6)
    ax1.annotate("Base", xy=(0.25, obj_m[phi == 0.25][0] if 0.25 in phi else obj_m[3]),
                 xytext=(0.35, ax1.get_ylim()[0] + 1), fontsize=7, color=C_GRAY,
                 arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.5))

    # Three-regime shading
    yl = ax1.get_ylim()
    ax1.axvspan(0, 0.10, alpha=0.06, color=C_BLUE, zorder=0)
    ax1.axvspan(0.10, 0.55, alpha=0.06, color=C_GREEN, zorder=0)
    ax1.axvspan(0.55, 1.0, alpha=0.06, color=C_ORANGE, zorder=0)
    ax1.text(0.04, yl[1] - 0.3, "I", fontsize=7, color=C_BLUE, fontweight="bold")
    ax1.text(0.30, yl[1] - 0.3, "II", fontsize=7, color=C_GREEN, fontweight="bold")
    ax1.text(0.72, yl[1] - 0.3, "III", fontsize=7, color=C_ORANGE, fontweight="bold")

    # Secondary: mean(a) band
    ax2 = ax1.twinx()
    ln2, = ax2.plot(phi, a_m, color=C_RED, marker="s", markersize=3.5,
                    linewidth=1.2, linestyle="--", zorder=2, label=r"mean($a$)")
    ax2.fill_between(phi, a_m - a_s, a_m + a_s, color=C_RED, alpha=0.10, zorder=0)
    ax2.set_ylabel(r"Mean DVFS factor $\bar{a}$", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)

    ax1.legend(handles=[ln1, ln2], loc="lower right", framealpha=0.9)
    ax1.set_title(f"$n = {n_days}$ days, shading = $\\mu \\pm \\sigma$", fontsize=8,
                  loc="right", style="italic", color=C_GRAY)

    _save(fig, "fig_study1_phi_sensitivity")


# ============================================================================
# STUDY 1b: Headroom Cliff — mean ± σ bands
# ============================================================================
def fig_study1b():
    df = _load("study1b_fixed_schedulable")
    if df is None:
        return

    n_days = df["day"].nunique()
    df_a = df[df["part"] == "A_base_load_scale"].copy()
    df_b = df[df["part"] == "B_job_scale"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.8))

    # ── (a) Base load scaling ──
    grp_a = df_a.groupby("param").agg(
        hr_mean=("headroom", "mean"),
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        jc_mean=("job_completion", "mean"), jc_std=("job_completion", "std"),
    ).reset_index().sort_values("param")

    hr = grp_a["hr_mean"].values
    obj_m = grp_a["obj_mean"].values / 1e3
    obj_s = grp_a["obj_std"].values / 1e3
    jc_m = grp_a["jc_mean"].values
    jc_s = grp_a["jc_std"].values

    ln1a, = ax1.plot(hr, obj_m, color=C_BLUE, marker="o", markersize=4.5, zorder=3,
                     label="Objective")
    ax1.fill_between(hr, obj_m - obj_s, obj_m + obj_s, color=C_BLUE, alpha=0.15, zorder=1)
    ax1.set_xlabel("Headroom below load cap (MW)")
    ax1.set_ylabel(r"Objective (\$k/day)", color=C_BLUE)
    ax1.tick_params(axis="y", colors=C_BLUE)
    ax1.invert_xaxis()
    ax1.grid(True)

    ax1b = ax1.twinx()
    ln1b, = ax1b.plot(hr, jc_m, color=C_RED, marker="D", markersize=3.5,
                      linewidth=1.2, linestyle="--", zorder=2, label="Job completion")
    ax1b.fill_between(hr, jc_m - jc_s, jc_m + jc_s, color=C_RED, alpha=0.10, zorder=0)
    ax1b.set_ylabel("Job completion (%)", color=C_RED)
    ax1b.tick_params(axis="y", colors=C_RED)

    ax1.legend(handles=[ln1a, ln1b], loc="lower left", framealpha=0.9)
    ax1.set_title("(a) Base-load (fixed load) scaling", fontsize=9)

    # ── (b) Job scaling ──
    grp_b = df_b.groupby("param").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        jc_mean=("job_completion", "mean"), jc_std=("job_completion", "std"),
    ).reset_index().sort_values("param")

    js = grp_b["param"].values
    obj_m2 = grp_b["obj_mean"].values / 1e3
    obj_s2 = grp_b["obj_std"].values / 1e3
    jc_m2 = grp_b["jc_mean"].values
    jc_s2 = grp_b["jc_std"].values

    ln2a, = ax2.plot(js, obj_m2, color=C_BLUE, marker="o", markersize=4.5, zorder=3,
                     label="Objective")
    ax2.fill_between(js, obj_m2 - obj_s2, obj_m2 + obj_s2, color=C_BLUE, alpha=0.15, zorder=1)
    ax2.set_xlabel("Job demand scale factor")
    ax2.set_ylabel(r"Objective (\$k/day)", color=C_BLUE)
    ax2.tick_params(axis="y", colors=C_BLUE)
    ax2.axvline(1.0, color=C_GRAY, linestyle=":", linewidth=0.7)
    ax2.grid(True)

    ax2b = ax2.twinx()
    ln2b, = ax2b.plot(js, jc_m2, color=C_RED, marker="D", markersize=3.5,
                      linewidth=1.2, linestyle="--", zorder=2, label="Job completion")
    ax2b.fill_between(js, jc_m2 - jc_s2, jc_m2 + jc_s2, color=C_RED, alpha=0.10, zorder=0)
    ax2b.set_ylabel("Job completion (%)", color=C_RED)
    ax2b.tick_params(axis="y", colors=C_RED)

    ax2.legend(handles=[ln2a, ln2b], loc="lower left", framealpha=0.9)
    ax2.set_title("(b) Job (schedulable) demand scaling", fontsize=9)

    # fig.suptitle(f"$n = {n_days}$ days, bands = $\\mu \\pm \\sigma$", fontsize=8,
    #              x=0.99, ha="right", style="italic", color=C_GRAY)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study1b_headroom_cliff")


# ============================================================================
# STUDY 2: Battery Ablation — mean ± σ error bars
# ============================================================================
def fig_study2():
    df = _load("study2_battery_ablation")
    if df is None:
        return

    n_days = df["day"].nunique()
    order = ["No_Battery", "Small_18_6", "Base_36_12", "Large_72_24", "XL_108_36"]
    labels = ["No Batt.", "Small\n18/6", "Base\n36/12", "Large\n72/24", "XL\n108/36"]

    grp = df.groupby("config").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        as_mean=("as_revenue", "mean"), as_std=("as_revenue", "std"),
    ).reindex(order).reset_index()

    obj_m = grp["obj_mean"].values / 1e3
    obj_s = grp["obj_std"].values / 1e3
    as_m = grp["as_mean"].values / 1e3
    energy_m = (grp["obj_mean"].values - grp["as_mean"].values) / 1e3

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    x = np.arange(len(order))
    bar_w = 0.55

    ax.bar(x, energy_m, bar_w, color=C_BLUE, alpha=0.85, label="Energy + scheduling", zorder=2)
    ax.bar(x, as_m, bar_w, bottom=energy_m, color=C_ORANGE, alpha=0.85, label="AS revenue", zorder=2)
    ax.errorbar(x, obj_m, yerr=obj_s, fmt="none", ecolor="k", capsize=3, capthick=0.8,
                linewidth=0.8, zorder=4)

    ax.set_ylabel(r"Value (\$k/day)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="y")
    ax.legend(loc="lower right", framealpha=0.9)

    # Annotate incremental value vs no-battery
    base_obj_m = grp["obj_mean"].values[0] / 1e3
    for i in range(1, len(order)):
        delta = obj_m[i] - base_obj_m
        ax.annotate(f"+\\${delta:.1f}k", xy=(x[i], obj_m[i] + obj_s[i] + 0.3),
                    ha="center", fontsize=6.5, color=C_GREEN, fontweight="bold")

    ax.set_title(f"$n = {n_days}$ days", fontsize=8, loc="right", style="italic", color=C_GRAY)
    _save(fig, "fig_study2_battery_ablation")


# ============================================================================
# STUDY 3: Extreme Cases — mean ± σ horizontal bars
# ============================================================================
def fig_study3():
    df = _load("study3_extreme_cases")
    if df is None:
        return

    n_days = df["day"].nunique()
    # Drop NaN objectives (infeasible days) before aggregating
    df_valid = df.dropna(subset=["objective"])

    grp = df_valid.groupby("case_name").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        jc_mean=("job_completion", "mean"),
        a_mean=("mean_dvfs", "mean"),
        n_feasible=("objective", "count"),
    ).reset_index()

    # Sort by objective
    case_order = ["Base", "Zero_AS", "3x_Energy", "phi0_NoBatt",
                  "Tight_IC_ramp5", "phi1_TightIC",
                  "Tight_IC_load", "Heavy_Jobs_2x",
                  "Min_Headroom_5MW", "Tight_IC_both"]
    label_map = {
        "Base": "Base", "Tight_IC_load": "Tight IC\n(load)",
        "Tight_IC_ramp5": "Tight IC\n(ramp=5)", "Tight_IC_both": "Tight IC\n(both)",
        "Zero_AS": "Zero AS\nRev", "3x_Energy": r"3$\times$ Energy" "\nPrice",
        "Heavy_Jobs_2x": "Heavy\nJobs (2×)", "Min_Headroom_5MW": "Min Head-\nroom (5MW)",
        "phi0_NoBatt": "φ≈0,\nNo Batt.", "phi1_TightIC": "φ≈1,\nTight IC",
    }

    existing_cases = [c for c in case_order if c in grp["case_name"].values]
    grp = grp.set_index("case_name").loc[existing_cases].reset_index()

    fig, ax = plt.subplots(figsize=(COL_W, 3.2)) # single column
    y = np.arange(len(grp))
    obj_m = grp["obj_mean"].values / 1e3
    obj_s = grp["obj_std"].values / 1e3
    colors = [C_GREEN if o > 0 else C_RED for o in obj_m]

    ax.barh(y, obj_m, height=0.6, color=colors, alpha=0.75, xerr=obj_s,
            error_kw=dict(capsize=3, capthick=0.8, ecolor="k", lw=0.8),
            zorder=2, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(c, c) for c in grp["case_name"]], fontsize=7)
    ax.set_xlabel(r"Objective (\$k/day)")
    ax.set_xscale("symlog", linthresh=10)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(axis="x")
    ax.invert_yaxis()

    # Annotate job completion & feasibility — right-aligned at right axis border
    _trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, row in grp.iterrows():
        jc = row["jc_mean"]
        txt = f"Job: {jc:.0f}%"
        if jc < 100:
            txt += " (!)"
        feas = int(row["n_feasible"])
        if feas < n_days:
            txt += f" [{feas}/{n_days}d]"
        ax.text(0.98, i, txt, va="center", ha="right", fontsize=6.5,
                transform=_trans,
                color=C_RED if jc < 100 else C_GRAY)

    ax.set_title(f"$n = {n_days}$ days, bars = $\\mu \\pm \\sigma$", fontsize=8,
                 loc="right", style="italic", color=C_GRAY)
    _save(fig, "fig_study3_stress_cases")


# ============================================================================
# STUDY 4: IC Sensitivity — mean ± σ bands
# ============================================================================
def fig_study4():
    df = _load("study4_ic_sensitivity")
    if df is None:
        return

    n_days = df["day"].nunique()
    df_lc = df[df["sweep"] == "load_cap"]
    df_rc = df[df["sweep"] == "ramp_cap"]

    # Check if dual_sum column exists
    has_duals = "dual_sum" in df.columns and df["dual_sum"].notna().any()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.8))

    # ── (a) Load capacity ──
    agg_dict = {"objective": ["mean", "std"]}
    if has_duals:
        agg_dict["dual_sum"] = ["mean", "std"]
    grp_lc = df_lc.groupby("param").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        **({} if not has_duals else dict(dual_mean=("dual_sum", "mean"), dual_std=("dual_sum", "std"))),
    ).reset_index().sort_values("param")
    lc = grp_lc["param"].values
    obj_m = grp_lc["obj_mean"].values / 1e3
    obj_s = grp_lc["obj_std"].values / 1e3

    ln1, = ax1.plot(lc, obj_m, color=C_BLUE, marker="o", markersize=4.5, zorder=3,
                    label="MILP objective")
    ax1.fill_between(lc, obj_m - obj_s, obj_m + obj_s, color=C_BLUE, alpha=0.15, zorder=1)

    # FD marginal value + LP dual overlay
    if len(lc) >= 2:
        fd_mid = (lc[:-1] + lc[1:]) / 2
        fd_val = np.diff(grp_lc["obj_mean"].values) / np.diff(lc)
        fd_w = 0.82 * np.diff(lc)
        ax1b = ax1.twinx()
        ax1b.bar(fd_mid, fd_val, width=fd_w, alpha=0.25, color=C_RED, zorder=1)
        ax1b.set_ylabel(r"Daily marginal value (\$/MW-day)")
        ax1b.set_ylim(bottom=0)
        bars_proxy = plt.Rectangle((0, 0), 1, 1, fc=C_RED, alpha=0.25)

        handles = [ln1, bars_proxy]
        labels = ["MILP objective", r"FD $\lambda$ (\$/MW-day)"]

        # LP dual overlay
        if has_duals and "dual_mean" in grp_lc.columns:
            dual_m = grp_lc["dual_mean"].values
            valid = ~np.isnan(dual_m)
            if valid.any():
                ln_dual, = ax1b.plot(lc[valid], dual_m[valid], color=C_GREEN,
                                     marker="^", markersize=4, linewidth=1.0,
                                     linestyle="--", zorder=4)
                handles.append(ln_dual)
                labels.append(r"LP $\Sigma\lambda^{\mathrm{load}}$")

        ax1.legend(handles=handles, labels=labels,
                   loc="lower right", fontsize=7, framealpha=0.9)

    ax1.axvline(105, color=C_GRAY, linestyle=":", linewidth=0.7)
    ax1.set_xlabel(r"Load limit $\bar{D}$ (MW)")
    ax1.set_ylabel(r"Objective (\$k/day)")
    ax1.grid(True)
    ax1.set_title("(a) Peak-load capacity", fontsize=9)

    # ── (b) Ramp capacity ──
    grp_rc = df_rc.groupby("param").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        **({} if not has_duals else dict(dual_mean=("dual_sum", "mean"), dual_std=("dual_sum", "std"))),
    ).reset_index().sort_values("param")
    rc = grp_rc["param"].values
    obj_m2 = grp_rc["obj_mean"].values / 1e3
    obj_s2 = grp_rc["obj_std"].values / 1e3

    ln3, = ax2.plot(rc, obj_m2, color=C_BLUE, marker="o", markersize=4.5, zorder=3,
                    label="MILP objective")
    ax2.fill_between(rc, obj_m2 - obj_s2, obj_m2 + obj_s2, color=C_BLUE, alpha=0.15, zorder=1)

    if len(rc) >= 2:
        fd_mid2 = (rc[:-1] + rc[1:]) / 2
        fd_val2 = np.diff(grp_rc["obj_mean"].values) / np.diff(rc)
        fd_w2 = 0.82 * np.diff(rc)
        ax2b = ax2.twinx()
        ax2b.bar(fd_mid2, fd_val2, width=fd_w2, alpha=0.25, color=C_RED, zorder=1)
        ax2b.set_ylabel(r"Daily marginal value (\$/(MW/h)\,-day)")
        ax2b.set_ylim(bottom=0)
        bars_proxy2 = plt.Rectangle((0, 0), 1, 1, fc=C_RED, alpha=0.25)

        handles2 = [ln3, bars_proxy2]
        labels2 = ["MILP objective", r"FD $\lambda$ (\$/(MW/h)\,-day)"]

        # LP dual overlay
        if has_duals and "dual_mean" in grp_rc.columns:
            dual_m2 = grp_rc["dual_mean"].values
            valid2 = ~np.isnan(dual_m2)
            if valid2.any():
                ln_dual2, = ax2b.plot(rc[valid2], dual_m2[valid2], color=C_GREEN,
                                      marker="^", markersize=4, linewidth=1.0,
                                      linestyle="--", zorder=4)
                handles2.append(ln_dual2)
                labels2.append(r"LP $\Sigma(\lambda^{\uparrow}{+}\lambda^{\downarrow})$")

        ax2.legend(handles=handles2, labels=labels2,
                   loc="center right", fontsize=7, framealpha=0.9)

    ax2.axvline(15, color=C_GRAY, linestyle=":", linewidth=0.7)
    ax2.set_xlabel(r"Ramp limit $\bar{\Delta}$ (MW/h)")
    ax2.set_ylabel(r"Objective (\$k/day)")
    ax2.grid(True)
    ax2.set_title("(b) Ramp-rate capacity", fontsize=9)

    fig.suptitle(f"$n = {n_days}$ days, bands = $\\mu \\pm \\sigma$", fontsize=8,
                 x=0.99, ha="right", style="italic", color=C_GRAY)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study4_ic_sensitivity")


# ============================================================================
# STUDY 4b: Temporal LP Duals — mean bars
# ============================================================================
def fig_study4b():
    df = _load("study4b_lp_duals")
    if df is None:
        return

    n_days = df["day"].nunique()
    grp = df.groupby("hour").agg(
        load_mean=("lambda_load", "mean"),
        rup_mean=("lambda_ramp_up", "mean"),
        rdn_mean=("lambda_ramp_down", "mean"),
    ).reset_index()

    hours = grp["hour"].values
    bar_w = 0.7

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(COL_W, 3.4), sharex=True,
                                          gridspec_kw={"height_ratios": [1.6, 1]})

    # ── Top: load shadow price ──
    ax_top.bar(hours, grp["load_mean"], bar_w, color=C_BLUE, alpha=0.85, zorder=2)
    ax_top.set_ylabel(r"$\lambda^{\mathrm{load}}_t$ (\$/MW)")
    ax_top.grid(axis="y")
    ax_top.set_title("(a) Load-cap shadow price", fontsize=9, pad=4)

    # ── Bottom: ramp duals ──
    off = 0.18
    rup_mean = grp["rup_mean"].to_numpy()
    rdn_mean = grp["rdn_mean"].to_numpy()

    ax_bot.bar(hours - off, rup_mean, bar_w * 0.55,
               color=C_RED, alpha=0.85, zorder=2,
               label=r"$\lambda^{\mathrm{ramp\text{-}up}}_t$")
    ax_bot.bar(hours + off, rdn_mean, bar_w * 0.55,
               color=C_ORANGE, alpha=0.85, zorder=2,
               label=r"$\lambda^{\mathrm{ramp\text{-}dn}}_t$")
    ax_bot.set_ylabel(r"$\lambda^{\mathrm{ramp}}_t$ (\$/(MW/h))")
    ax_bot.set_xlabel("Hour of day")
    ax_bot.set_xlim(0.3, 24.7)
    ax_bot.set_xticks(np.arange(2, 25, 2))
    ax_bot.set_yscale("symlog", linthresh=0.1, linscale=0.4)
    ax_bot.grid(axis="y")
    ax_bot.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=1)
    ax_bot.set_title("(b) Ramp shadow prices", fontsize=9, pad=4)

    # fig.suptitle(f"$n = {n_days}$ days", fontsize=8, x=0.97, ha="right",
                #  style="italic", color=C_GRAY)
    fig.tight_layout(h_pad=1.0, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study4b_lp_duals_temporal")


# ============================================================================
# STUDY 5: OOS Robustness — mean ± σ grouped bars
# ============================================================================
def fig_study5():
    df = _load("study5_oos_robustness")
    if df is None:
        return

    n_days = df["day"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.4))

    # ── (a) Violation rates ──
    labels = ["Load", "Ramp"]
    in_means = [df["in_load_viol_rate"].mean(), df["in_ramp_viol_rate"].mean()]
    in_stds = [df["in_load_viol_rate"].std(), df["in_ramp_viol_rate"].std()]
    out_means = [df["oos_load_viol_rate"].mean(), df["oos_ramp_viol_rate"].mean()]
    out_stds = [df["oos_load_viol_rate"].std(), df["oos_ramp_viol_rate"].std()]

    x = np.arange(len(labels))
    w = 0.30
    ax1.bar(x - w/2, in_means, w, yerr=in_stds, color=C_BLUE, alpha=0.8,
            label="In-sample (50)", zorder=2,
            error_kw=dict(capsize=3, capthick=0.8, ecolor="k", lw=0.8))
    ax1.bar(x + w/2, out_means, w, yerr=out_stds, color=C_RED, alpha=0.8,
            label="Out-of-sample (500)", zorder=2,
            error_kw=dict(capsize=3, capthick=0.8, ecolor="k", lw=0.8))

    for i in range(len(labels)):
        ax1.plot([i - 0.4, i + 0.4], [5, 5], color="k", linewidth=1.2, linestyle="--", zorder=3)
    ax1.plot([], [], color="k", linewidth=1.2, linestyle="--", label="Risk tolerance (5%)")

    ax1.set_ylabel("Violation rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.grid(axis="y")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax1.set_title("(a) Period-level violation rates", fontsize=9)

    # ── (b) Excess magnitude ──
    cats = ["Load excess (MW)", "Ramp excess (MW/h)"]
    mean_vals = [df["oos_load_mean"].mean(), df["oos_ramp_mean"].mean()]
    mean_errs = [df["oos_load_mean"].std(), df["oos_ramp_mean"].std()]
    p95_vals = [df["oos_load_p95"].mean(), df["oos_ramp_p95"].mean()]
    p95_errs = [df["oos_load_p95"].std(), df["oos_ramp_p95"].std()]
    max_vals = [df["oos_load_max"].mean(), df["oos_ramp_max"].mean()]
    max_errs = [df["oos_load_max"].std(), df["oos_ramp_max"].std()]

    x2 = np.arange(len(cats))
    w2 = 0.22
    ekw = dict(capsize=3, capthick=0.8, ecolor="k", lw=0.8)
    ax2.bar(x2 - w2, mean_vals, w2, yerr=mean_errs, color=C_BLUE, alpha=0.8,
            label="Mean", zorder=2, error_kw=ekw)
    ax2.bar(x2, p95_vals, w2, yerr=p95_errs, color=C_ORANGE, alpha=0.8,
            label="95th pctile", zorder=2, error_kw=ekw)
    ax2.bar(x2 + w2, max_vals, w2, yerr=max_errs, color=C_RED, alpha=0.8,
            label="Maximum", zorder=2, error_kw=ekw)

    ax2.set_ylabel("Violation magnitude")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, fontsize=8)
    ax2.grid(axis="y")
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax2.set_title("(b) Violation magnitude (among violators)", fontsize=9)

    # fig.suptitle(f"$n = {n_days}$ days", fontsize=8, x=0.99, ha="right",
    #              style="italic", color=C_GRAY)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study5_oos_robustness")


# ============================================================================
# STUDY 6: DVFS Level Ablation — mean ± σ
# ============================================================================
def fig_study6():
    df = _load("study6_dvfs_level_ablation")
    if df is None:
        return

    n_days = df["day"].nunique()
    order = ["No_DVFS", "pm5_3lvl", "pm10_5lvl", "pm15_7lvl", "pm20_9lvl"]
    lbls = ["1\n(None)", "3\n(±5%)", "5\n(±10%)", "7\n(±15%)", "9\n(±20%)"]

    grp = df.groupby("config").agg(
        obj_mean=("objective", "mean"), obj_std=("objective", "std"),
        wrev_mean=("worst_revenue", "mean"),
        pen_mean=("dvfs_penalty", "mean"),
        a_mean=("mean_dvfs", "mean"), a_std=("mean_dvfs", "std"),
    ).reindex(order).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.8))

    # ── (a) Revenue-penalty decomposition — per-day Δ mean ± σ ──
    # Merge each config against the same day's No_DVFS baseline to get per-day deltas
    base_day = (df[df["config"] == "No_DVFS"]
                [["day", "worst_revenue", "dvfs_penalty", "objective"]]
                .rename(columns={"worst_revenue": "base_wrev",
                                 "dvfs_penalty":  "base_pen",
                                 "objective":     "base_obj"}))
    df_delta = df[df["config"] != "No_DVFS"].merge(base_day, on="day")
    df_delta["d_rev"] = df_delta["worst_revenue"] - df_delta["base_wrev"]
    df_delta["d_pen"] = df_delta["dvfs_penalty"]  - df_delta["base_pen"]
    df_delta["d_obj"] = df_delta["objective"]     - df_delta["base_obj"]

    grp_d = (df_delta.groupby("config")
             .agg(d_rev_mean=("d_rev", "mean"), d_rev_std=("d_rev", "std"),
                  d_pen_mean=("d_pen", "mean"), d_pen_std=("d_pen", "std"),
                  d_obj_mean=("d_obj", "mean"), d_obj_std=("d_obj", "std"))
             .reindex(order[1:]).reset_index())

    x = np.arange(len(order))
    bar_w = 0.35
    ekw = dict(capsize=3, capthick=0.8, ecolor="k", lw=0.8)
    ax1.bar(x[1:] - bar_w/2, grp_d["d_rev_mean"] / 1e3, bar_w, color=C_GREEN, alpha=0.8,
            yerr=grp_d["d_rev_std"] / 1e3, error_kw=ekw,
            label=r"$\Delta$ Worst-case Rev", zorder=2)
    ax1.bar(x[1:] + bar_w/2, -grp_d["d_pen_mean"] / 1e3, bar_w, color=C_RED, alpha=0.8,
            yerr=grp_d["d_pen_std"] / 1e3, error_kw=ekw,
            label=r"$-\Delta$ DVFS Penalty", zorder=2)

    # ax1.errorbar(x[1:], grp_d["d_obj_mean"] / 1e3, yerr=grp_d["d_obj_std"] / 1e3,
    #              fmt="D-", color="k", markersize=4, linewidth=1.2,
    #              capsize=3, capthick=0.8, zorder=4, label=r"$\Delta$ Objective (net)")
    delta_obj = grp["obj_mean"].values - grp["obj_mean"].values[0]
    ax1.plot(x[1:], delta_obj[1:] / 1e3, color="k", marker="D", markersize=4,
             linewidth=1.2, zorder=4, label=r"$\Delta$ Objective (net)")
    ax1.axhline(0, color="k", linewidth=0.4)
    ax1.set_ylabel(r"Change vs No DVFS (\$k)")
    ax1.set_xticks(x[1:])
    ax1.set_xticklabels(lbls[1:], fontsize=7)
    ax1.grid(axis="y")
    ax1.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    ax1.set_title("(a) Revenue\u2013penalty decomposition", fontsize=9)

    # ── (b) Objective and mean(a) ──
    levels = [1, 3, 5, 7, 9]
    obj_m = grp["obj_mean"].values / 1e3
    obj_s = grp["obj_std"].values / 1e3
    a_m = grp["a_mean"].values
    a_s = grp["a_std"].values

    ln1, = ax2.plot(levels, obj_m, color=C_BLUE, marker="o", markersize=5, zorder=3,
                    label="Objective")
    ax2.fill_between(levels, obj_m - obj_s, obj_m + obj_s, color=C_BLUE, alpha=0.15, zorder=1)
    ax2.axhline(obj_m[0], color=C_GRAY, linestyle=":", linewidth=0.7)
    ax2.set_xlabel("Number of DVFS levels")
    ax2.set_ylabel(r"Objective (\$k/day)", color=C_BLUE)
    ax2.tick_params(axis="y", colors=C_BLUE)
    ax2.set_xticks(levels)
    ax2.set_xticklabels(lbls, fontsize=6.5)
    ax2.grid(True)

    ax2b = ax2.twinx()
    ln2, = ax2b.plot(levels, a_m, color=C_RED, marker="s", markersize=4,
                     linewidth=1.2, linestyle="--", zorder=2, label=r"mean($a$)")
    ax2b.fill_between(levels, a_m - a_s, a_m + a_s, color=C_RED, alpha=0.10, zorder=0)
    ax2b.set_ylabel(r"Mean DVFS factor $\bar{a}$", color=C_RED)
    ax2b.tick_params(axis="y", colors=C_RED)

    ax2.legend(handles=[ln1, ln2], loc="lower left", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Objective and DVFS utilization", fontsize=9)

    # fig.suptitle(f"$n = {n_days}$ days, bars/bands = $\\mu \\pm \\sigma$", fontsize=8,
    #              x=0.99, ha="right", style="italic", color=C_GRAY)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study6_dvfs_level_ablation")


# ============================================================================
# STUDY 7: Load Composition — mean ± σ bands
# ============================================================================
def fig_study7():
    df = _load("study7_load_composition")
    if df is None:
        return

    n_days = df["day"].nunique()

    # -- Layout: broken y-axis (top/bot) for objective, single axis for DVFS
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(DBL_W, 3.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1],
                  hspace=0.10, wspace=0.40)
    ax1_top = fig.add_subplot(gs[0, 0])          # Light / Base / Heavy
    ax1_bot = fig.add_subplot(gs[1, 0], sharex=ax1_top)  # Stressed
    ax2     = fig.add_subplot(gs[:, 1])           # DVFS (single pane)

    styles = [
        ("Light",    C_BLUE,   "o"),
        ("Base",     C_GREEN,  "s"),
        ("Heavy",    C_RED,    "D"),
        ("Stressed", C_ORANGE, "v"),
    ]

    for level, color, marker in styles:
        sub = df[df["total_level"] == level]
        if sub.empty:
            continue
        total_mw = sub["total_load"].mean()
        label = f"{level} ({total_mw:.0f} MW)"

        grp = sub.groupby("fixed_frac").agg(
            obj_mean=("objective", "mean"), obj_std=("objective", "std"),
            a_mean=("mean_dvfs", "mean"), a_std=("mean_dvfs", "std"),
        ).reset_index()

        ff = grp["fixed_frac"].values
        obj_m = grp["obj_mean"].values / 1e3
        obj_s = grp["obj_std"].values / 1e3
        a_m = grp["a_mean"].values
        a_s = grp["a_std"].values

        # (a) Objective — draw on BOTH broken axes (clipping hides out-of-range)
        for ax in (ax1_top, ax1_bot):
            ax.plot(ff, obj_m, color=color, marker=marker, markersize=4,
                    linewidth=1.3, label=label, zorder=3)
            ax.fill_between(ff, obj_m - obj_s, obj_m + obj_s,
                            color=color, alpha=0.10, zorder=1)
            # Mark best (highest mean objective)
            best_idx = np.argmax(obj_m)
            ax.plot(ff[best_idx], obj_m[best_idx],
                    marker="*", color=color, markersize=10,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=5)

        # (b) mean(a)
        ax2.plot(ff, a_m, color=color, marker=marker, markersize=4, linewidth=1.3,
                 label=label, zorder=3)
        ax2.fill_between(ff, a_m - a_s, a_m + a_s, color=color, alpha=0.10, zorder=1)

    # -- Broken y-axis limits & cosmetics --
    ax1_top.set_ylim(155, 195)
    ax1_bot.set_ylim(-15, 135)
    ax1_top.spines["bottom"].set_visible(False)
    ax1_bot.spines["top"].set_visible(False)
    ax1_top.tick_params(bottom=False, labelbottom=False)

    # Diagonal break marks
    d = 0.015
    bk = dict(color="k", clip_on=False, linewidth=0.8)
    for xpos in (-d, 1 - d):
        ax1_top.plot((xpos, xpos + 2*d), (-d, +d),
                     transform=ax1_top.transAxes, **bk)
        ax1_bot.plot((xpos, xpos + 2*d), (1-d, 1+d),
                     transform=ax1_bot.transAxes, **bk)

    ax1_bot.set_xlabel("Fixed load fraction (%)")
    ax1_bot.set_ylabel(r"Objective (\$k/day)")
    ax1_bot.yaxis.set_label_coords(-0.14, 1.3)   # centre between two panes
    ax1_top.set_xlim(55, 100)
    ax1_top.grid(True)
    ax1_bot.grid(True)
    ax1_top.legend(fontsize=7, loc="lower left", framealpha=0.9,
                   title="Total load level", title_fontsize=7)
    ax1_top.set_title("(a) Objective vs load composition", fontsize=9)

    # -- DVFS panel --
    ax2.axhline(1.0, color=C_GRAY, linestyle=":", linewidth=0.7)
    ax2.annotate(r"$a_{\mathrm{ref}}=1.0$", xy=(96, 1.0), fontsize=7,
                 color=C_GRAY, ha="right", va="bottom")
    ax2.set_xlabel("Fixed load fraction (%)")
    ax2.set_ylabel(r"Mean DVFS factor $\bar{a}$")
    ax2.set_xlim(55, 100)
    ax2.grid(True)

    # Shade over/under-clocking regions
    yl = ax2.get_ylim()
    ax2.axhspan(1.0, yl[1], alpha=0.04, color=C_GREEN, zorder=0)
    ax2.axhspan(yl[0], 1.0, alpha=0.04, color=C_RED, zorder=0)
    ax2.text(57, yl[1] - 0.02, "Overclocking", fontsize=6.5, color=C_GREEN, style="italic")
    ax2.text(57, yl[0] + 0.01, "Underclocking", fontsize=6.5, color=C_RED, style="italic")

    ax2.legend(fontsize=7, loc="lower right", framealpha=0.9,
               title="Total load level", title_fontsize=7)
    ax2.set_title(r"(b) DVFS response ($\bar{a}$) vs composition", fontsize=9)

    fig.suptitle(f"$n = {n_days}$ days, bands = $\\mu \\pm \\sigma$", fontsize=8,
                 x=0.99, y=1.01, ha="right", style="italic", color=C_GRAY)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.90)
    _save(fig, "fig_study7_load_composition")


# ============================================================================
# STUDY 4c: IC Sensitivity × Load Composition
# ============================================================================
def fig_study4c():
    # Study 4c data (ff=60/80/95)
    df4c = _load("study4c_ic_composition")
    # Study 4 original data (natural composition)
    df4  = _load("study4_ic_sensitivity")
    if df4c is None:
        return

    n_days = df4c["day"].nunique()

    from matplotlib.lines import Line2D

    styles = [
        ("Base (85%)",  None,  C_BLUE,   "o"),   # from Study 4 (natural composition)
        ("60%",   60.0,  C_GREEN,  "s"),
        ("80%",   80.0,  C_ORANGE, "D"),
        ("95%",   95.0,  C_RED,    "^"),
    ]

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(DBL_W, 3.2))

    for sweep_name, ax, sweep_key, xlabel, xline in [
        ("load_cap", ax1, "load_cap", r"Load limit $\bar{D}$ (MW)",        105),
        ("ramp_cap", ax2, "ramp_cap", r"Ramp limit $\bar{\Delta}$ (MW/h)", 15),
    ]:
        axb = ax.twinx()
        n_groups = len(styles)

        for i, (label, ff_val, color, marker) in enumerate(styles):
            if ff_val is None:
                if df4 is None:
                    continue
                sub = df4[(df4["sweep"] == sweep_key)]
            else:
                sub = df4c[(df4c["sweep"] == sweep_key) &
                           (df4c["fixed_frac"] == ff_val)]
            if sub.empty:
                continue

            has_duals = "dual_sum" in sub.columns and sub["dual_sum"].notna().any()

            grp = sub.groupby("param").agg(
                obj_mean=("objective", "mean"), obj_std=("objective", "std"),
                **({} if not has_duals else dict(
                    dual_mean=("dual_sum", "mean"),
                    dual_std=("dual_sum", "std"))),
            ).reset_index().sort_values("param")

            x = grp["param"].values
            obj_m = grp["obj_mean"].values / 1e3
            obj_s = grp["obj_std"].values / 1e3

            # MILP objective on primary axis
            ax.plot(x, obj_m, color=color, marker=marker, markersize=4,
                    linewidth=1.3, zorder=5)
            ax.fill_between(x, obj_m - obj_s, obj_m + obj_s,
                            color=color, alpha=0.10, zorder=1)

            # FD bars on secondary axis — stagger 4 groups side-by-side
            if len(x) >= 2:
                dx = np.diff(x)                      # interval widths (non-uniform)
                fd_mid = (x[:-1] + x[1:]) / 2        # bar centers
                fd_val = np.diff(grp["obj_mean"].values) / dx  # $/MW-day or $/((MW/h)-day)
                slot_w = 0.80 * dx / n_groups         # each group's bar width per interval
                offset  = (i - (n_groups - 1) / 2) * slot_w
                axb.bar(fd_mid + offset, np.maximum(fd_val, 0),
                        width=slot_w, color=color, alpha=0.30, zorder=2)

            # LP dual on secondary axis
            if has_duals and "dual_mean" in grp.columns:
                dual_m = grp["dual_mean"].values
                valid = ~np.isnan(dual_m)
                if valid.any():
                    axb.plot(x[valid], np.maximum(dual_m[valid], 0), color=color,
                             marker=marker, markersize=3, linewidth=0.9,
                             linestyle="--", alpha=0.75, zorder=6)

        ax.axvline(xline, color=C_GRAY, linestyle=":", linewidth=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Objective (\$k/day)")
        ax.grid(True)
        axb.set_ylim(bottom=0)
        if sweep_name == "load_cap":
            axb.set_ylabel(r"Daily marginal / LP $\Sigma\lambda$ (\$/MW-day)")
        else:
            axb.set_ylabel(r"Daily marginal / LP $\Sigma\lambda$ (\$/(MW/h)-day)")

        # Unified legend: group lines + FD bar proxy + LP dual proxy
        h_obj = [Line2D([], [], color=s[2], marker=s[3], markersize=4,
                        linewidth=1.3, label=s[0]) for s in styles]
        h_fd  = plt.Rectangle((0, 0), 1, 1, fc=C_GRAY, alpha=0.30,
                               label="FD marginal (bars)")
        h_lp  = Line2D([], [], color=C_GRAY, linestyle="--", linewidth=0.9,
                       label="LP dual (dashed)")
        handles = h_obj + [h_fd, h_lp]
        loc = "lower right" if sweep_name == "load_cap" else "center right"
        ax.legend(handles=handles, fontsize=6.5, loc=loc, framealpha=0.9,
                  title="Fixed fraction", title_fontsize=6.5)

    ax1.set_title("(a) Peak-load capacity", fontsize=9)
    ax2.set_title("(b) Ramp-rate capacity", fontsize=9)

    # fig.suptitle(f"$n = {n_days}$ days, bands = $\\mu \\pm \\sigma$", fontsize=8,
    #              x=0.99, y=1.01, ha="right", style="italic", color=C_GRAY)
    fig.tight_layout(w_pad=3.0, rect=[0, 0, 1, 0.96])
    _save(fig, "fig_study4c_ic_composition")


# ============================================================================
# STUDY 8: Stressed Grid — Battery Value at IC Binding Points
# ============================================================================
def fig_study8():
    """
    Two-panel figure comparing NoBatt vs BESS_36_12 (36 MWh / 12 MW) as the
    interconnection limits are swept around the identified binding points:
      (a) Load-cap sweep 75–115 MW, ramp fixed at 10 MW/h
      (b) Ramp-cap sweep 3–30 MW/h, load fixed at 95 MW

    Primary axis:  Mean ± σ MILP objective for each config (solid/dashed lines).
    Secondary axis: FD marginal value bars (side-by-side per config) and LP dual
                    sum as dotted lines, both on the secondary y-axis, so the
                    "price of grid relief" can be read directly for each config.
    A vertical dashed line marks the identified binding point in each panel.
    """
    df = _load("study8_stressed_bess_value")
    if df is None:
        return

    n_days = df["day"].nunique()
    has_duals = "dual_sum" in df.columns and df["dual_sum"].notna().any()

    BIND_LOAD = 95.0
    BIND_RAMP = 10.0

    C_NOBATT = C_RED
    C_BESS   = C_BLUE

    cfg_styles = {
        "NoBatt":     (C_NOBATT, "--", "s", "No Battery"),
        "BESS_36_12": (C_BESS,   "-",  "o", "BESS 36/12"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 3.0))

    panel_cfg = [
        (ax1, "load_cap", BIND_LOAD,
         r"Load limit $\bar{D}$ (MW)",
         r"FD / LP $\Sigma\lambda^{\mathrm{load}}$ (\$/MW-day)",
         "(a) Peak-load capacity sweep\n(ramp fixed at 10 MW/h)"),
        (ax2, "ramp_cap", BIND_RAMP,
         r"Ramp limit $\bar{\Delta}$ (MW/h)",
         r"FD / LP $\Sigma(\lambda^{\uparrow}+\lambda^{\downarrow})$ (\$/(MW/h)-day)",
         "(b) Ramp-rate capacity sweep\n(load fixed at 95 MW)"),
    ]

    for ax, sweep_key, bind_val, xlabel, fd_ylabel, title in panel_cfg:
        axb = ax.twinx()
        df_sw = df[df["sweep"] == sweep_key]

        obj_at_bind = {}   # track each config's mean obj at binding point

        for cfg, (color, ls, mk, lbl) in cfg_styles.items():
            sub = df_sw[df_sw["config"] == cfg]
            if sub.empty:
                continue

            agg_kwargs = dict(obj_mean=("objective", "mean"),
                              obj_std=("objective", "std"))
            if has_duals:
                agg_kwargs["dual_mean"] = ("dual_sum", "mean")
                agg_kwargs["dual_std"]  = ("dual_sum", "std")

            grp = (sub.groupby("param")
                      .agg(**agg_kwargs)
                      .reset_index()
                      .sort_values("param"))

            x      = grp["param"].values
            obj_m  = grp["obj_mean"].values / 1e3
            obj_s  = grp["obj_std"].values  / 1e3

            # ── Primary axis: mean ± σ objective ──
            ax.plot(x, obj_m, color=color, marker=mk, markersize=4,
                    linewidth=1.4, linestyle=ls, zorder=3, label=lbl)
            ax.fill_between(x, obj_m - obj_s, obj_m + obj_s,
                            color=color, alpha=0.12, zorder=1)

            # Record objective at binding point for annotation
            bind_rows = grp[np.isclose(grp["param"], bind_val)]
            if not bind_rows.empty:
                obj_at_bind[cfg] = bind_rows["obj_mean"].values[0] / 1e3

            # ── Secondary axis: FD bars (side-by-side) ──
            if len(x) >= 2:
                dx      = np.diff(x)
                fd_mid  = (x[:-1] + x[1:]) / 2
                fd_val  = np.diff(grp["obj_mean"].values) / dx   # $/cap-unit per day
                bar_w   = 0.38 * dx
                # NoBatt bars sit left of midpoint; BESS bars sit right
                offset  = 0.20 * dx if cfg == "BESS_36_12" else -0.20 * dx
                axb.bar(fd_mid + offset, np.maximum(fd_val, 0),
                        width=bar_w, color=color, alpha=0.28, zorder=2)

            # ── Secondary axis: LP dual dotted line ──
            if has_duals and "dual_mean" in grp.columns:
                dm    = grp["dual_mean"].values
                valid = ~np.isnan(dm)
                if valid.any():
                    axb.plot(x[valid], np.maximum(dm[valid], 0),
                             color=color, marker="^", markersize=3.5,
                             linewidth=0.9, linestyle=":", alpha=0.85, zorder=5)

        # ── Binding-point reference line ──
        ax.axvline(bind_val, color=C_GRAY, linestyle=":", linewidth=0.9)

        # ── Annotate battery value (ΔObj) at binding point ──
        if "NoBatt" in obj_at_bind and "BESS_36_12" in obj_at_bind:
            delta = obj_at_bind["BESS_36_12"] - obj_at_bind["NoBatt"]
            ylim  = ax.get_ylim()
            ax.annotate(
                f"BESS\nvalue\n+${delta:.1f}k",
                xy=(bind_val, obj_at_bind["BESS_36_12"]),
                xytext=(bind_val + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        obj_at_bind["BESS_36_12"] - 0.15 * (ylim[1] - ylim[0])),
                fontsize=6.5, color=C_GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.7),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Objective (\$k/day)")
        axb.set_ylabel(fd_ylabel, fontsize=8)
        axb.set_ylim(bottom=0)
        ax.grid(True)
        ax.set_title(title, fontsize=9)

    # ── Unified legend: config lines + FD bar proxy + LP dual proxy ──
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h_bess   = Line2D([], [], color=C_BESS,   linestyle="-",  marker="o",
                      markersize=4, linewidth=1.4, label="BESS 36/12")
    h_nobatt = Line2D([], [], color=C_NOBATT, linestyle="--", marker="s",
                      markersize=4, linewidth=1.4, label="No Battery")
    h_fd     = Patch(facecolor=C_GRAY, alpha=0.28, label="FD marginal (bars)")
    h_lp     = Line2D([], [], color=C_GRAY, linestyle=":", marker="^",
                      markersize=3.5, linewidth=0.9, label="LP dual (dotted)")
    h_bind   = Line2D([], [], color=C_GRAY, linestyle=":", linewidth=0.9,
                      label="Binding point")
    for ax in (ax1, ax2):
        ax.legend(handles=[h_bess, h_nobatt, h_fd, h_lp, h_bind],
                  fontsize=6.5, loc="lower right", framealpha=0.9)

    # fig.suptitle(
    #     f"Study 8: Battery Value under Stressed Grid (IC binding points)\n"
    #     f"$n = {n_days}$ days, bands = $\\mu \\pm \\sigma$",
    #     fontsize=8, x=0.98, ha="right", style="italic", color=C_GRAY,
    # )
    fig.tight_layout(w_pad=3.0, rect=[0, 0, 1, 0.90])
    _save(fig, "fig_study8_stressed_bess_value")


# ############################################################################
# MAIN
# ############################################################################
if __name__ == "__main__":
    import sys
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    _funcs = {
        "study1": fig_study1, "study1b": fig_study1b,
        "study2": fig_study2, "study3":  fig_study3,
        "study4": fig_study4, "study4b": fig_study4b,
        "study4c": fig_study4c,
        "study5": fig_study5, "study6":  fig_study6,
        "study7": fig_study7,
        "study8": fig_study8,
    }

    print("Generating multi-day ablation study figures …")
    print(f"  CSV directory: {CSV_DIR}")
    print(f"  Output directory: {OUT_DIR}")
    print()

    if target in _funcs:
        _funcs[target]()
    else:
        for fn in _funcs.values():
            fn()

    print()
    print(f"Done — all figures saved to {OUT_DIR}/")
