"""
Generate publication-quality battery sensitivity figures for IEEE Transactions manuscript.

Figures produced:
  (a) fig_net_value_combined.pdf  — Net daily value vs battery size (2XL + M3, 7% & 10%)
  (b) fig_value_decomp_merged.pdf — Stacked-bar value decomposition (2XL on top, M3 below)
  (c) fig_optimal_size_cycling.pdf — Optimal battery size vs cycling constraint (single-col)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "battery_sizing" / "csv"
OUT_DIR = ROOT / "results" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── battery specs ──────────────────────────────────────────────────────────
SPECS = {
    "Megapack_2XL": dict(power=1.9, energy=3.9, capex=1.2e6, eta_d=np.sqrt(0.92),
                         label="Megapack 2XL (2h)", color="#1f77b4", marker="o"),
    "Megapack_3":   dict(power=1.25, energy=5.0, capex=1.0e6, eta_d=np.sqrt(0.91),
                         label="Megapack 3 (4h)", color="#d62728", marker="s"),
}
SOC_INIT = 0.6
PROJECT_YEARS = 20
CYC_COLORS = {0.5: "#1f77b4", 1.0: "#2ca02c", 1.5: "#ff7f0e", 2.0: "#d62728"}
CYC_LABELS = {0.5: r"$\bar{n}_{\mathrm{cyc}}=0.5$",
              1.0: r"$\bar{n}_{\mathrm{cyc}}=1.0$",
              1.5: r"$\bar{n}_{\mathrm{cyc}}=1.5$",
              2.0: r"$\bar{n}_{\mathrm{cyc}}=2.0$"}

def crf(r):
    return r * (1 + r)**PROJECT_YEARS / ((1 + r)**PROJECT_YEARS - 1)

# ── load data ──────────────────────────────────────────────────────────────
def load_cycling_data():
    frames = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("sizing_ncyc_") and f.endswith(".csv") and "_soc50" not in f:
            frames.append(pd.read_csv(os.path.join(DATA_DIR, f)))
    return pd.concat(frames, ignore_index=True)

def classify(name):
    s = str(name)
    if s == "No_Battery":
        return "No_Battery", 0
    n = int(s.rstrip("x").split("_")[-1].replace("x", ""))
    btype = "Megapack_2XL" if "2XL" in s else "Megapack_3"
    return btype, n

def decompose(df):
    """Compute per-config value decomposition with SOC depletion correction."""
    bl = df[df["config_name"] == "No_Battery"].set_index(["n_cyc_max", "day"])["objective_value"]
    rows = []
    for (cyc, cfg), grp in df[df["config_name"] != "No_Battery"].groupby(["n_cyc_max", "config_name"]):
        btype, n_units = classify(cfg)
        spec = SPECS[btype]
        cap = n_units * spec["energy"]
        eta_d = spec["eta_d"]
        va_list, as_list, nonas_list = [], [], []
        for _, r in grp.iterrows():
            key = (r["n_cyc_max"], r["day"])
            if key not in bl.index:
                continue
            v = r["objective_value"] - bl.loc[key]
            delta_soc = max(0.0, SOC_INIT - r["final_soc_frac"])
            depl = delta_soc * cap * eta_d * r["avg_energy_price"]
            va_list.append(v - depl)
            as_list.append(r["as_revenue"])
            nonas_list.append(v - r["as_revenue"] - depl)
        if not va_list:
            continue
        rows.append(dict(
            n_cyc_max=cyc, battery_type=btype, n_units=n_units,
            energy_mwh=cap,
            va_mean=np.mean(va_list), va_std=np.std(va_list, ddof=1) if len(va_list) > 1 else 0,
            as_mean=np.mean(as_list), as_std=np.std(as_list, ddof=1) if len(as_list) > 1 else 0,
            nonas_mean=np.mean(nonas_list), nonas_std=np.std(nonas_list, ddof=1) if len(nonas_list) > 1 else 0,
        ))
    return pd.DataFrame(rows)


# ── IEEE style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

COL_W = 3.5   # IEEE single-column width (inches)
DBL_W = 7.16  # IEEE double-column width (inches)

# ── FIGURE (a): Net value vs battery size ──────────────────────────────────
def fig_net_value(decomp):
    """2×2 grid: rows = 7% / 10%, cols = 2XL / M3."""
    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 3.8), sharex="col")

    for row_idx, (rate, rate_label) in enumerate([(0.07, "7%"), (0.10, "10%")]):
        cr = crf(rate)
        for col_idx, (btype, spec) in enumerate(
            [("Megapack_2XL", SPECS["Megapack_2XL"]),
             ("Megapack_3", SPECS["Megapack_3"])]):
            ax = axes[row_idx, col_idx]
            daily_cost_per_unit = spec["capex"] * cr / 365

            for cyc in [0.5, 1.0, 1.5, 2.0]:
                sub = decomp[(decomp["n_cyc_max"] == cyc) &
                             (decomp["battery_type"] == btype)].sort_values("n_units")
                if sub.empty:
                    continue
                x = sub["energy_mwh"].values
                net = sub["va_mean"].values - sub["n_units"].values * daily_cost_per_unit
                ax.plot(x, net, color=CYC_COLORS[cyc], label=CYC_LABELS[cyc],
                        marker=spec["marker"], markersize=3, linewidth=1.2)
                opt_idx = np.argmax(net)
                ax.plot(x[opt_idx], net[opt_idx], marker="*", color=CYC_COLORS[cyc],
                        markersize=8, markeredgewidth=0.5, markeredgecolor="k",
                        zorder=5)

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
            ax.set_ylabel("Net value (\\$/day)")
            if row_idx == 0:
                ax.set_title(f"{spec['label']}")
            if row_idx == 1:
                ax.set_xlabel("Battery energy (MWh)")
            ax.text(0.03, 0.95, f"$r={rate_label}$".replace("%", r"\%"),
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
            ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[0, 1].legend(loc="upper left", framealpha=0.9, ncol=1)
    fig.align_ylabels()
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    out = os.path.join(OUT_DIR, "fig_net_value_combined.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── FIGURE (b): Value decomposition stacked bar ───────────────────────────
def fig_decomp_stacked(decomp):
    """Two rows (2XL top, M3 bottom), each with stacked AS + Non-AS bars."""
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.0), sharex=False)

    for ax_idx, (btype, spec) in enumerate(
        [("Megapack_2XL", SPECS["Megapack_2XL"]),
         ("Megapack_3", SPECS["Megapack_3"])]):
        ax = axes[ax_idx]
        # Use n_cyc = 1.0 (the natural cycling, preferred scenario)
        sub = decomp[(decomp["n_cyc_max"] == 1.0) &
                     (decomp["battery_type"] == btype)].sort_values("n_units")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        width = 0.7
        nonas = sub["nonas_mean"].values
        as_rev = sub["as_mean"].values
        total = sub["va_mean"].values

        ax.bar(x, nonas, width, label="Non-AS (flexibility)", color="#2ca02c", alpha=0.8)
        ax.bar(x, as_rev, width, bottom=nonas, label="AS revenue", color="#1f77b4", alpha=0.8)
        ax.plot(x, total, "k-o", markersize=3, linewidth=1.2, label="Total value added")

        ax.set_ylabel("Value added (\\$/day)")
        ax.set_title(spec["label"], fontsize=9)
        tick_step = 2
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels([f"{int(n)}" for n in sub["n_units"].values[::tick_step]])
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    axes[1].set_xlabel("Number of battery units")
    fig.tight_layout(h_pad=0.8)
    out = os.path.join(OUT_DIR, "fig_value_decomp_merged.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── FIGURE (c): Optimal size vs cycling constraint ────────────────────────
def fig_optimal_vs_cycling(decomp):
    """Single-column figure: optimal energy (MWh) vs n_cyc_max, both discount rates."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W + 0.3, 2.6), sharey=True)

    for ax_idx, (rate, rate_label) in enumerate([(0.07, "7%"), (0.10, "10%")]):
        ax = axes[ax_idx]
        cr = crf(rate)
        for btype, spec in [("Megapack_2XL", SPECS["Megapack_2XL"]),
                            ("Megapack_3", SPECS["Megapack_3"])]:
            daily_cost_per_unit = spec["capex"] * cr / 365
            opts = []
            for cyc in [0.5, 1.0, 1.5, 2.0]:
                sub = decomp[(decomp["n_cyc_max"] == cyc) &
                             (decomp["battery_type"] == btype)].sort_values("n_units")
                if sub.empty:
                    continue
                net = sub["va_mean"].values - sub["n_units"].values * daily_cost_per_unit
                best_idx = np.argmax(net)
                opts.append((cyc, sub.iloc[best_idx]["energy_mwh"], net[best_idx]))
            if not opts:
                continue
            cycs, ens, _ = zip(*opts)
            ax.plot(cycs, ens, color=spec["color"], marker=spec["marker"],
                    markersize=5, linewidth=1.5, label=spec["label"])

        ax.set_title(f"$r = {rate_label}$".replace("%", r"\%"), fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_xlim(0.3, 2.2)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

    axes[0].set_ylabel("Optimal energy (MWh)")
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.supxlabel(r"Cycling limit $\bar{n}_{\mathrm{cyc}}$ (cy/day)", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.0)
    out = os.path.join(OUT_DIR, "fig_optimal_size_cycling.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── FIGURE (d): Combined net value + decomposition ────────────────────────
def fig_combined_value(decomp):
    """Top row: net value (4 panels: 2×rate × 2×tech), bottom: side-by-side bars.

    Bottom row shows 2XL and M3 bars side-by-side at n_cyc=1.0 for direct
    comparison, with total VA lines including ±1 std shading.
    """
    fig = plt.figure(figsize=(DBL_W, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.35)

    # ── top row: net value (same as fig_net_value but 1 row, 2 cols) ──
    # We use 2 cols for 2 discount rates; each panel has both techs
    for col_idx, (rate, rate_label) in enumerate([(0.07, "7"), (0.10, "10")]):
        ax = fig.add_subplot(gs[0, col_idx])
        cr = crf(rate)
        for btype, spec in [("Megapack_2XL", SPECS["Megapack_2XL"]),
                            ("Megapack_3", SPECS["Megapack_3"])]:
            daily_cost_per_unit = spec["capex"] * cr / 365
            for cyc in [0.5, 1.0, 1.5, 2.0]:
                sub = decomp[(decomp["n_cyc_max"] == cyc) &
                             (decomp["battery_type"] == btype)].sort_values("n_units")
                if sub.empty:
                    continue
                x = sub["energy_mwh"].values
                net = sub["va_mean"].values - sub["n_units"].values * daily_cost_per_unit
                ls = "-" if btype == "Megapack_2XL" else "--"
                ax.plot(x, net, color=CYC_COLORS[cyc], linestyle=ls,
                        marker=spec["marker"], markersize=2.5, linewidth=1.0)
                opt_idx = np.argmax(net)
                ax.plot(x[opt_idx], net[opt_idx], marker="*", color=CYC_COLORS[cyc],
                        markersize=7, markeredgewidth=0.4, markeredgecolor="k", zorder=5)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel("Net value (\\$/day)")
        ax.set_xlabel("Battery energy (MWh)")
        ax.set_title(f"$r = {rate_label}\\%$", fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        # store for legend
        if col_idx == 1:
            ax_right = ax

    # legend for top row: tech (line style) + cycling (color)
    from matplotlib.lines import Line2D
    tech_handles = [Line2D([0], [0], color="gray", linestyle="-", linewidth=1.2,
                           marker="o", markersize=3, label="2XL (2h)"),
                    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2,
                           marker="s", markersize=3, label="M3 (4h)")]
    cyc_handles = [Line2D([0], [0], color=CYC_COLORS[c], linewidth=1.2,
                          label=CYC_LABELS[c]) for c in [0.5, 1.0, 1.5, 2.0]]
    ax_right.legend(handles=tech_handles + cyc_handles, loc="lower right",
                    fontsize=6.5, framealpha=0.9, ncol=2)

    # ── bottom row: side-by-side stacked bars at n_cyc=1.0 ──
    for col_idx, (rate, rate_label) in enumerate([(0.07, "7"), (0.10, "10")]):
        ax = fig.add_subplot(gs[1, col_idx])

        sub_2xl = decomp[(decomp["n_cyc_max"] == 1.0) &
                         (decomp["battery_type"] == "Megapack_2XL")].sort_values("n_units")
        sub_m3  = decomp[(decomp["n_cyc_max"] == 1.0) &
                         (decomp["battery_type"] == "Megapack_3")].sort_values("n_units")

        n_units_list = sorted(set(sub_2xl["n_units"].tolist()) | set(sub_m3["n_units"].tolist()))
        x = np.arange(len(n_units_list))
        width = 0.35

        for offset, (sub, spec, btype_label) in [
            (-width/2, (sub_2xl, SPECS["Megapack_2XL"], "2XL")),
            (+width/2, (sub_m3, SPECS["Megapack_3"], "M3"))]:

            sub_indexed = sub.set_index("n_units")
            nonas = np.array([sub_indexed.loc[n, "nonas_mean"] if n in sub_indexed.index else 0 for n in n_units_list])
            as_rev = np.array([sub_indexed.loc[n, "as_mean"] if n in sub_indexed.index else 0 for n in n_units_list])
            va = np.array([sub_indexed.loc[n, "va_mean"] if n in sub_indexed.index else 0 for n in n_units_list])
            va_s = np.array([sub_indexed.loc[n, "va_std"] if n in sub_indexed.index else 0 for n in n_units_list])

            c_as = "#1f77b4" if btype_label == "2XL" else "#aec7e8"
            c_na = "#2ca02c" if btype_label == "2XL" else "#98df8a"
            ax.bar(x + offset, nonas, width, label=f"Non-AS ({btype_label})", color=c_na, alpha=0.85)
            ax.bar(x + offset, as_rev, width, bottom=nonas, label=f"AS ({btype_label})", color=c_as, alpha=0.85)
            # ±1σ shaded band + mean line (matches reference style)
            ax.fill_between(x + offset, va - va_s, va + va_s,
                            alpha=0.20, color=spec["color"], linewidth=0, zorder=3)
            ax.plot(x + offset, va, color=spec["color"], marker=spec["marker"],
                    markersize=3, linewidth=1.0, label=f"VA$\\pm\\sigma$ ({btype_label})", zorder=4)

        tick_step = 2
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels([str(n) for n in n_units_list[::tick_step]])
        ax.set_xlabel("Number of battery units")
        ax.set_ylabel("Value added (\\$/day)")
        ax.set_title(f"Decomposition at $\\bar{{n}}_{{\\mathrm{{cyc}}}}=1.0$, $r={rate_label}\\%$",
                     fontsize=8)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        if col_idx == 0:
            ax.legend(loc="upper left", fontsize=5.5, framealpha=0.9, ncol=2)

    out = os.path.join(OUT_DIR, "fig_combined_value.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── FIGURE (e): Combined optimal size vs cycling (single panel) ───────────
def fig_optimal_cycling_combined(decomp):
    """Single panel with all 4 lines: 2 techs × 2 discount rates."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.6))

    # Colors: tech family (blue/red for 7%, teal/orange for 10%)
    RATE_STYLES = {0.07: ("-",  "7",  {"Megapack_2XL": "#1f77b4", "Megapack_3": "#d62728"}),
                   0.10: ("--", "10", {"Megapack_2XL": "#17becf", "Megapack_3": "#ff7f0e"})}

    for rate, (ls, rate_label, colors) in RATE_STYLES.items():
        cr = crf(rate)
        for btype, spec in [("Megapack_2XL", SPECS["Megapack_2XL"]),
                            ("Megapack_3", SPECS["Megapack_3"])]:
            daily_cost_per_unit = spec["capex"] * cr / 365
            opts = []
            for cyc in [0.5, 1.0, 1.5, 2.0]:
                sub = decomp[(decomp["n_cyc_max"] == cyc) &
                             (decomp["battery_type"] == btype)].sort_values("n_units")
                if sub.empty:
                    continue
                net = sub["va_mean"].values - sub["n_units"].values * daily_cost_per_unit
                best_idx = np.argmax(net)
                opts.append((cyc, sub.iloc[best_idx]["energy_mwh"]))
            if not opts:
                continue
            cycs, ens = zip(*opts)
            short = "2XL" if "2XL" in btype else "M3"
            ax.plot(cycs, ens, color=colors[btype], linestyle=ls,
                    marker=spec["marker"], markersize=5, linewidth=1.5,
                    label=f"{short} ($r={rate_label}\\%$)")

    ax.set_xlabel(r"Cycling limit $\bar{n}_{\mathrm{cyc}}$ (cy/day)")
    ax.set_ylabel("BESS energy capacity (MWh)", fontsize=9)
    ax.set_xlim(0.3, 2.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="center right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_optimal_cycling_combined.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_cycling_data()
    print(f"Loaded {len(df)} rows from cycling sweep CSVs")
    decomp = decompose(df)
    print(f"Computed {len(decomp)} config decompositions")

    fig_net_value(decomp)
    fig_decomp_stacked(decomp)
    fig_optimal_vs_cycling(decomp)
    fig_combined_value(decomp)
    fig_optimal_cycling_combined(decomp)
    print("All figures generated.")
