#!/usr/bin/env python3
"""Plot the representative dispatch from solver-produced CSV tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "representative"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_load(dispatch: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    h = dispatch["hour"]
    ax.axhline(dispatch["load_cap"].iloc[0], color="#d62728", ls="--", lw=1.4,
               label="Interconnection limit")
    ax.plot(h, dispatch["fixed_load_bess"], color="#4c78a8", ls=":", lw=1.4,
            label="Fixed load (with BESS)")
    ax.plot(h, dispatch["dc_load_bess"], color="#7a5195", ls="-.", marker="s",
            ms=3, lw=1.5, label="Internal DC load (with BESS)")
    ax.plot(h, dispatch["net_load_bess"], color="black", marker="o", ms=3,
            lw=1.8, label="Grid withdrawal (with BESS)")
    ax.plot(h, dispatch["net_load_no_bess"], color="#f28e2b", ls="--", marker="D",
            ms=3, lw=1.5, label="Grid withdrawal (no BESS)")
    ax.set(xlabel="Hour", ylabel="Power (MW)", xlim=(1, 24))
    ax.set_xticks(np.arange(2, 25, 2))
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.savefig(RESULT_DIR / "load.png")
    plt.close(fig)


def plot_battery(dispatch: pd.DataFrame, soc: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.08]})
    h_soc = soc["hour"].to_numpy()
    mean = 100 * soc["scenario_mean"].to_numpy()
    std = 100 * soc["scenario_std"].to_numpy()
    ax1.fill_between(h_soc, mean - std, mean + std, color="#2a9d8f", alpha=0.22,
                     label=r"Scenario mean $\pm\sigma$")
    ax1.plot(h_soc, mean, color="#2a9d8f", ls="--", lw=1.3, label="Scenario mean")
    ax1.plot(h_soc, 100 * soc["scheduled_soc"], color="#2e7d32", marker="o", ms=2.5,
             lw=1.7, label="Base schedule")
    ax1.axhline(100 * soc["soc_min"].iloc[0], color="#d62728", ls=":", lw=1)
    ax1.axhline(100 * soc["soc_max"].iloc[0], color="#d62728", ls=":", lw=1,
                label="SOC limits")
    ax1.set_ylabel("SOC (%)")
    ax1.grid(alpha=0.25, lw=0.4)
    ax1.legend(ncol=2, frameon=False, loc="best")

    h = dispatch["hour"].to_numpy()
    charge = dispatch["charge"].to_numpy()
    discharge = dispatch["discharge"].to_numpy()
    reserve = dispatch["reserve"].to_numpy()
    frp_up = dispatch["frp_up"].to_numpy()
    frp_down = dispatch["frp_down"].to_numpy()
    ax2.bar(h, charge, color="#4c78a8", label="Charge")
    ax2.bar(h, frp_up, bottom=charge, color="#9ecae9", label="Up FRP")
    ax2.bar(h, -discharge, color="#b22222", label="Discharge")
    ax2.bar(h, -reserve, bottom=-discharge, color="#f08080", label="Reserve")
    ax2.bar(h, -frp_down, bottom=-(discharge + reserve), color="#f7b6b2",
            label="Down FRP")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set(xlabel="Hour", ylabel="Power (MW)", xlim=(0.25, 24.75))
    ax2.set_xticks(np.arange(2, 25, 2))
    ax2.grid(axis="y", alpha=0.25, lw=0.4)
    ax2.legend(ncol=5, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "battery.png")
    plt.close(fig)


def plot_dvfs(dispatch: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.7))
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="Nominal")
    ax.plot(dispatch["hour"], dispatch["dvfs_no_bess"], color="#f28e2b", ls="--",
            marker="D", ms=3, label="No BESS")
    ax.plot(dispatch["hour"], dispatch["dvfs_bess"], color="#4c78a8", marker="o",
            ms=3, label="With BESS")
    ax.set(xlabel="Hour", ylabel="DVFS factor", xlim=(1, 24))
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False)
    fig.savefig(RESULT_DIR / "dvfs_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    dispatch = pd.read_csv(RESULT_DIR / "dispatch.csv")
    soc = pd.read_csv(RESULT_DIR / "soc.csv")
    plot_load(dispatch)
    plot_battery(dispatch, soc)
    plot_dvfs(dispatch)
    print(f"Saved representative figures to {RESULT_DIR}")
