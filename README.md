# DCFlex

Code and data for **“Watts vs. Bytes: Turning Data Centers into Grid
Assets via Storage–Compute Co-Optimization.”** The benchmark jointly schedules compute workloads, discrete DVFS operating points of servers, and a co-located battery while enforcing robust grid interconnection, and ancillary service constraints.

This repository contains the model, the exact processed workload input used in the paper, the complete multiday study drivers, and the figure-generation pipeline. Exploratory notebooks, cluster submission files, intermediate plots, and legacy model variants are intentionally excluded.

## Reproduce the paper

Requirements:

- Julia 1.11.7 (the version recorded in `Manifest.toml`)
- Python 3.12 or newer
- GNU Make
- Optional: a valid Gurobi 13 license for the paper's solver configuration

Run the full 31-day benchmark and regenerate all figures:

```bash
make
```

`make` instantiates the exact Julia environment from `Manifest.toml`, creates a
local Python virtual environment with pinned plotting packages, runs the full
study suite, regenerates the publication figures, and verifies expected row
counts and output files. The full workflow solves more than 8,000 robust
optimization instances and can take hours; independent days are parallelized
using all available Julia threads.

For a short installation and model check:

```bash
make smoke
```

Useful overrides:

```bash
make THREADS=8                 # control day-level parallelism
make N_DAYS=3                  # reduced multiday run for development
make SOLVER=gurobi             # require the paper's Gurobi 13 solver
make SOLVER=highs              # force the open-source fallback
```

The default `SOLVER=auto` uses Gurobi when a valid license is available and
otherwise falls back to HiGHS. Use `SOLVER=gurobi` when validating the exact
paper solver configuration. Solver choice and parallel MIP execution may cause
small numerical or timing differences without changing the study conclusions.

## Study map

| Paper result                                             | Driver                                        | Main output                                          |
| -------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| Representative dispatch                                  | `scripts/plot_tuned_representative_case.jl` | `results/representative/`                          |
| Interconnection, DVFS, workload, BESS, and OOS ablations | `scripts/ablation_study_multiday.jl`        | `results/ablation/csv/`                            |
| LP relaxation and minimum AS participation               | `scripts/ablation_lp_relaxation_and_as.jl`  | `results/ablation/csv/`                            |
| Commercial BESS sizing and cycling sensitivity           | `scripts/cluster_sizing_runner.jl`          | `results/battery_sizing/csv/`                      |
| Manuscript figures                                       | `scripts/plot_*.py`                         | `results/figures/` and `results/representative/` |

The multiday scripts use deterministic per-day seeds (`42 + 7 × day index`) and 50 joint uncertainty trajectories per day. The full default horizon is 24
hourly periods over the 31 daily slices in the included 744-hour trace.

## Repository layout

```text
src/                 Julia package and robust MILP formulation
scripts/             Case construction, study runners, plots, verification
data/workload/       Processed 31-day workload input
test/                 Fast model and data checks
results/              Generated CSVs, reports, and figures (git-ignored)
Project.toml          Direct Julia dependencies and compatibility bounds
Manifest.toml         Fully resolved Julia dependency lockfile
requirements.txt      Pinned Python plotting dependencies
Makefile              One-command end-to-end workflow
```

## Individual targets

```bash
make setup       # install Julia and Python dependencies
make studies     # run all optimization studies
make figures     # regenerate figures from study CSVs
make verify      # validate output coverage
make test        # run package tests
make framework   # regenerate only the conceptual framework figure
make clean       # remove generated environments, caches, and results
```

## Data and solver notes

The bundled workload file is the processed input used by the reported runs.
CAISO/PJM observations enter through the empirical statistics and hourly profile
parameters encoded by the case generator; see [`data/README.md`](data/README.md)
for provenance and the input checksum.

Gurobi is distributed as a Julia package dependency so version 13.0.0 is pinned
by the manifest, but its license is external and cannot be distributed in this
repository. HiGHS is fully open source and requires no separate license.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The code is
released under the [MIT License](LICENSE). The processed benchmark inputs retain
the terms of their original sources as described in `data/README.md`.
