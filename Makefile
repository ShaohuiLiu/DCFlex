.DEFAULT_GOAL := all

JULIA ?= julia
PYTHON ?= python3
THREADS ?= auto
N_DAYS ?= 31
SOLVER ?= auto
JULIA_DEPOT ?= $(CURDIR)/.julia-depot:$(HOME)/.julia

JULIA_BASE := JULIA_DEPOT_PATH=$(JULIA_DEPOT) $(JULIA)
JULIA_RUN := JULIA_DEPOT_PATH=$(JULIA_DEPOT) DCFLEX_SOLVER=$(SOLVER) $(JULIA) --project=. --threads=$(THREADS)
VENV := .venv
PYTHON_RUN := XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.cache/matplotlib $(VENV)/bin/python
ABLATION_STAMP := results/ablation/.complete-$(N_DAYS)-$(SOLVER)
LP_AS_STAMP := results/ablation/.lp-as-complete-$(N_DAYS)-$(SOLVER)
REPRESENTATIVE_STAMP := results/representative/.complete-$(SOLVER)
BATTERY_STAMP := results/battery_sizing/.complete-$(N_DAYS)-$(SOLVER)
MODEL_SOURCES := $(wildcard src/*.jl) Project.toml Manifest.toml
CASE_SOURCES := scripts/real_market_data_case_vcc.jl
ABLATION_SOURCES := scripts/ablation_study_multiday.jl scripts/ablation_helpers.jl $(CASE_SOURCES) $(MODEL_SOURCES)
LP_AS_SOURCES := scripts/ablation_lp_relaxation_and_as.jl scripts/ablation_helpers.jl $(CASE_SOURCES) $(MODEL_SOURCES)
BATTERY_SOURCES := scripts/cluster_sizing_runner.jl scripts/battery_sizing_helpers.jl $(CASE_SOURCES) $(MODEL_SOURCES)
REPRESENTATIVE_SOURCES := scripts/plot_tuned_representative_case.jl scripts/dvfs_contrast_scan.jl scripts/representative_day_tuning.jl $(CASE_SOURCES) $(MODEL_SOURCES)

.PHONY: all setup studies figures verify smoke test framework clean help

all: verify

help:
	@echo "DCFlex reproducibility targets"
	@echo "  make            Run all 31-day paper studies and regenerate figures"
	@echo "  make smoke      Run tests and a representative two-solve case"
	@echo "  make figures    Regenerate figures from existing CSV results"
	@echo "  make test       Run the Julia test suite"
	@echo "  make clean      Remove generated results and local caches"
	@echo "Variables: N_DAYS=31 THREADS=auto SOLVER=auto|gurobi|highs"

setup: .julia-instantiated $(VENV)/.installed

.julia-instantiated: Project.toml Manifest.toml
	$(JULIA_BASE) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	@touch $@

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip --disable-pip-version-check install --requirement requirements.txt
	@touch $@

studies: $(REPRESENTATIVE_STAMP) $(ABLATION_STAMP) $(LP_AS_STAMP) $(BATTERY_STAMP)

$(REPRESENTATIVE_STAMP): .julia-instantiated $(REPRESENTATIVE_SOURCES)
	mkdir -p results/representative
	$(JULIA_RUN) scripts/plot_tuned_representative_case.jl
	@touch $@

$(ABLATION_STAMP): .julia-instantiated $(ABLATION_SOURCES)
	mkdir -p results/ablation/csv
	$(JULIA_RUN) scripts/ablation_study_multiday.jl all $(N_DAYS)
	@touch $@

$(LP_AS_STAMP): .julia-instantiated $(LP_AS_SOURCES)
	mkdir -p results/ablation/csv
	$(JULIA_RUN) scripts/ablation_lp_relaxation_and_as.jl all $(N_DAYS)
	@touch $@

$(BATTERY_STAMP): .julia-instantiated $(BATTERY_SOURCES)
	mkdir -p results/battery_sizing/csv
	$(JULIA_RUN) scripts/cluster_sizing_runner.jl 0.5 $(N_DAYS)
	$(JULIA_RUN) scripts/cluster_sizing_runner.jl 1.0 $(N_DAYS)
	$(JULIA_RUN) scripts/cluster_sizing_runner.jl 1.5 $(N_DAYS)
	$(JULIA_RUN) scripts/cluster_sizing_runner.jl 2.0 $(N_DAYS)
	@touch $@

figures: setup results/figures/.complete

results/figures/.complete: scripts/plot_representative.py scripts/plot_ablation_studies.py scripts/plot_battery_sizing.py scripts/plot_framework.py $(REPRESENTATIVE_STAMP) $(ABLATION_STAMP) $(BATTERY_STAMP)
	mkdir -p results/figures .cache/matplotlib
	$(PYTHON_RUN) scripts/plot_representative.py
	$(PYTHON_RUN) scripts/plot_ablation_studies.py
	$(PYTHON_RUN) scripts/plot_battery_sizing.py
	$(PYTHON_RUN) scripts/plot_framework.py
	@touch $@

framework: setup
	mkdir -p results/figures .cache/matplotlib
	$(PYTHON_RUN) scripts/plot_framework.py

verify: setup studies figures
	JULIA_DEPOT_PATH=$(JULIA_DEPOT) N_DAYS=$(N_DAYS) $(JULIA) --project=. scripts/verify_results.jl

test: .julia-instantiated
	JULIA_DEPOT_PATH=$(JULIA_DEPOT) DCFLEX_SOLVER=highs $(JULIA) --project=. -e 'using Pkg; Pkg.test()'

smoke: setup test framework
	mkdir -p results/representative
	JULIA_DEPOT_PATH=$(JULIA_DEPOT) DCFLEX_SOLVER=highs $(JULIA) --project=. --threads=$(THREADS) scripts/plot_tuned_representative_case.jl
	$(PYTHON_RUN) scripts/plot_representative.py

clean:
	rm -rf results .cache .julia-depot .julia-instantiated $(VENV)
	mkdir -p results
	touch results/.gitkeep
