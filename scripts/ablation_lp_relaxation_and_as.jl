"""
ablation_lp_relaxation_and_as.jl

Two additional ablation studies for the DC + BESS co-optimization (VCC formulation).

──────────────────────────────────────────────────────────────────────────────
Study A — Fully-LP Relaxation Ablation
──────────────────────────────────────────────────────────────────────────────
The exact model is an MILP because of two sources of integrality:

  • DVFS   : the bilinear power term p_{j,t} = a_t·w_{j,t} is handled by discrete
             frequency levels with selection binaries z_{l,t}.
  • Battery: the charge/discharge mode indicators α_t, β_t are binary with the
             complementarity α_t + β_t ≤ 1 (no simultaneous charge & discharge).

We relax these two pieces *independently* and *jointly* by continuous variables
and quantify the cost of each relaxation.  Two distinct routes make the DVFS
part continuous:

  • McCormick reformulation (`dvfs_mode=:mccormick`): replace the discrete
    formulation entirely by the McCormick envelope of p = a·w over [a_min,a_max].
  • Direct LP relaxation (`relax_dvfs_binary=true`): keep the exact discrete
    formulation and simply relax the level-selection binaries z_{l,t} to [0,1]
    (the textbook LP relaxation of the MILP itself).

  Config           DVFS                        BESS modes        Class
  ───────────────  ──────────────────────────  ────────────────  ─────
  MILP (exact)     discrete levels (binary z)  binary α,β        MILP   ← reference
  LP-DVFS          McCormick (cont.)           binary α,β        MILP
  LPR-DVFS         discrete, relaxed z∈[0,1]   binary α,β        MILP
  LP-BESS          discrete levels (binary z)  continuous α,β    MILP
  Full-LP          McCormick (cont.)           continuous α,β    LP
  Full-LPR         discrete, relaxed z∈[0,1]   continuous α,β    LP     ← direct LP relaxation of MILP

Metrics: objective, relaxation gap vs MILP, solve time / speed-up, McCormick
bilinear residual max|p−a·w|, mode-indicator fractionality, simultaneous
charge & discharge (energy "burn") and simultaneous AS commitment on both sides
(reserve/FRP stacking).  A negative-price probe confirms the textbook result
that the battery complementarity relaxation is *exact unless the energy price is
negative* (the only case where wasting energy via simultaneous charge+discharge
is profitable).

──────────────────────────────────────────────────────────────────────────────
Study B — Minimum Ancillary-Service Participation (Must-Offer Obligation)
──────────────────────────────────────────────────────────────────────────────
Real ancillary-service / resource-adequacy programs impose a *must-offer*
obligation: an enrolled resource has to offer a contracted minimum capacity into
the AS market during a commitment window.  We add a per-period floor on the total
AS offer (reserve + FRP-up + FRP-down) over the high-value evening window and
sweep it across realistic values for the 12 MW co-located BESS, measuring the
cost of the obligation, the AS-revenue response, battery cycling, job completion
and feasibility.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
    julia --project=. --threads=auto scripts/ablation_lp_relaxation_and_as.jl [study] [n_days]

    study ∈ {A, B, all}   (default: all)
    n_days                 (default: 31)

Outputs CSVs and a text report under `results/ablation/`.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuMP, Printf, Statistics, Random, Dates, DataFrames, CSV
import MathOptInterface as MOI

# Reuse the base-case generator and model rebuilding helpers.
include(joinpath(@__DIR__, "ablation_helpers.jl"))

# ============================================================================
# Configuration
# ============================================================================
const STUDY_ARG = length(ARGS) >= 1 ? ARGS[1] : "all"
const N_DAYS    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 31
const CSV_DIR   = joinpath(@__DIR__, "..", "results", "ablation", "csv")
const REPORT    = joinpath(@__DIR__, "..", "results", "ablation", "lp_relaxation_and_as_results.txt")
const TOL       = 1e-4               # numerical tolerance for "active" power
const TIME_LIMIT = 180.0             # solver time limit per solve [s]
const MIP_GAP    = 1e-4              # tight gap so relaxation comparisons are clean
mkpath(CSV_DIR)

# Evening peak AS commitment window: hours 14–21 (1-based periods 15–22).
const AS_WINDOW = collect(15:22)

# ============================================================================
# Helpers
# ============================================================================

# The base case requires the real DC load trace; if it is missing the generator
# silently falls back to a synthetic profile and results are NOT comparable to
# earlier runs. Fail loudly instead.
const REAL_LOAD_CSV = joinpath(@__DIR__, "..", "data", "workload", "hourly_aggregated_utilization.csv")
isfile(REAL_LOAD_CSV) || error(
    "Real load trace missing: $REAL_LOAD_CSV\n" *
    "Without it the base case silently degrades to synthetic load. Restore it with:\n" *
    "  git checkout -- data/workload/hourly_aggregated_utilization.csv")

"""Generate base VCC data for a specific day (deterministic per-day seed)."""
function make_day_data(day_idx::Int)
    seed = 42 + day_idx * 7
    generate_real_market_case_vcc(seed=seed, T=24, use_real_load=true,
                                  day_index=day_idx, quiet=true)
end

"""Save a DataFrame to CSV under CSV_DIR."""
function save_csv(df::DataFrame, name::String)
    path = joinpath(CSV_DIR, "$(name).csv")
    open(path, "w") do io
        println(io, join(names(df), ","))
        for row in eachrow(df)
            vals = [isa(row[c], AbstractString) ? "\"$(row[c])\"" : string(row[c]) for c in names(df)]
            println(io, join(vals, ","))
        end
    end
    println("  Saved $(nrow(df)) rows → $path")
    return path
end

"""Load a previously saved CSV from CSV_DIR (returns `nothing` if absent)."""
function load_saved_csv(name::String)
    path = joinpath(CSV_DIR, "$(name).csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    println("  Loaded $(nrow(df)) rows from previous run → $path")
    return df
end

"""
Build, configure and solve one VCC instance with the requested relaxation /
ancillary-service options.  Returns (sol, model, elapsed, status).
"""
function solve_config(data::ProblemDataVCC;
                      dvfs_mode::Symbol=:discrete,
                      relax_battery_binary::Bool=false,
                      relax_dvfs_binary::Bool=false,
                      enable_battery::Bool=true,
                      min_as_reserve::Float64=0.0,
                      min_as_frp_up::Float64=0.0,
                      min_as_frp_down::Float64=0.0,
                      min_as_total::Float64=0.0,
                      as_window::Union{Nothing,AbstractVector{Int}}=nothing,
                      n_dvfs_levels::Int=7,
                      tag::String="")
    model = build_model_vcc(data;
                            enable_battery=enable_battery,
                            dvfs_mode=dvfs_mode,
                            n_dvfs_levels=n_dvfs_levels,
                            relax_battery_binary=relax_battery_binary,
                            relax_dvfs_binary=relax_dvfs_binary,
                            min_as_reserve=min_as_reserve,
                            min_as_frp_up=min_as_frp_up,
                            min_as_frp_down=min_as_frp_down,
                            min_as_total=min_as_total,
                            as_window=as_window)
    set_silent(model)
    set_time_limit_sec(model, TIME_LIMIT)
    set_attribute(model, MOI.RelativeGapTolerance(), MIP_GAP)
    t0 = time()
    optimize!(model)
    elapsed = time() - t0
    st = termination_status(model)
    sol = has_values(model) ? solution_summary_vcc(model) : nothing
    return sol, model, elapsed, st
end

"""Job completion fraction in [0,1] from a solution summary."""
job_completion(sol) = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)

"""Total ancillary-service revenue [USD/day] from a solution summary."""
function as_revenue(sol, data::ProblemDataVCC)
    sol[:enable_battery] || return 0.0
    Δt = data.delta_t
    return sum(
        data.pricing.lambda_reserve[i] * sol[:bess][i+1, :reserve] +
        data.pricing.lambda_frp_up[i]  * sol[:bess][i+1, :frp_up]  +
        data.pricing.lambda_frp_down[i]* sol[:bess][i+1, :frp_down]
        for i in 1:length(data.horizon)) * Δt
end

"""Number of charge-equivalent cycles over the day from a solution summary."""
function bess_cycles(sol, data::ProblemDataVCC)
    sol[:enable_battery] || return 0.0
    discharged = sum(sol[:bess][sol[:bess].time .> 0, :discharge])
    return discharged / data.bess.capacity
end

"""
Read battery-relaxation diagnostics directly from the solved model:
  • simultaneous *energy* charge & discharge  (true energy burn)
  • simultaneous *AS* commitment on both sides (reserve/FRP stacking)
  • mode-indicator fractionality (how far α,β are from {0,1})
  • coincidence of energy burn with negative energy price
"""
function battery_relaxation_metrics(model::Model, data::ProblemDataVCC)
    vars = model.ext[:dc_vars]
    Tset = collect(data.horizon)
    Δt = data.delta_t
    n_energy_sim = 0; energy_sim_mwh = 0.0
    n_as_sim = 0;     as_overlap_mw = 0.0
    max_alpha_frac = 0.0; max_beta_frac = 0.0
    n_sim_negprice = 0;  min_price_at_sim = Inf
    for (i, t) in enumerate(Tset)
        c  = JuMP.value(vars.P_charge[t])
        d  = JuMP.value(vars.P_discharge[t])
        r  = JuMP.value(vars.P_reserve[t])
        fu = JuMP.value(vars.P_frp_up[t])
        fd = JuMP.value(vars.P_frp_down[t])
        charge_side    = c + fu
        discharge_side = d + r + fd
        if c > TOL && d > TOL
            n_energy_sim += 1
            energy_sim_mwh += min(c, d) * Δt
            price = data.pricing.lambda_energy[i]
            min_price_at_sim = min(min_price_at_sim, price)
            price < 0 && (n_sim_negprice += 1)
        end
        if charge_side > TOL && discharge_side > TOL
            n_as_sim += 1
            as_overlap_mw += min(charge_side, discharge_side)
        end
        av = JuMP.value(vars.α[t]); bv = JuMP.value(vars.β[t])
        max_alpha_frac = max(max_alpha_frac, min(av, 1 - av))
        max_beta_frac  = max(max_beta_frac,  min(bv, 1 - bv))
    end
    return (n_energy_sim=n_energy_sim, energy_sim_mwh=energy_sim_mwh,
            n_as_sim=n_as_sim, as_overlap_mw=as_overlap_mw,
            max_alpha_frac=max_alpha_frac, max_beta_frac=max_beta_frac,
            n_sim_negprice=n_sim_negprice,
            min_price_at_sim=(isfinite(min_price_at_sim) ? min_price_at_sim : NaN))
end

"""Maximum bilinear residual max_{j,t} |p − a·w| (0 by construction for discrete)."""
function mccormick_gap(model::Model)
    mc = mccormick_feasibility_check(model)
    return mc[:max_gap]
end

"""
DVFS level-selection diagnostics for the discrete formulation (binary or relaxed z):
  • n_frac_periods : periods where the level selection is fractional (max_l z < 1−TOL)
  • max_z_frac     : worst fractionality, max_t (1 − max_l z[l,t])
  • mean_active_levels : mean number of levels with z[l,t] > TOL per period
Returns NaN/0 fields when the model has no z variables (McCormick mode).
"""
function dvfs_relaxation_metrics(model::Model, data::ProblemDataVCC)
    vars = model.ext[:dc_vars]
    if !haskey(pairs(vars), :z)
        return (n_frac_periods=0, max_z_frac=NaN, mean_active_levels=NaN)
    end
    z = vars.z
    Tset = collect(data.horizon)
    Lset = 1:length(model.ext[:dc_settings].dvfs_levels)
    n_frac = 0; max_frac = 0.0; total_active = 0
    for t in Tset
        zvals = [JuMP.value(z[l, t]) for l in Lset]
        zmax = maximum(zvals)
        frac = 1.0 - zmax
        frac > TOL && (n_frac += 1)
        max_frac = max(max_frac, frac)
        total_active += count(>(TOL), zvals)
    end
    return (n_frac_periods=n_frac, max_z_frac=max_frac,
            mean_active_levels=total_active / length(Tset))
end

"""Number of binary variables in the model (0 ⇒ the instance is a pure LP)."""
n_binary_vars(model::Model) = num_constraints(model, VariableRef, MOI.ZeroOne)

# ============================================================================
# Study A — Fully-LP Relaxation Ablation
# ============================================================================
function run_studyA(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY A: Fully-LP Relaxation Ablation — $n_days days × 6 configs")
    println("="^80)

    # (label, dvfs_mode, relax_dvfs_binary, relax_battery_binary)
    configs = [
        ("MILP",     :discrete,  false, false),  # exact reference
        ("LP_DVFS",  :mccormick, false, false),  # McCormick DVFS reformulation only
        ("LPR_DVFS", :discrete,  true,  false),  # direct relaxation of z binaries only
        ("LP_BESS",  :discrete,  false, true),   # relaxed battery binaries only
        ("Full_LP",  :mccormick, false, true),   # McCormick + relaxed battery → pure LP
        ("Full_LPR", :discrete,  true,  true),   # relaxed z + relaxed battery → pure LP
    ]                                            #   (direct LP relaxation of the MILP)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed  = Threads.Atomic{Int}(0)
    t_start    = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        rows = NamedTuple[]
        obj_milp = NaN
        time_milp = NaN
        for (label, dmode, relax_z, relax) in configs
            sol, model, elapsed, st = solve_config(base; dvfs_mode=dmode,
                                                   relax_dvfs_binary=relax_z,
                                                   relax_battery_binary=relax,
                                                   tag="d$(day_idx)_$label")
            if sol === nothing
                continue
            end
            if label == "MILP"
                obj_milp = sol[:objective_value]
                time_milp = elapsed
            end
            bm = battery_relaxation_metrics(model, base)
            dm = dvfs_relaxation_metrics(model, base)
            push!(rows, (day=day_idx+1, config=label,
                         dvfs_mode=String(dmode), relax_dvfs=relax_z, relax_bess=relax,
                         n_binaries=n_binary_vars(model),
                         objective=sol[:objective_value],
                         worst_revenue=sol[:worst_case_revenue],
                         job_completion=job_completion(sol) * 100.0,
                         mean_dvfs=mean(sol[:vcc].a_factor),
                         as_revenue=as_revenue(sol, base),
                         mccormick_max_gap=mccormick_gap(model),
                         n_frac_dvfs_periods=dm.n_frac_periods,
                         max_z_frac=dm.max_z_frac,
                         mean_active_levels=dm.mean_active_levels,
                         n_energy_sim=bm.n_energy_sim,
                         energy_sim_mwh=bm.energy_sim_mwh,
                         n_as_sim=bm.n_as_sim,
                         as_overlap_mw=bm.as_overlap_mw,
                         max_alpha_frac=bm.max_alpha_frac,
                         max_beta_frac=bm.max_beta_frac,
                         n_sim_negprice=bm.n_sim_negprice,
                         min_price_at_sim=bm.min_price_at_sim,
                         solve_time=elapsed,
                         status=String(Symbol(st))))
        end
        # Append relaxation gap vs MILP (objective_LP - objective_MILP ≥ 0 for max)
        for k in eachindex(rows)
            rows[k] = merge(rows[k], (gap_vs_milp = rows[k].objective - obj_milp,
                                      speedup = time_milp / rows[k].solve_time))
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study A] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "studyA_lp_relaxation")
    return df
end

# ============================================================================
# Study A — Negative-Price Probe
# ============================================================================
"""
Inject negative energy prices into the cheapest hours of each day and compare the
exact MILP against the Full-LP relaxation.  Confirms that simultaneous
charge+discharge ("energy burn") appears in the LP *only* under negative prices.
"""
function run_studyA_negprice(; n_days::Int=N_DAYS, n_neg::Int=4, neg_price::Float64=-25.0)
    println("\n" * "="^80)
    println("STUDY A (probe): Negative-Price Energy-Burn Test — $n_days days")
    println("  Forcing the $n_neg cheapest hours/day to $(neg_price) USD/MWh")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed  = Threads.Atomic{Int}(0)
    t_start    = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        # Build a negative-price variant: set the n_neg cheapest hours to neg_price.
        neg_energy = copy(base.pricing.lambda_energy)
        order = sortperm(neg_energy)
        for k in 1:min(n_neg, length(order))
            neg_energy[order[k]] = neg_price
        end
        neg_pricing = PricingData(neg_energy, base.pricing.lambda_reserve,
                                  base.pricing.lambda_frp_up, base.pricing.lambda_frp_down,
                                  base.pricing.lambda_weight)
        d_neg = rebuild_with_pricing(base, neg_pricing)

        rows = NamedTuple[]
        for (label, dmode, relax_z, relax) in [("MILP",     :discrete,  false, false),
                                               ("Full_LP",  :mccormick, false, true),
                                               ("Full_LPR", :discrete,  true,  true)]
            sol, model, elapsed, st = solve_config(d_neg; dvfs_mode=dmode,
                                                   relax_dvfs_binary=relax_z,
                                                   relax_battery_binary=relax,
                                                   tag="d$(day_idx)_neg_$label")
            sol === nothing && continue
            bm = battery_relaxation_metrics(model, d_neg)
            push!(rows, (day=day_idx+1, config=label,
                         objective=sol[:objective_value],
                         n_energy_sim=bm.n_energy_sim,
                         energy_sim_mwh=bm.energy_sim_mwh,
                         n_sim_negprice=bm.n_sim_negprice,
                         min_price_at_sim=bm.min_price_at_sim,
                         solve_time=elapsed,
                         status=String(Symbol(st))))
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Probe] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "studyA_negprice_probe")
    return df
end

# ============================================================================
# Study B — Minimum Ancillary-Service Participation
# ============================================================================
function run_studyB(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY B: Minimum AS Participation (Must-Offer) — $n_days days")
    println("  Window: hours 14–21 (periods $(first(AS_WINDOW))–$(last(AS_WINDOW)))")
    println("="^80)

    # Per-period minimum AS offer [MW] swept up to the 12 MW battery nameplate to
    # expose the feasibility cliff.  Two obligation types:
    #   • "total"   : floor on reserve + FRP-up + FRP-down  (flexible must-offer)
    #   • "reserve" : floor on spinning reserve only         (stricter, RA-style)
    as_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0]

    day_frames = Vector{DataFrame}(undef, n_days)
    completed  = Threads.Atomic{Int}(0)
    t_start    = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        rows = NamedTuple[]
        for product in ("total", "reserve")
            for lvl in as_levels
                win = lvl > 0.0 ? AS_WINDOW : nothing
                kw = product == "total" ?
                     (; min_as_total=lvl, as_window=win) :
                     (; min_as_reserve=lvl, as_window=win)
                sol, model, elapsed, st = solve_config(base; dvfs_mode=:discrete,
                                                       relax_battery_binary=false,
                                                       kw...,
                                                       tag="d$(day_idx)_$(product)=$lvl")
                feasible = sol !== nothing &&
                           (st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED)
                if feasible
                    push!(rows, (day=day_idx+1, product=product, min_as_mw=lvl, feasible=true,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 as_revenue=as_revenue(sol, base),
                                 job_completion=job_completion(sol) * 100.0,
                                 cycles=bess_cycles(sol, base),
                                 mean_dvfs=mean(sol[:vcc].a_factor),
                                 solve_time=elapsed,
                                 status=String(Symbol(st))))
                else
                    push!(rows, (day=day_idx+1, product=product, min_as_mw=lvl, feasible=false,
                                 objective=NaN, worst_revenue=NaN, as_revenue=NaN,
                                 job_completion=NaN, cycles=NaN, mean_dvfs=NaN,
                                 solve_time=elapsed, status=String(Symbol(st))))
                end
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study B] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "studyB_min_as_participation")
    return df
end

# ============================================================================
# Aggregation & reporting
# ============================================================================
msd(x) = (m = mean(skipmissing(x)); s = length(collect(skipmissing(x))) > 1 ? std(skipmissing(x)) : 0.0; (m, s))

function aggregate_and_report(dfA, dfAneg, dfB, io::IO)
    println(io, "="^90)
    println(io, "LP RELAXATION & MINIMUM-AS PARTICIPATION — AGGREGATE RESULTS")
    println(io, "Generated: $(Dates.now())   Solver: $(DCFlex.solver_name())   Days: $N_DAYS")
    println(io, "="^90)

    # ---- Study A aggregate ----
    if dfA !== nothing
        println(io, "\n" * "-"^90)
        println(io, "STUDY A — Fully-LP Relaxation (mean ± std across days)")
        println(io, "-"^90)
        @printf(io, "%-9s %14s %12s %10s %9s %11s %10s %9s\n",
                "config", "objective", "gap_vs_MILP", "solve_s", "speedup",
                "mc_gap", "AS_stack", "E_burn")
        for cfg in ["MILP", "LP_DVFS", "LPR_DVFS", "LP_BESS", "Full_LP", "Full_LPR"]
            sub = dfA[dfA.config .== cfg, :]
            isempty(sub) && continue
            om, os = msd(sub.objective)
            gm, gs = msd(sub.gap_vs_milp)
            tm, _  = msd(sub.solve_time)
            spm, _ = msd(sub.speedup)
            mcm, _ = msd(sub.mccormick_max_gap)
            asm, _ = msd(sub.as_overlap_mw)
            ebm, _ = msd(sub.energy_sim_mwh)
            @printf(io, "%-9s %9.1fk±%-4.1f %8.2fk±%-2.1f %8.2f %8.2fx %10.2e %8.2f %8.3f\n",
                    cfg, om/1000, os/1000, gm/1000, gs/1000, tm, spm, mcm, asm, ebm)
        end
        # Stacking / fractionality detail
        println(io, "\n  Battery relaxation tightness detail (mean across days):")
        for cfg in ["LP_BESS", "Full_LP", "Full_LPR"]
            sub = dfA[dfA.config .== cfg, :]
            isempty(sub) && continue
            n_as_periods = mean(sub.n_as_sim)
            n_burn       = mean(sub.n_energy_sim)
            afrac        = mean(sub.max_alpha_frac)
            bfrac        = mean(sub.max_beta_frac)
            n_negprice   = sum(sub.n_sim_negprice)
            days_with_stack = count(>(0), sub.n_as_sim)
            @printf(io, "    %-8s : AS-stacking periods/day=%.2f (days w/ stacking=%d/%d), energy-burn periods/day=%.2f,\n",
                    cfg, n_as_periods, days_with_stack, nrow(sub), n_burn)
            @printf(io, "               max α-frac=%.3f, max β-frac=%.3f, energy-burn under +price (count)=%d\n",
                    afrac, bfrac, Int(sum(sub.n_energy_sim) - n_negprice))
        end
        # DVFS z-relaxation fractionality detail
        if hasproperty(dfA, :n_frac_dvfs_periods)
            println(io, "\n  DVFS z-relaxation tightness detail (mean across days):")
            for cfg in ["LPR_DVFS", "Full_LPR"]
                sub = dfA[dfA.config .== cfg, :]
                isempty(sub) && continue
                nfp = mean(sub.n_frac_dvfs_periods)
                mzf = mean(sub.max_z_frac)
                mal = mean(sub.mean_active_levels)
                days_frac = count(>(0), sub.n_frac_dvfs_periods)
                @printf(io, "    %-8s : fractional-z periods/day=%.2f (days w/ fractional z=%d/%d), max z-frac=%.3f, active levels/period=%.2f\n",
                        cfg, nfp, days_frac, nrow(sub), mzf, mal)
            end
        end
        # Gap summary
        milp = dfA[dfA.config .== "MILP", :]
        om, _ = isempty(milp) ? (NaN, NaN) : msd(milp.objective)
        for (cfg, pretty) in [("Full_LP", "Full-LP (McCormick)"), ("Full_LPR", "Full-LPR (direct MILP relaxation)")]
            full = dfA[dfA.config .== cfg, :]
            isempty(full) && continue
            gm, gs = msd(full.gap_vs_milp)
            @printf(io, "\n  %s mean relaxation gap = \$%.2fk/day ± \$%.2fk  (%.3f%% of |MILP objective|)\n",
                    pretty, gm/1000, gs/1000, abs(om) > 0 ? 100*gm/abs(om) : NaN)
            spm, _ = msd(full.speedup)
            @printf(io, "  %s mean speed-up over MILP = %.2f×\n", pretty, spm)
        end
    end

    # ---- Study A negative-price probe ----
    if dfAneg !== nothing
        println(io, "\n" * "-"^90)
        println(io, "STUDY A (probe) — Negative-Price Energy-Burn Test (mean across days)")
        println(io, "-"^90)
        for cfg in ["MILP", "Full_LP", "Full_LPR"]
            sub = dfAneg[dfAneg.config .== cfg, :]
            isempty(sub) && continue
            burn_periods = mean(sub.n_energy_sim)
            burn_mwh     = mean(sub.energy_sim_mwh)
            days_burn    = count(>(0), sub.n_energy_sim)
            @printf(io, "  %-8s : energy-burn periods/day=%.2f, burn=%.3f MWh/day, days w/ burn=%d/%d\n",
                    cfg, burn_periods, burn_mwh, days_burn, nrow(sub))
        end
        milp = dfAneg[dfAneg.config .== "MILP", :]
        milp_by_day = Dict(r.day => r.objective for r in eachrow(milp))
        for cfg in ["Full_LP", "Full_LPR"]
            full = dfAneg[dfAneg.config .== cfg, :]
            isempty(full) && continue
            frac_neg = sum(full.n_sim_negprice) / max(1, sum(full.n_energy_sim))
            @printf(io, "  → Of all %s energy-burn periods, %.1f%% occur at a negative price.\n", cfg, 100*frac_neg)
            # Objective gap (LP - MILP) on the negative-price instances, paired by day.
            if !isempty(milp)
                gaps = [r.objective - get(milp_by_day, r.day, NaN) for r in eachrow(full)]
                gm, gs = msd(gaps)
                @printf(io, "  → %s vs MILP objective gap under negative prices = \$%.2fk/day ± \$%.2fk\n", cfg, gm/1000, gs/1000)
            end
        end
        println(io, "    (If energy burn were exploited, these gaps would be large; small gaps confirm tightness.)")
    end

    # ---- Study B aggregate ----
    if dfB !== nothing
        println(io, "\n" * "-"^90)
        println(io, "STUDY B — Minimum AS Participation (mean ± std across days)")
        println(io, "  Per-period must-offer floor over hours 14–21.  Battery nameplate = 12 MW.")
        println(io, "-"^90)
        for product in ("total", "reserve")
            prod_df = dfB[dfB.product .== product, :]
            isempty(prod_df) && continue
            label = product == "total" ? "TOTAL AS (reserve+FRP↑+FRP↓)" : "RESERVE only"
            println(io, "\n  Obligation product: $label")
            @printf(io, "  %8s %8s %14s %14s %12s %10s %9s\n",
                    "min[MW]", "feas%", "objective", "Δobj_vs_0", "AS_rev", "complete%", "cycles")
            base0 = prod_df[prod_df.min_as_mw .== 0.0, :]
            obj0_by_day = Dict(r.day => r.objective for r in eachrow(base0))
            for lvl in sort(unique(prod_df.min_as_mw))
                sub = prod_df[prod_df.min_as_mw .== lvl, :]
                feas = sub[sub.feasible, :]
                feas_pct = 100 * nrow(feas) / nrow(sub)
                if isempty(feas)
                    @printf(io, "  %7.1f %7.0f%% %14s %14s %12s %10s %9s\n",
                            lvl, feas_pct, "INFEAS", "—", "—", "—", "—")
                    continue
                end
                om, os = msd(feas.objective)
                dobj = [r.objective - get(obj0_by_day, r.day, NaN) for r in eachrow(feas)]
                dm, ds = msd(dobj)
                arm, _ = msd(feas.as_revenue)
                jcm, _ = msd(feas.job_completion)
                cym, _ = msd(feas.cycles)
                @printf(io, "  %7.1f %7.0f%% %9.1fk±%-4.1f %8.2fk±%-3.1f %10.2fk %9.1f %9.3f\n",
                        lvl, feas_pct, om/1000, os/1000, dm/1000, ds/1000, arm/1000, jcm, cym)
            end
        end
    end
    println(io, "\n" * "="^90)
end

# ============================================================================
# Main
# ============================================================================
function main()
    println("="^80)
    println("DC + BESS — LP Relaxation & Minimum-AS Participation Studies")
    println("="^80)
    println("  Study: $STUDY_ARG | Days: $N_DAYS | Threads: $(Threads.nthreads())")
    println("  Solver: $(DCFlex.solver_name())")
    println("  Started: $(Dates.now())")

    dfA = nothing; dfAneg = nothing; dfB = nothing
    if STUDY_ARG in ("A", "all")
        dfA = run_studyA()
        dfAneg = run_studyA_negprice()
    end
    if STUDY_ARG in ("B", "all")
        dfB = run_studyB()
    end

    # For studies not rerun, fall back to the CSVs from a previous run so the
    # aggregate report keeps all sections instead of silently dropping them.
    dfA    === nothing && (dfA    = load_saved_csv("studyA_lp_relaxation"))
    dfAneg === nothing && (dfAneg = load_saved_csv("studyA_negprice_probe"))
    dfB    === nothing && (dfB    = load_saved_csv("studyB_min_as_participation"))

    open(REPORT, "w") do io
        aggregate_and_report(dfA, dfAneg, dfB, io)
    end
    # Echo the report to stdout as well.
    println("\n" * read(REPORT, String))
    println("Report written to: $REPORT")
    println("Finished: $(Dates.now())")
end

main()
