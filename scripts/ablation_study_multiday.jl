"""
Multi-Day Ablation Study Runner for DC+BESS Co-optimization (VCC Formulation)

Runs all ablation studies across ALL 31 days of real load data, parallelizing
over days with Threads.@threads.  Results are saved as CSV files for
post-processing (mean ± std aggregation and plotting).

Usage:
    # Run all studies (≈ 3,100 MILP solves with 16 threads):
    julia --project=. --threads=auto scripts/ablation_study_multiday.jl

    # Run a single study:
    julia --project=. --threads=auto scripts/ablation_study_multiday.jl 1
    julia --project=. --threads=auto scripts/ablation_study_multiday.jl 1b

    # Run with specific number of days:
    julia --project=. --threads=auto scripts/ablation_study_multiday.jl all 7

The reusable single-day model transformations live in `ablation_helpers.jl`;
this driver wraps each study in a day loop.
Each study function returns a DataFrame with a `day` column; results are
written to results/ablation/csv/<study_name>.csv.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuMP, HiGHS, Printf, Statistics, Random, Dates, DataFrames
import MathOptInterface as MOI

# ── Reuse all helpers from the single-day ablation study ──
include(joinpath(@__DIR__, "ablation_helpers.jl"))

# ============================================================================
# Configuration
# ============================================================================
const N_DAYS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 31
const CSV_DIR = joinpath(@__DIR__, "..", "results", "ablation", "csv")
mkpath(CSV_DIR)

# ============================================================================
# Thread-safe day-level parallelism helpers
# ============================================================================

"""Generate base data for a specific day."""
function make_day_data(day_idx::Int)
    seed = 42 + day_idx * 7  # deterministic per-day seed
    generate_real_market_case_vcc(seed=seed, T=24, use_real_load=true,
                                  day_index=day_idx, quiet=true)
end

"""Save a DataFrame to CSV."""
function save_csv(df::DataFrame, name::String)
    path = joinpath(CSV_DIR, "$(name).csv")
    open(path, "w") do io
        # Header
        println(io, join(names(df), ","))
        # Rows
        for row in eachrow(df)
            vals = [isa(row[c], AbstractString) ? "\"$(row[c])\"" : string(row[c]) for c in names(df)]
            println(io, join(vals, ","))
        end
    end
    println("  Saved $(nrow(df)) rows → $path")
end

# ============================================================================
# Study 1 — φ Sensitivity (multi-day)
# ============================================================================
function run_study1_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 1: φ Sensitivity — $n_days days × 13 φ-values")
    println("="^80)

    phi_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 0.99]
    n_configs = length(phi_values)

    # Pre-allocate per-day storage
    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        rows = NamedTuple[]
        for phi in phi_values
            data_phi = rebuild_with_phi(base, phi)
            sol, _, _, elapsed = solve_case(data_phi; tag="d$(day_idx)_φ=$phi")
            if sol !== nothing
                mean_a = mean(sol[:vcc].a_factor)
                as_rev = sol[:enable_battery] ? sum(
                    base.pricing.lambda_reserve[i] * sol[:bess][i+1, :reserve] +
                    base.pricing.lambda_frp_up[i] * sol[:bess][i+1, :frp_up] +
                    base.pricing.lambda_frp_down[i] * sol[:bess][i+1, :frp_down]
                    for i in 1:length(base.horizon)) : 0.0
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                push!(rows, (day=day_idx+1, phi=phi,
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc * 100.0,
                             mean_dvfs=mean_a, as_revenue=as_rev,
                             solve_time=elapsed))
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 1] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study1_phi_sensitivity")
    return df
end

# ============================================================================
# Study 1b — Fixed vs Schedulable (multi-day)
# ============================================================================
function run_study1b_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 1b: Fixed vs Schedulable — $n_days days")
    println("="^80)

    bl_scales = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    job_scales = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        rows = NamedTuple[]

        # Part A: base load scaling
        for sc in bl_scales
            d = rebuild_with_base_load_scale(base, sc)
            hr = d.vcc.hardware_limit - maximum(d.base_load_upper)
            sol, _, _, elapsed = solve_case(d; tag="d$(day_idx)_bl=$sc")
            if sol !== nothing
                mean_a = mean(sol[:vcc].a_factor)
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                push!(rows, (day=day_idx+1, part="A_base_load_scale", param=sc,
                             headroom=hr, objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0, mean_dvfs=mean_a,
                             total_load_slack=sol[:slack_violations][:total_load_slack],
                             solve_time=elapsed))
            end
        end

        # Part B: job scaling
        for js in job_scales
            d = rebuild_with_job_scale(base, js, min(js, 2.0))
            sol, _, _, elapsed = solve_case(d; tag="d$(day_idx)_js=$js")
            if sol !== nothing
                mean_a = mean(sol[:vcc].a_factor)
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                hr = base.vcc.hardware_limit - maximum(base.base_load_upper)  # unchanged by job scale
                push!(rows, (day=day_idx+1, part="B_job_scale", param=js,
                             headroom=hr, objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0, mean_dvfs=mean_a,
                             total_load_slack=sol[:slack_violations][:total_load_slack],
                             solve_time=elapsed))
            end
        end

        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 1b] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study1b_fixed_schedulable")
    return df
end

# ============================================================================
# Study 2 — Battery Ablation (multi-day)
# ============================================================================
function run_study2_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 2: Battery Ablation — $n_days days × 5 configs")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        configs = [
            ("No_Battery",   false, BESSParameters(1.0, 0.5, 0.2, 0.9, 0.0, 0.0, 0.95, 0.95, 45.0, 0.5)),
            ("Small_18_6",   true,  BESSParameters(18.0, 0.6, 0.15, 0.90, 6.0, 6.0, 0.95, 0.95, 45.0, 0.5)),
            ("Base_36_12",   true,  base.bess),
            ("Large_72_24",  true,  BESSParameters(72.0, 0.6, 0.15, 0.90, 24.0, 24.0, 0.95, 0.95, 45.0, 0.5)),
            ("XL_108_36",    true,  BESSParameters(108.0, 0.6, 0.15, 0.90, 36.0, 36.0, 0.95, 0.95, 45.0, 0.5)),
        ]

        rows = NamedTuple[]
        for (name, enable, bess_cfg) in configs
            data_batt = rebuild_with_bess(base, bess_cfg)
            sol, model, _, elapsed = solve_case(data_batt; enable_battery=enable, tag="d$(day_idx)_$name")
            if sol !== nothing
                as_rev = 0.0; discharged = 0.0; cyc = 0.0
                if enable && sol[:enable_battery]
                    bess_ops = sol[:bess][sol[:bess].time .> 0, :]
                    discharged = sum(bess_ops.discharge)
                    cyc = discharged / bess_cfg.capacity
                    as_rev = sum(
                        base.pricing.lambda_reserve[i] * bess_ops[i, :reserve] +
                        base.pricing.lambda_frp_up[i] * bess_ops[i, :frp_up] +
                        base.pricing.lambda_frp_down[i] * bess_ops[i, :frp_down]
                        for i in 1:nrow(bess_ops))
                end
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                push!(rows, (day=day_idx+1, config=name, capacity_mwh=bess_cfg.capacity,
                             power_mw=bess_cfg.charge_max,
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0, as_revenue=as_rev,
                             cycles=cyc, solve_time=elapsed))
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 2] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study2_battery_ablation")
    return df
end

# ============================================================================
# Study 3 — Extreme / Stress Cases (multi-day)
# ============================================================================
function run_study3_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 3: Extreme Cases — $n_days days × 10 cases")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        T = length(base.horizon)

        cases = Pair{String, Tuple{Any, Bool}}[]
        push!(cases, "Base" => (base, true))

        max_base_upper = maximum(base.base_load_upper)
        tight_cap = max_base_upper + 2.0
        push!(cases, "Tight_IC_load" => (rebuild_with_interconnection(base; load_cap=tight_cap), true))
        push!(cases, "Tight_IC_ramp5" => (rebuild_with_interconnection(base; ramp_cap=5.0), true))
        push!(cases, "Tight_IC_both" => (rebuild_with_interconnection(base; load_cap=tight_cap, ramp_cap=5.0), true))

        zero_as = PricingData(base.pricing.lambda_energy, zeros(T), zeros(T), zeros(T), base.pricing.lambda_weight)
        push!(cases, "Zero_AS" => (rebuild_with_pricing(base, zero_as), true))

        extreme_p = PricingData(base.pricing.lambda_energy .* 3.0, base.pricing.lambda_reserve,
                                base.pricing.lambda_frp_up, base.pricing.lambda_frp_down, base.pricing.lambda_weight)
        push!(cases, "3x_Energy" => (rebuild_with_pricing(base, extreme_p), true))

        heavy_jobs = [Job(j.name, j.release, j.deadline, j.work * 2.0, j.rate_max * 1.5, j.tardiness_weight) for j in base.jobs]
        push!(cases, "Heavy_Jobs_2x" => (rebuild_with_jobs(base, heavy_jobs), true))

        shift = base.vcc.hardware_limit - maximum(base.base_load_upper) - 5.0
        if shift > 0
            shifted_load = base.base_load .+ shift
            shifted_upper = base.base_load_upper .+ shift
            d_shifted = ProblemDataVCC(base.horizon, base.delta_t, shifted_load, shifted_upper,
                                       base.jobs, base.bess, base.pricing, base.cost,
                                       base.interconnection, base.scenarios, base.vcc)
            push!(cases, "Min_Headroom_5MW" => (d_shifted, true))
        end

        push!(cases, "phi0_NoBatt" => (rebuild_with_phi(base, 0.01), false))

        d_phi1 = rebuild_with_phi(base, 0.99)
        push!(cases, "phi1_TightIC" => (rebuild_with_interconnection(d_phi1; load_cap=tight_cap, ramp_cap=8.0), true))

        rows = NamedTuple[]
        for (name, (data_case, enable_bat)) in cases
            sol, _, _, elapsed = solve_case(data_case; enable_battery=enable_bat, tag="d$(day_idx)_$name")
            if sol !== nothing
                mean_a = mean(sol[:vcc].a_factor)
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                push!(rows, (day=day_idx+1, case_name=name,
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0, mean_dvfs=mean_a,
                             solve_time=elapsed))
            else
                push!(rows, (day=day_idx+1, case_name=name,
                             objective=NaN, worst_revenue=NaN,
                             job_completion=NaN, mean_dvfs=NaN,
                             solve_time=elapsed))
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 3] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study3_extreme_cases")
    return df
end

# ============================================================================
# Study 4 — Interconnection Limit Sensitivity (multi-day)
# ============================================================================
function run_study4_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 4: IC Sensitivity — $n_days days × 16 config points")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        base_lc = base.interconnection.load_cap
        base_rc = base.interconnection.ramp_cap

        load_caps = [base_lc - 10, base_lc - 5, base_lc - 2, base_lc,
                     base_lc + 2, base_lc + 5, base_lc + 10, base_lc + 20]
        ramp_caps = [max(2.0, base_rc - 10), base_rc - 5, base_rc - 2, base_rc,
                     base_rc + 5, base_rc + 10, base_rc + 20, base_rc + 50]

        rows = NamedTuple[]
        for lc in load_caps
            lc > 0 || continue
            d = rebuild_with_interconnection(base; load_cap=lc)
            sol, model, _, elapsed = solve_case(d; tag="d$(day_idx)_lc=$lc")
            if sol !== nothing
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                # Extract LP dual sum for load constraint
                dual_sum = NaN
                try
                    duals = extract_lp_duals(model, d)
                    if duals !== nothing && haskey(duals, :total_lambda_load)
                        dual_sum = duals[:total_lambda_load]
                    end
                catch; end
                push!(rows, (day=day_idx+1, sweep="load_cap", param=lc,
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0,
                             total_load_slack=sol[:slack_violations][:total_load_slack],
                             dual_sum=dual_sum,
                             solve_time=elapsed))
            end
        end
        for rc in ramp_caps
            rc > 0 || continue
            d = rebuild_with_interconnection(base; ramp_cap=rc)
            sol, model, _, elapsed = solve_case(d; tag="d$(day_idx)_rc=$rc")
            if sol !== nothing
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                ramp_sl = sol[:slack_violations][:total_ramp_slack_up] +
                          sol[:slack_violations][:total_ramp_slack_down]
                # Extract LP dual sum for ramp constraints
                dual_sum = NaN
                try
                    duals = extract_lp_duals(model, d)
                    if duals !== nothing && haskey(duals, :total_lambda_ramp_up)
                        dual_sum = duals[:total_lambda_ramp_up] + duals[:total_lambda_ramp_down]
                    end
                catch; end
                push!(rows, (day=day_idx+1, sweep="ramp_cap", param=rc,
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0,
                             total_load_slack=ramp_sl,
                             dual_sum=dual_sum,
                             solve_time=elapsed))
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 4] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study4_ic_sensitivity")
    return df
end

# ============================================================================
# Study 4b — LP Dual Extraction (multi-day)
# ============================================================================
function run_study4b_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 4b: LP Duals — $n_days days (base case)")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        Tset = collect(base.horizon)
        T = length(Tset)

        rows = NamedTuple[]
        sol, model, _, elapsed = solve_case(base; tag="d$(day_idx)_base-dual")
        if sol !== nothing
            duals = extract_lp_duals(model, base)
            if duals !== nothing && haskey(duals, :lambda_load_per_period)
                for (i, t) in enumerate(Tset)
                    push!(rows, (day=day_idx+1, hour=t,
                                 lambda_load=duals[:lambda_load_per_period][i],
                                 lambda_ramp_up=duals[:lambda_ramp_up_per_period][i],
                                 lambda_ramp_down=duals[:lambda_ramp_down_per_period][i]))
                end
            end
        end

        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 4b] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study4b_lp_duals")
    return df
end

# ============================================================================
# Study 5 — Out-of-Sample Robustness (multi-day)
# ============================================================================
function run_study5_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 5: OOS Robustness — $n_days days")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        Tset = collect(base.horizon)
        T = length(Tset)
        ic = base.interconnection

        rows = NamedTuple[]

        sol, model, _, elapsed = solve_case(base; tag="d$(day_idx)_oos")
        if sol === nothing
            day_frames[day_idx + 1] = DataFrame(rows)
            Threads.atomic_add!(completed, 1)
            continue
        end

        vars = model.ext[:dc_vars]
        D_vals = [JuMP.value(vars.D[t]) for t in Tset]
        ΔD_vals = [JuMP.value(vars.ΔD[t]) for t in Tset]

        # In-sample violations
        n_in = length(base.scenarios)
        in_load_viols = 0; in_ramp_viols = 0
        for scen in base.scenarios
            prev_err = 0.0
            for k in 1:T
                if D_vals[k] + scen.load_error[k] - ic.load_cap > 1e-4; in_load_viols += 1; end
                err_diff = scen.load_error[k] - prev_err
                re = max(ΔD_vals[k]+err_diff - ic.ramp_cap, -(ΔD_vals[k]+err_diff) - ic.ramp_cap, 0.0)
                if re > 1e-4; in_ramp_viols += 1; end
                prev_err = scen.load_error[k]
            end
        end

        # Out-of-sample
        n_oos = 500
        Random.seed!(12345 + day_idx * 13)
        caiso_stats = load_caiso_frp_stats()
        pjm_stats = load_pjm_reserve_stats()
        oos_scens = generate_real_deployment_scenarios_vcc(caiso_stats, pjm_stats, T, n_oos)

        oos_load_viols = 0; oos_ramp_viols = 0
        oos_max_le = 0.0; oos_max_re = 0.0
        load_excesses = Float64[]; ramp_excesses = Float64[]
        for scen in oos_scens
            max_le = 0.0; max_re = 0.0; prev_err = 0.0
            for k in 1:T
                le = D_vals[k] + scen.load_error[k] - ic.load_cap
                if le > 1e-4; oos_load_viols += 1; max_le = max(max_le, le); end
                err_diff = scen.load_error[k] - prev_err
                re = max(ΔD_vals[k]+err_diff - ic.ramp_cap, -(ΔD_vals[k]+err_diff) - ic.ramp_cap, 0.0)
                if re > 1e-4; oos_ramp_viols += 1; max_re = max(max_re, re); end
                oos_max_le = max(oos_max_le, max(le, 0.0))
                oos_max_re = max(oos_max_re, re)
                prev_err = scen.load_error[k]
            end
            push!(load_excesses, max_le)
            push!(ramp_excesses, max_re)
        end

        total_checks = n_oos * T
        le_pos = filter(x -> x > 1e-4, load_excesses)
        re_pos = filter(x -> x > 1e-4, ramp_excesses)

        push!(rows, (day=day_idx+1,
                     in_load_viol_rate=in_load_viols / (n_in * T) * 100.0,
                     in_ramp_viol_rate=in_ramp_viols / (n_in * T) * 100.0,
                     oos_load_viol_rate=oos_load_viols / total_checks * 100.0,
                     oos_ramp_viol_rate=oos_ramp_viols / total_checks * 100.0,
                     oos_scen_load_pct=count(x -> x > 1e-4, load_excesses) / n_oos * 100.0,
                     oos_scen_ramp_pct=count(x -> x > 1e-4, ramp_excesses) / n_oos * 100.0,
                     oos_load_mean=isempty(le_pos) ? 0.0 : mean(le_pos),
                     oos_load_p95=isempty(le_pos) ? 0.0 : quantile(le_pos, 0.95),
                     oos_load_max=oos_max_le,
                     oos_ramp_mean=isempty(re_pos) ? 0.0 : mean(re_pos),
                     oos_ramp_p95=isempty(re_pos) ? 0.0 : quantile(re_pos, 0.95),
                     oos_ramp_max=oos_max_re))

        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 5] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study5_oos_robustness")
    return df
end

# ============================================================================
# Study 6 — DVFS Level Ablation (multi-day)
# ============================================================================
function run_study6_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 6: DVFS Level Ablation — $n_days days × 5 configs")
    println("="^80)

    dvfs_configs = [
        ("No_DVFS",    [1.0]),
        ("pm5_3lvl",   collect(0.95:0.05:1.05)),
        ("pm10_5lvl",  collect(0.90:0.05:1.10)),
        ("pm15_7lvl",  collect(0.85:0.05:1.15)),
        ("pm20_9lvl",  collect(0.80:0.05:1.20)),
    ]

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        rows = NamedTuple[]
        for (name, levels) in dvfs_configs
            sol, _, _, elapsed = solve_case_with_dvfs(base; tag="d$(day_idx)_$name", dvfs_levels=levels)
            if sol !== nothing
                a_factors = sol[:vcc].a_factor
                mean_a = mean(a_factors)
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                as_rev = sol[:enable_battery] ? sum(
                    base.pricing.lambda_reserve[i] * sol[:bess][i+1, :reserve] +
                    base.pricing.lambda_frp_up[i] * sol[:bess][i+1, :frp_up] +
                    base.pricing.lambda_frp_down[i] * sol[:bess][i+1, :frp_down]
                    for i in 1:length(base.horizon)) : 0.0
                dvfs_pen = base.cost.c1 * sum(abs(a - base.cost.a_ref) for a in a_factors)
                push!(rows, (day=day_idx+1, config=name, n_levels=length(levels),
                             objective=sol[:objective_value],
                             worst_revenue=sol[:worst_case_revenue],
                             job_completion=jc*100.0, mean_dvfs=mean_a,
                             as_revenue=as_rev, dvfs_penalty=dvfs_pen,
                             solve_time=elapsed))
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 6] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study6_dvfs_level_ablation")
    return df
end

# ============================================================================
# Study 4c — IC Sensitivity × Load Composition (multi-day)
# ============================================================================
function run_study4c_multiday(; n_days::Int=N_DAYS)
    fixed_fractions = [0.60, 0.80, 0.95]
    n_ff = length(fixed_fractions)
    println("\n" * "="^80)
    println("STUDY 4c: IC × Composition — $n_days days × $(n_ff) ff × 16 IC pts")
    println("="^80)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        T = length(base.horizon)
        base_mean = mean(base.base_load)
        base_avg_job = sum(j.work for j in base.jobs) / T
        total_base = base_mean + base_avg_job

        # Base IC grid (same as Study 4)
        base_lc = base.interconnection.load_cap
        base_rc = base.interconnection.ramp_cap
        load_caps = [base_lc - 10, base_lc - 5, base_lc - 2, base_lc,
                     base_lc + 2, base_lc + 5, base_lc + 10, base_lc + 20]
        ramp_caps = [max(2.0, base_rc - 10), base_rc - 5, base_rc - 2, base_rc,
                     base_rc + 5, base_rc + 10, base_rc + 20, base_rc + 50]

        rows = NamedTuple[]
        for ff in fixed_fractions
            ff_pct = ff * 100.0
            d_comp = rebuild_with_load_composition(base, total_base, ff)

            # (a) load-cap sweep
            for lc in load_caps
                lc > 0 || continue
                d = rebuild_with_interconnection(d_comp; load_cap=lc)
                sol, model, _, elapsed = solve_case(d; tag="d$(day_idx)_ff$(ff_pct)_lc=$lc")
                if sol !== nothing
                    jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                    dual_sum = NaN
                    try
                        duals = extract_lp_duals(model, d)
                        if duals !== nothing && haskey(duals, :total_lambda_load)
                            dual_sum = duals[:total_lambda_load]
                        end
                    catch; end
                    push!(rows, (day=day_idx+1, fixed_frac=ff_pct,
                                 sweep="load_cap", param=lc,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 job_completion=jc*100.0,
                                 total_load_slack=sol[:slack_violations][:total_load_slack],
                                 dual_sum=dual_sum,
                                 solve_time=elapsed))
                end
            end

            # (b) ramp-cap sweep
            for rc in ramp_caps
                rc > 0 || continue
                d = rebuild_with_interconnection(d_comp; ramp_cap=rc)
                sol, model, _, elapsed = solve_case(d; tag="d$(day_idx)_ff$(ff_pct)_rc=$rc")
                if sol !== nothing
                    jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                    ramp_sl = sol[:slack_violations][:total_ramp_slack_up] +
                              sol[:slack_violations][:total_ramp_slack_down]
                    dual_sum = NaN
                    try
                        duals = extract_lp_duals(model, d)
                        if duals !== nothing && haskey(duals, :total_lambda_ramp_up)
                            dual_sum = duals[:total_lambda_ramp_up] + duals[:total_lambda_ramp_down]
                        end
                    catch; end
                    push!(rows, (day=day_idx+1, fixed_frac=ff_pct,
                                 sweep="ramp_cap", param=rc,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 job_completion=jc*100.0,
                                 total_load_slack=ramp_sl,
                                 dual_sum=dual_sum,
                                 solve_time=elapsed))
                end
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 4c] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study4c_ic_composition")
    return df
end

# ============================================================================
# Study 7 — Load Composition Sensitivity (multi-day)
# ============================================================================
function run_study7_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 7: Load Composition — $n_days days × 28 configs")
    println("="^80)

    fixed_fractions = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    day_frames = Vector{DataFrame}(undef, n_days)
    completed = Threads.Atomic{Int}(0)
    t_start = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)
        T = length(base.horizon)
        base_mean = mean(base.base_load)
        base_avg_job = sum(j.work for j in base.jobs) / T
        total_base = base_mean + base_avg_job

        # Stressed: headroom ≈ 24 MW at natural ff + 1.5× job demand (from Study 1b)
        hw = base.vcc.hardware_limit
        eps_abs = hw * abs(base.vcc.epsilon_min)
        max_bl = maximum(base.base_load)
        stressed_base_mean = (hw - 24.0 - eps_abs) * base_mean / max_bl
        total_stressed = stressed_base_mean + base_avg_job * 1.5

        total_levels = [
            ("Light",    total_base * 0.94),
            ("Base",     total_base),
            ("Heavy",    total_base * 1.05),
            ("Stressed", total_stressed),
        ]

        rows = NamedTuple[]
        for (level_name, total_load) in total_levels
            for ff in fixed_fractions
                d = rebuild_with_load_composition(base, total_load, ff)
                hr = d.vcc.hardware_limit - maximum(d.base_load_upper)
                sol, _, _, elapsed = solve_case(d; tag="d$(day_idx)_$(level_name)_ff=$ff")
                if sol !== nothing
                    mean_a = mean(sol[:vcc].a_factor)
                    jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                    push!(rows, (day=day_idx+1, total_level=level_name,
                                 total_load=total_load, fixed_frac=ff*100.0,
                                 headroom=hr,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 job_completion=jc*100.0, mean_dvfs=mean_a,
                                 solve_time=elapsed))
                end
            end
        end
        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 7] Day %2d done (%d/%d, %.0fs elapsed)\n", day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study7_load_composition")
    return df
end

# ============================================================================
# Study 8 — Stressed Grid: Battery Value at IC Binding Points (multi-day)
# ============================================================================
"""
Sweep the interconnection limits around the identified binding points
(load_cap ≈ 95 MW, ramp_cap ≈ 10 MW/h) and compare with/without the base-case
36 MWh / 12 MW BESS.  Two sub-sweeps per day:
  • Load sweep: vary load_cap ∈ [75, 80, 85, 90, 95, 100, 105, 110, 115] MW,
                keep ramp_cap fixed at 10 MW/h (binding ramp condition).
  • Ramp sweep: vary ramp_cap ∈ [3, 5, 7, 10, 12, 15, 20, 30] MW/h,
                keep load_cap fixed at 95 MW (binding load condition).
For each (config, sweep, param) triple the function records the MILP objective,
worst-case revenue, LP dual sum (FD values can be computed in Python), job
completion, constraint slack, and AS revenue so the Python plotter can show
how the battery shifts both the objective curve and the shadow prices.
"""
function run_study8_multiday(; n_days::Int=N_DAYS)
    println("\n" * "="^80)
    println("STUDY 8: Stressed Grid — Battery Value at IC Binding Points")
    println("="^80)
    println("  Binding points: load_cap = 95 MW, ramp_cap = 10 MW/h")
    println("  Load sweep: 75–115 MW with ramp fixed at 10 MW/h")
    println("  Ramp sweep: 3–30 MW/h with load fixed at 95 MW")
    println("  Configs: NoBatt vs BESS_36_12 (36 MWh / 12 MW)")
    println()

    # Binding points identified from fig_study4_ic_sensitivity.pdf
    BIND_LOAD = 95.0   # MW  — peak-load breakpoint (high FD shadow price)
    BIND_RAMP = 10.0   # MW/h — ramp-rate breakpoint (high FD shadow price)

    load_cap_sweep = [75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    ramp_cap_sweep = [3.0,  5.0,  7.0, 10.0,  12.0,  15.0,  20.0,  30.0]

    # Dummy BESS for "no-battery" case: negligible capacity, zero power
    NOBATT = BESSParameters(1.0, 0.5, 0.2, 0.9, 0.0, 0.0, 0.95, 0.95, 45.0, 0.5)

    day_frames = Vector{DataFrame}(undef, n_days)
    completed  = Threads.Atomic{Int}(0)
    t_start    = time()

    Threads.@threads for day_idx in 0:(n_days - 1)
        base = make_day_data(day_idx)

        # Per-day BESS config table: (label, enable_battery, bess_params)
        bess_cfgs = [
            ("NoBatt",     false, NOBATT),
            ("BESS_36_12", true,  base.bess),   # 36 MWh / 12 MW base case
        ]

        rows = NamedTuple[]

        # ── Load-cap sweep (ramp fixed at BIND_RAMP) ─────────────────────────
        for (cfg_name, enable_batt, bess_cfg) in bess_cfgs
            d_bess = rebuild_with_bess(base, bess_cfg)
            for lc in load_cap_sweep
                lc > 0.0 || continue
                d = rebuild_with_interconnection(d_bess; load_cap=lc, ramp_cap=BIND_RAMP)
                sol, model, _, elapsed = solve_case(d; enable_battery=enable_batt,
                                                    tag="d$(day_idx)_$(cfg_name)_lc=$(lc)")
                if sol !== nothing
                    jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                    load_sl = sol[:slack_violations][:total_load_slack]
                    dual_sum = NaN
                    try
                        duals = extract_lp_duals(model, d)
                        if duals !== nothing && haskey(duals, :total_lambda_load)
                            dual_sum = duals[:total_lambda_load]
                        end
                    catch; end
                    as_rev = 0.0
                    if enable_batt && sol[:enable_battery]
                        bops = sol[:bess][sol[:bess].time .> 0, :]
                        as_rev = sum(
                            base.pricing.lambda_reserve[i]   * bops[i, :reserve]   +
                            base.pricing.lambda_frp_up[i]    * bops[i, :frp_up]    +
                            base.pricing.lambda_frp_down[i]  * bops[i, :frp_down]
                            for i in 1:nrow(bops))
                    end
                    push!(rows, (day=day_idx+1, config=cfg_name, sweep="load_cap",
                                 param=lc,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 job_completion=jc*100.0,
                                 total_slack=load_sl,
                                 dual_sum=dual_sum,
                                 as_revenue=as_rev,
                                 solve_time=elapsed))
                end
            end
        end

        # ── Ramp-cap sweep (load fixed at BIND_LOAD) ─────────────────────────
        for (cfg_name, enable_batt, bess_cfg) in bess_cfgs
            d_bess = rebuild_with_bess(base, bess_cfg)
            for rc in ramp_cap_sweep
                rc > 0.0 || continue
                d = rebuild_with_interconnection(d_bess; load_cap=BIND_LOAD, ramp_cap=rc)
                sol, model, _, elapsed = solve_case(d; enable_battery=enable_batt,
                                                    tag="d$(day_idx)_$(cfg_name)_rc=$(rc)")
                if sol !== nothing
                    jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                    ramp_sl = sol[:slack_violations][:total_ramp_slack_up] +
                              sol[:slack_violations][:total_ramp_slack_down]
                    dual_sum = NaN
                    try
                        duals = extract_lp_duals(model, d)
                        if duals !== nothing && haskey(duals, :total_lambda_ramp_up)
                            dual_sum = duals[:total_lambda_ramp_up] +
                                       duals[:total_lambda_ramp_down]
                        end
                    catch; end
                    as_rev = 0.0
                    if enable_batt && sol[:enable_battery]
                        bops = sol[:bess][sol[:bess].time .> 0, :]
                        as_rev = sum(
                            base.pricing.lambda_reserve[i]   * bops[i, :reserve]   +
                            base.pricing.lambda_frp_up[i]    * bops[i, :frp_up]    +
                            base.pricing.lambda_frp_down[i]  * bops[i, :frp_down]
                            for i in 1:nrow(bops))
                    end
                    push!(rows, (day=day_idx+1, config=cfg_name, sweep="ramp_cap",
                                 param=rc,
                                 objective=sol[:objective_value],
                                 worst_revenue=sol[:worst_case_revenue],
                                 job_completion=jc*100.0,
                                 total_slack=ramp_sl,
                                 dual_sum=dual_sum,
                                 as_revenue=as_rev,
                                 solve_time=elapsed))
                end
            end
        end

        day_frames[day_idx + 1] = DataFrame(rows)
        c = Threads.atomic_add!(completed, 1) + 1
        @printf("  [Study 8] Day %2d done (%d/%d, %.0fs elapsed)\n",
                day_idx+1, c, n_days, time()-t_start)
    end

    df = vcat(day_frames...)
    save_csv(df, "study8_stressed_bess_value")
    return df
end

# ============================================================================
# Main Driver
# ============================================================================
function main_multiday()
    study_arg = length(ARGS) >= 1 ? ARGS[1] : "all"

    println("="^80)
    println("DC + BESS Co-optimization — Multi-Day Ablation Study (31 days)")
    println("="^80)
    println("  Days: $N_DAYS | Threads: $(Threads.nthreads())")
    println("  Solver: $(DCFlex.solver_name())")
    println("  Output: $CSV_DIR")
    println("  Study: $study_arg")
    println("  Started: $(Dates.now())")
    println()

    study_map = Dict(
        "1"  => run_study1_multiday,
        "1b" => run_study1b_multiday,
        "2"  => run_study2_multiday,
        "3"  => run_study3_multiday,
        "4"  => run_study4_multiday,
        "4b" => run_study4b_multiday,
        "4c" => run_study4c_multiday,
        "5"  => run_study5_multiday,
        "6"  => run_study6_multiday,
        "7"  => run_study7_multiday,
        "8"  => run_study8_multiday,
    )

    if study_arg == "all"
        for key in ["1", "1b", "2", "3", "4", "4b", "4c", "5", "6", "7", "8"]
            study_map[key]()
        end
    elseif haskey(study_map, study_arg)
        study_map[study_arg]()
    else
        error("Unknown study: '$study_arg'. Use one of: all, 1, 1b, 2, 3, 4, 4b, 4c, 5, 6, 7, 8")
    end

    println("\n" * "="^80)
    println("  All requested studies complete.  $(Dates.now())")
    println("  CSV results in: $CSV_DIR")
    println("="^80)
end

main_multiday()
