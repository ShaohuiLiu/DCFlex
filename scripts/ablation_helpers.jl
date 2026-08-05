"""
Reusable ablation helpers for DC+BESS co-optimization (VCC formulation).

The public study drivers use these functions for five families of experiments:

  Study 1 — φ (Flexible Load Fraction) Sensitivity
            Sweeps φ from near-zero to near-one, quantifying how the split
            between fixed and DVFS-sensitive load affects cost and scheduling.

  Study 2 — Battery Ablation
            Compares no-battery vs. base-case battery vs. larger battery,
            isolating the value of co-located BESS.

  Study 3 — Extreme / Stress Cases
            Edge scenarios: very tight interconnection, zero ancillary-service
            revenue, extreme energy prices, heavy job load, minimal headroom.

  Study 4 — Interconnection Limit Sensitivity (Marginal Cost)
            Parametrically varies load_cap and ramp_cap, recording dual
            information (objective change per MW) to approximate λ on the
            interconnection constraints.

  Study 5 — Out-of-Sample Robustness Test
            Solves with in-sample scenarios (50), then fixes first-stage
            decisions and evaluates constraint violations on 500 fresh
            out-of-sample scenarios.

The calibrated base case is defined in `real_market_data_case_vcc.jl`.
"""

using DCFlex
using JuMP
using HiGHS
using Printf
using Statistics
using Random
using DataFrames
using MathOptInterface
const MOI = MathOptInterface

# ── Load the VCC problem-data generator from the sibling script ──────────
include(joinpath(@__DIR__, "real_market_data_case_vcc.jl"))

# ============================================================================
# Helpers
# ============================================================================

"""Build, solve, and summarise a VCC instance. Returns (summary_dict, model, data)."""
function solve_case(data::ProblemDataVCC; enable_battery::Bool=true, tag::String="", quiet::Bool=true)
    model = build_model_vcc(data; enable_battery=enable_battery)
    set_silent(model)
    set_time_limit_sec(model, 300.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)
    t0 = time()
    optimize!(model)
    elapsed = time() - t0
    status = termination_status(model)
    if !quiet
        println("  [$tag] status=$status  time=$(round(elapsed, digits=1))s")
    end
    if has_values(model)
        sol = solution_summary_vcc(model)
        return sol, model, data, elapsed
    else
        return nothing, model, data, elapsed
    end
end

"""Rebuild ProblemDataVCC with modified VCCParameters (new φ)."""
function rebuild_with_phi(base_data::ProblemDataVCC, phi_new::Float64)
    vcc_new = VCCParameters(
        base_data.vcc.hardware_limit, phi_new;
        forecast_error_sigma=base_data.vcc.forecast_error_sigma,
        epsilon_max=base_data.vcc.epsilon_max,
        epsilon_min=base_data.vcc.epsilon_min,
        enable_vcc_variable=base_data.vcc.enable_vcc_variable,
        idle_resource_cost=base_data.vcc.idle_resource_cost)
    base_load_upper = [base_data.base_load[i] - vcc_new.hardware_limit * vcc_new.epsilon_min
                       for i in 1:length(base_data.base_load)]
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_load_upper, base_data.jobs, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, vcc_new)
end

"""Rebuild ProblemDataVCC with modified InterconnectionLimits."""
function rebuild_with_interconnection(base_data::ProblemDataVCC;
        load_cap::Float64=base_data.interconnection.load_cap,
        ramp_cap::Float64=base_data.interconnection.ramp_cap)
    ic = base_data.interconnection
    ic_new = InterconnectionLimits(load_cap, ramp_cap,
                ic.delta_load_risk, ic.delta_ramp_risk,
                ic.initial_net_load, ic.big_m_load, ic.big_m_ramp)
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, base_data.jobs, base_data.bess,
                   base_data.pricing, base_data.cost, ic_new,
                   base_data.scenarios, base_data.vcc)
end

"""Rebuild ProblemDataVCC with modified BESS parameters."""
function rebuild_with_bess(base_data::ProblemDataVCC, bess_new::BESSParameters)
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, base_data.jobs, bess_new,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

"""Rebuild ProblemDataVCC with modified PricingData."""
function rebuild_with_pricing(base_data::ProblemDataVCC, pricing_new::PricingData)
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, base_data.jobs, base_data.bess,
                   pricing_new, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

"""Rebuild ProblemDataVCC with a different set of scenarios."""
function rebuild_with_scenarios(base_data::ProblemDataVCC, scens::Vector{Scenario})
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, base_data.jobs, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   scens, base_data.vcc)
end

"""Rebuild ProblemDataVCC with modified jobs."""
function rebuild_with_jobs(base_data::ProblemDataVCC, jobs_new::Vector{Job})
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, jobs_new, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

# ============================================================================
# Study 1 — φ (Flexible Load Fraction) Sensitivity
# ============================================================================

function study1_phi_sensitivity(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 1: φ (Flexible Load Fraction) Sensitivity")
    println(io, "="^90)
    println(io, """
    P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t
    φ ≈ 0  →  fixed load is constant (insensitive to DVFS)
    φ ≈ 1  →  fixed load scales linearly with DVFS frequency
    Base case: φ = $(base_data.vcc.phi)
    """)

    phi_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 0.99]
    results = DataFrame(phi=Float64[], objective=Float64[], worst_revenue=Float64[],
                        job_completion=Float64[], mean_dvfs=Float64[],
                        total_load_slack=Float64[], total_hw_slack=Float64[],
                        as_revenue=Float64[], solve_time=Float64[])

    for phi in phi_values
        print("  φ=$(round(phi, digits=2))  … ")
        data_phi = rebuild_with_phi(base_data, phi)
        sol, _, _, elapsed = solve_case(data_phi; tag="φ=$phi")
        if sol !== nothing
            mean_a = mean(sol[:vcc].a_factor)
            as_rev = sol[:enable_battery] ? sum(
                base_data.pricing.lambda_reserve[i] * sol[:bess][i+1, :reserve] +
                base_data.pricing.lambda_frp_up[i] * sol[:bess][i+1, :frp_up] +
                base_data.pricing.lambda_frp_down[i] * sol[:bess][i+1, :frp_down]
                for i in 1:length(base_data.horizon)) : 0.0
            push!(results, (phi, sol[:objective_value], sol[:worst_case_revenue],
                            1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work),
                            mean_a, sol[:slack_violations][:total_load_slack],
                            sol[:slack_violations][:total_hardware_slack],
                            as_rev, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2)),  a_mean=$(round(mean_a, digits=3))")
        else
            println("INFEASIBLE")
        end
    end

    # Print table
    println(io, @sprintf("  %-6s  %12s  %12s  %8s  %8s  %10s  %10s  %10s  %6s",
                         "φ", "Objective(\$)", "WorstRev(\$)", "Job(%)", "mean(a)",
                         "LoadSlack", "HWSslack", "AS Rev(\$)", "Time"))
    println(io, "  " * "-"^96)
    for r in eachrow(results)
        println(io, @sprintf("  %-6.2f  %12.2f  %12.2f  %7.1f%%  %8.4f  %10.4f  %10.4f  %10.2f  %5.1fs",
                             r.phi, r.objective, r.worst_revenue, r.job_completion*100,
                             r.mean_dvfs, r.total_load_slack, r.total_hw_slack,
                             r.as_revenue, r.solve_time))
    end
    println(io)

    # Marginal analysis
    if nrow(results) >= 2
        println(io, "  Marginal Δ(objective) / Δφ:")
        for i in 2:nrow(results)
            dphi = results.phi[i] - results.phi[i-1]
            dobj = results.objective[i] - results.objective[i-1]
            println(io, @sprintf("    φ %.2f→%.2f : Δobj = %+.2f  (%.1f \$/unit-φ)",
                                 results.phi[i-1], results.phi[i], dobj, dobj/dphi))
        end
    end
    return results
end

# ============================================================================
# Study 2 — Battery Ablation
# ============================================================================

function study2_battery_ablation(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 2: Battery Ablation")
    println(io, "="^90)
    println(io, """
    Compares the system with (a) no battery, (b) base battery (36 MWh / 12 MW),
    (c) larger battery variants.
    """)

    configs = [
        ("No Battery",       false, BESSParameters(1.0, 0.5, 0.2, 0.9, 0.0, 0.0, 0.95, 0.95, 45.0, 0.5)),
        ("Small (18/6)",     true,  BESSParameters(18.0, 0.6, 0.15, 0.90, 6.0, 6.0, 0.95, 0.95, 45.0, 0.5)),
        ("Base (36/12)",     true,  base_data.bess),
        ("Large (72/24)",    true,  BESSParameters(72.0, 0.6, 0.15, 0.90, 24.0, 24.0, 0.95, 0.95, 45.0, 0.5)),
        ("XL (108/36)",      true,  BESSParameters(108.0, 0.6, 0.15, 0.90, 36.0, 36.0, 0.95, 0.95, 45.0, 0.5)),
    ]

    results = DataFrame(config=String[], enable_batt=Bool[], capacity_mwh=Float64[],
                        power_mw=Float64[], objective=Float64[], worst_revenue=Float64[],
                        job_completion=Float64[], total_load_slack=Float64[],
                        total_ramp_slack=Float64[], as_revenue=Float64[],
                        energy_charged=Float64[], energy_discharged=Float64[],
                        cycles=Float64[], solve_time=Float64[])

    for (name, enable, bess_cfg) in configs
        print("  $name … ")
        data_batt = rebuild_with_bess(base_data, bess_cfg)
        sol, model, _, elapsed = solve_case(data_batt; enable_battery=enable, tag=name)
        if sol !== nothing
            as_rev = 0.0; charged = 0.0; discharged = 0.0; cyc = 0.0
            if enable && sol[:enable_battery]
                bess_ops = sol[:bess][sol[:bess].time .> 0, :]
                charged = sum(bess_ops.charge)
                discharged = sum(bess_ops.discharge)
                cyc = discharged / bess_cfg.capacity
                as_rev = sum(
                    base_data.pricing.lambda_reserve[i] * bess_ops[i, :reserve] +
                    base_data.pricing.lambda_frp_up[i] * bess_ops[i, :frp_up] +
                    base_data.pricing.lambda_frp_down[i] * bess_ops[i, :frp_down]
                    for i in 1:nrow(bess_ops))
            end
            job_comp = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            ramp_sl = sol[:slack_violations][:total_ramp_slack_up] +
                      sol[:slack_violations][:total_ramp_slack_down]
            push!(results, (name, enable, bess_cfg.capacity, bess_cfg.charge_max,
                            sol[:objective_value], sol[:worst_case_revenue], job_comp,
                            sol[:slack_violations][:total_load_slack], ramp_sl,
                            as_rev, charged, discharged, cyc, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2))")
        else
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("  %-16s %6s %6s  %12s  %12s  %7s  %9s  %9s  %10s  %6s  %6s  %5s  %5s",
                         "Config", "MWh", "MW", "Obj(\$)", "WrstRev(\$)", "Job(%)",
                         "LdSlack", "RmpSlack", "AS Rev(\$)",
                         "Chg", "Dis", "Cyc", "Time"))
    println(io, "  " * "-"^130)
    for r in eachrow(results)
        println(io, @sprintf("  %-16s %6.0f %6.0f  %12.2f  %12.2f  %6.1f%%  %9.4f  %9.4f  %10.2f  %6.1f  %6.1f  %5.3f  %4.1fs",
                             r.config, r.capacity_mwh, r.power_mw, r.objective,
                             r.worst_revenue, r.job_completion*100,
                             r.total_load_slack, r.total_ramp_slack, r.as_revenue,
                             r.energy_charged, r.energy_discharged, r.cycles, r.solve_time))
    end

    # Value of battery
    if nrow(results) >= 2
        no_batt = results[results.config .== "No Battery", :]
        base_batt = results[results.config .== "Base (36/12)", :]
        if nrow(no_batt) == 1 && nrow(base_batt) == 1
            batt_value = base_batt.objective[1] - no_batt.objective[1]
            println(io, @sprintf("\n  Value of base battery: \$%.2f / day", batt_value))
            println(io, @sprintf("  Value per MWh capacity: \$%.2f / MWh-day", batt_value / 36.0))
        end
    end
    return results
end

# ============================================================================
# Study 3 — Extreme / Stress Cases
# ============================================================================

function study3_extreme_cases(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 3: Extreme / Stress Cases")
    println(io, "="^90)

    results = DataFrame(case=String[], objective=Float64[], worst_revenue=Float64[],
                        job_completion=Float64[], total_load_slack=Float64[],
                        total_hw_slack=Float64[], mean_dvfs=Float64[],
                        solve_time=Float64[])

    cases = Pair{String, Any}[]  # ordered list of (name => data)

    # --- 3a. Base case (reference) ---
    push!(cases, "Base Case" => base_data)

    # --- 3b. Very tight interconnection (load_cap barely above max base_load_upper) ---
    max_base_upper = maximum(base_data.base_load_upper)
    tight_cap = max_base_upper + 2.0  # only 2 MW headroom above fixed load upper bound
    push!(cases, "Tight IC (load)" => rebuild_with_interconnection(base_data; load_cap=tight_cap))

    # --- 3c. Very tight ramp limit ---
    push!(cases, "Tight IC (ramp=5)" => rebuild_with_interconnection(base_data; ramp_cap=5.0))

    # --- 3d. Both tight ---
    tmp = rebuild_with_interconnection(base_data; load_cap=tight_cap, ramp_cap=5.0)
    push!(cases, "Tight IC (both)" => tmp)

    # --- 3e. Zero AS revenue ---
    T = length(base_data.horizon)
    zero_as_pricing = PricingData(
        base_data.pricing.lambda_energy,
        zeros(T), zeros(T), zeros(T),
        base_data.pricing.lambda_weight)
    push!(cases, "Zero AS Revenue" => rebuild_with_pricing(base_data, zero_as_pricing))

    # --- 3f. Extreme energy prices (3x) ---
    extreme_pricing = PricingData(
        base_data.pricing.lambda_energy .* 3.0,
        base_data.pricing.lambda_reserve,
        base_data.pricing.lambda_frp_up,
        base_data.pricing.lambda_frp_down,
        base_data.pricing.lambda_weight)
    push!(cases, "3× Energy Price" => rebuild_with_pricing(base_data, extreme_pricing))

    # --- 3g. Heavy job load (double work requirements) ---
    heavy_jobs = [Job(j.name, j.release, j.deadline, j.work * 2.0, j.rate_max * 1.5, j.tardiness_weight)
                  for j in base_data.jobs]
    push!(cases, "Heavy Jobs (2×)" => rebuild_with_jobs(base_data, heavy_jobs))

    # --- 3h. Minimal headroom: raise base load so headroom ≈ 5 MW ---
    shift = base_data.vcc.hardware_limit - maximum(base_data.base_load_upper) - 5.0
    if shift > 0
        shifted_load = base_data.base_load .+ shift
        shifted_upper = base_data.base_load_upper .+ shift
        push!(cases, "Minimal Headroom (5 MW)" => ProblemDataVCC(
            base_data.horizon, base_data.delta_t, shifted_load, shifted_upper,
            base_data.jobs, base_data.bess, base_data.pricing, base_data.cost,
            base_data.interconnection, base_data.scenarios, base_data.vcc))
    end

    # --- 3i. φ near zero (almost all fixed) + no battery ---
    push!(cases, "φ≈0, No Battery" => rebuild_with_phi(base_data, 0.01))

    # --- 3j. φ near one + tight IC ---
    d = rebuild_with_phi(base_data, 0.99)
    push!(cases, "φ≈1, Tight IC" => rebuild_with_interconnection(d; load_cap=tight_cap, ramp_cap=8.0))

    for (name, data_case) in cases
        print("  $name … ")
        enable_bat = (name != "φ≈0, No Battery")  # disable battery for this specific case
        sol, _, _, elapsed = solve_case(data_case; enable_battery=enable_bat, tag=name)
        if sol !== nothing
            mean_a = mean(sol[:vcc].a_factor)
            job_comp = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            push!(results, (name, sol[:objective_value], sol[:worst_case_revenue],
                            job_comp, sol[:slack_violations][:total_load_slack],
                            sol[:slack_violations][:total_hardware_slack],
                            mean_a, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2))")
        else
            push!(results, (name, NaN, NaN, NaN, NaN, NaN, NaN, elapsed))
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("  %-28s  %12s  %12s  %7s  %10s  %10s  %8s  %6s",
                         "Case", "Obj(\$)", "WrstRev(\$)", "Job(%)", "LdSlack",
                         "HWSlack", "mean(a)", "Time"))
    println(io, "  " * "-"^105)
    for r in eachrow(results)
        obj_s = isnan(r.objective) ? "INFEASIBLE" : @sprintf("%12.2f", r.objective)
        rev_s = isnan(r.worst_revenue) ? "  -  " : @sprintf("%12.2f", r.worst_revenue)
        jc_s  = isnan(r.job_completion) ? "  -  " : @sprintf("%6.1f%%", r.job_completion*100)
        println(io, @sprintf("  %-28s  %12s  %12s  %7s  %10.4f  %10.4f  %8.4f  %5.1fs",
                             r.case, obj_s, rev_s, jc_s,
                             isnan(r.total_load_slack) ? 0.0 : r.total_load_slack,
                             isnan(r.total_hw_slack) ? 0.0 : r.total_hw_slack,
                             isnan(r.mean_dvfs) ? 0.0 : r.mean_dvfs,
                             r.solve_time))
    end
    return results
end

# ============================================================================
# Study 4 — Interconnection Limit Sensitivity (Marginal Costs)
# ============================================================================

function study4_interconnection_sensitivity(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 4: Interconnection Limit Sensitivity (Marginal Cost λ)")
    println(io, "="^90)
    println(io, """
    We parametrically sweep load_cap and ramp_cap around the base values and
    record the change in objective.  The finite-difference quotient
        λ ≈ Δ(objective) / Δ(limit)
    approximates the shadow price (dual variable) of each interconnection
    constraint, i.e., the marginal value (\$/MW or \$/MW·h⁻¹) of relaxing
    the corresponding limit by 1 unit.
    Base load_cap = $(base_data.interconnection.load_cap) MW
    Base ramp_cap = $(base_data.interconnection.ramp_cap) MW/h
    """)

    # --- 4a. Load capacity sweep ---
    base_load_cap = base_data.interconnection.load_cap
    load_caps = [base_load_cap - 10.0, base_load_cap - 5.0, base_load_cap - 2.0,
                 base_load_cap, base_load_cap + 2.0, base_load_cap + 5.0,
                 base_load_cap + 10.0, base_load_cap + 20.0]

    lc_results = DataFrame(load_cap=Float64[], objective=Float64[], worst_revenue=Float64[],
                           total_load_slack=Float64[], job_completion=Float64[], solve_time=Float64[])

    println(io, "  4a. Load Capacity (peak load limit) Sweep:")
    for lc in load_caps
        (lc > 0) || continue
        print("  load_cap=$lc … ")
        d = rebuild_with_interconnection(base_data; load_cap=lc)
        sol, _, _, elapsed = solve_case(d; tag="lc=$lc")
        if sol !== nothing
            jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            push!(lc_results, (lc, sol[:objective_value], sol[:worst_case_revenue],
                               sol[:slack_violations][:total_load_slack], jc, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2))")
        else
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("  %-12s  %12s  %12s  %10s  %7s  %6s",
                         "load_cap(MW)", "Obj(\$)", "WrstRev(\$)", "LdSlack", "Job(%)", "Time"))
    println(io, "  " * "-"^70)
    for r in eachrow(lc_results)
        println(io, @sprintf("  %-12.1f  %12.2f  %12.2f  %10.4f  %6.1f%%  %5.1fs",
                             r.load_cap, r.objective, r.worst_revenue,
                             r.total_load_slack, r.job_completion*100, r.solve_time))
    end

    # Marginal cost
    if nrow(lc_results) >= 2
        println(io, "\n  Marginal cost of load capacity (λ_load ≈ Δobj/Δcap):")
        for i in 2:nrow(lc_results)
            dc = lc_results.load_cap[i] - lc_results.load_cap[i-1]
            dobj = lc_results.objective[i] - lc_results.objective[i-1]
            abs(dc) < 1e-6 && continue
            lambda = dobj / dc
            println(io, @sprintf("    cap %.1f→%.1f MW : λ = %+.2f \$/MW",
                                 lc_results.load_cap[i-1], lc_results.load_cap[i], lambda))
        end
    end

    # --- 4b. Ramp capacity sweep ---
    base_ramp_cap = base_data.interconnection.ramp_cap
    ramp_caps = [max(2.0, base_ramp_cap - 10.0), base_ramp_cap - 5.0,
                 base_ramp_cap - 2.0, base_ramp_cap,
                 base_ramp_cap + 5.0, base_ramp_cap + 10.0,
                 base_ramp_cap + 20.0, base_ramp_cap + 50.0]

    rc_results = DataFrame(ramp_cap=Float64[], objective=Float64[], worst_revenue=Float64[],
                           total_ramp_slack=Float64[], job_completion=Float64[], solve_time=Float64[])

    println(io, "\n  4b. Ramp Capacity (ramp rate limit) Sweep:")
    for rc in ramp_caps
        (rc > 0) || continue
        print("  ramp_cap=$rc … ")
        d = rebuild_with_interconnection(base_data; ramp_cap=rc)
        sol, _, _, elapsed = solve_case(d; tag="rc=$rc")
        if sol !== nothing
            jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            ramp_sl = sol[:slack_violations][:total_ramp_slack_up] +
                      sol[:slack_violations][:total_ramp_slack_down]
            push!(rc_results, (rc, sol[:objective_value], sol[:worst_case_revenue],
                               ramp_sl, jc, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2))")
        else
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("\n  %-14s  %12s  %12s  %10s  %7s  %6s",
                         "ramp_cap(MW/h)", "Obj(\$)", "WrstRev(\$)", "RmpSlack", "Job(%)", "Time"))
    println(io, "  " * "-"^70)
    for r in eachrow(rc_results)
        println(io, @sprintf("  %-14.1f  %12.2f  %12.2f  %10.4f  %6.1f%%  %5.1fs",
                             r.ramp_cap, r.objective, r.worst_revenue,
                             r.total_ramp_slack, r.job_completion*100, r.solve_time))
    end

    if nrow(rc_results) >= 2
        println(io, "\n  Marginal cost of ramp capacity (λ_ramp ≈ Δobj/Δcap):")
        for i in 2:nrow(rc_results)
            dc = rc_results.ramp_cap[i] - rc_results.ramp_cap[i-1]
            dobj = rc_results.objective[i] - rc_results.objective[i-1]
            abs(dc) < 1e-6 && continue
            lambda = dobj / dc
            println(io, @sprintf("    cap %.1f→%.1f MW/h : λ = %+.2f \$/(MW/h)",
                                 rc_results.ramp_cap[i-1], rc_results.ramp_cap[i], lambda))
        end
    end

    return (load_cap=lc_results, ramp_cap=rc_results)
end

# ============================================================================
# Study 5 — Out-of-Sample Robustness Test
# ============================================================================

function study5_out_of_sample(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 5: Out-of-Sample Robustness Test")
    println(io, "="^90)
    println(io, """
    Procedure:
      (1) Solve the base problem with $(length(base_data.scenarios)) in-sample scenarios.
      (2) Fix first-stage decisions (job schedule w_{j,t}, DVFS a_t, battery P_charge/P_discharge).
      (3) Generate 500 fresh out-of-sample scenarios with the same distribution.
      (4) Evaluate interconnection constraint violations (load_cap, ramp_cap) for each
          out-of-sample scenario.
      (5) Report empirical violation rate and worst-case violation magnitude.
    """)

    # Step 1: Solve base
    print("  Solving in-sample problem … ")
    sol, model, _, elapsed = solve_case(base_data; tag="in-sample")
    if sol === nothing
        println(io, "  ERROR: Base case infeasible. Cannot perform out-of-sample test.")
        return nothing
    end
    println("done ($(round(elapsed, digits=1))s)")

    vars = model.ext[:dc_vars]
    Tset = collect(base_data.horizon)
    T = length(Tset)
    ic = base_data.interconnection

    # Step 2: Extract fixed first-stage decisions
    D_vals = [JuMP.value(vars.D[t]) for t in Tset]
    ΔD_vals = [JuMP.value(vars.ΔD[t]) for t in Tset]
    a_vals = [JuMP.value(vars.a[t]) for t in Tset]

    # Also extract battery commitments for per-scenario SOC check
    enable_battery = base_data.bess.charge_max > 0
    P_charge_vals = enable_battery ? [JuMP.value(vars.P_charge[t]) for t in Tset] : zeros(T)
    P_discharge_vals = enable_battery ? [JuMP.value(vars.P_discharge[t]) for t in Tset] : zeros(T)
    P_reserve_vals = enable_battery ? [JuMP.value(vars.P_reserve[t]) for t in Tset] : zeros(T)
    P_frp_up_vals = enable_battery ? [JuMP.value(vars.P_frp_up[t]) for t in Tset] : zeros(T)
    P_frp_down_vals = enable_battery ? [JuMP.value(vars.P_frp_down[t]) for t in Tset] : zeros(T)

    # In-sample statistics
    n_in = length(base_data.scenarios)
    in_load_violations = 0
    in_ramp_violations = 0
    in_max_load_excess = 0.0
    in_max_ramp_excess = 0.0
    for scen in base_data.scenarios
        prev_err = 0.0
        for k in 1:T
            actual_load = D_vals[k] + scen.load_error[k]
            excess_load = actual_load - ic.load_cap
            if excess_load > 1e-4
                in_load_violations += 1
                in_max_load_excess = max(in_max_load_excess, excess_load)
            end
            err_diff = scen.load_error[k] - prev_err
            actual_ramp = ΔD_vals[k] + err_diff
            ramp_excess = max(actual_ramp - ic.ramp_cap, -actual_ramp - ic.ramp_cap, 0.0)
            if ramp_excess > 1e-4
                in_ramp_violations += 1
                in_max_ramp_excess = max(in_max_ramp_excess, ramp_excess)
            end
            prev_err = scen.load_error[k]
        end
    end

    println(io, "  In-sample results ($(n_in) scenarios × $(T) periods):")
    println(io, @sprintf("    Load violations:  %d / %d  (%.1f%%),  max excess = %.3f MW",
                         in_load_violations, n_in*T, in_load_violations/(n_in*T)*100, in_max_load_excess))
    println(io, @sprintf("    Ramp violations:  %d / %d  (%.1f%%),  max excess = %.3f MW/h",
                         in_ramp_violations, n_in*T, in_ramp_violations/(n_in*T)*100, in_max_ramp_excess))

    # Step 3: Generate out-of-sample scenarios
    n_oos = 500
    Random.seed!(12345)  # Different seed from training
    caiso_stats = load_caiso_frp_stats()
    pjm_stats = load_pjm_reserve_stats()
    oos_scenarios = generate_real_deployment_scenarios_vcc(caiso_stats, pjm_stats, T, n_oos)

    # Step 4: Evaluate violations
    print("  Evaluating $(n_oos) out-of-sample scenarios … ")
    oos_load_violations = 0
    oos_ramp_violations = 0
    oos_max_load_excess = 0.0
    oos_max_ramp_excess = 0.0
    oos_load_excess_per_scen = Float64[]
    oos_ramp_excess_per_scen = Float64[]

    # Also track SOC violations
    oos_soc_violations = 0
    oos_max_soc_violation = 0.0

    b = base_data.bess
    for (s_idx, scen) in enumerate(oos_scenarios)
        max_le = 0.0
        max_re = 0.0
        prev_err = 0.0
        E_prev = b.capacity * b.soc_init

        for k in 1:T
            # Load constraint check
            actual_load = D_vals[k] + scen.load_error[k]
            excess_load = actual_load - ic.load_cap
            if excess_load > 1e-4
                oos_load_violations += 1
                oos_max_load_excess = max(oos_max_load_excess, excess_load)
                max_le = max(max_le, excess_load)
            end

            # Ramp constraint check
            err_diff = scen.load_error[k] - prev_err
            actual_ramp = ΔD_vals[k] + err_diff
            ramp_excess = max(actual_ramp - ic.ramp_cap, -actual_ramp - ic.ramp_cap, 0.0)
            if ramp_excess > 1e-4
                oos_ramp_violations += 1
                oos_max_ramp_excess = max(oos_max_ramp_excess, ramp_excess)
                max_re = max(max_re, ramp_excess)
            end

            # SOC constraint check (if battery)
            if enable_battery
                charge_deployed = P_charge_vals[k] + scen.beta_frp_up[k] * P_frp_up_vals[k]
                discharge_deployed = P_discharge_vals[k] + scen.beta_reserve[k] * P_reserve_vals[k] +
                                     scen.beta_frp_down[k] * P_frp_down_vals[k]
                inflow = (charge_deployed * b.eta_charge - discharge_deployed / b.eta_discharge) * base_data.delta_t
                E_curr = E_prev + inflow
                soc = E_curr / b.capacity
                soc_viol = max(b.soc_min - soc, soc - b.soc_max, 0.0)
                if soc_viol > 1e-4
                    oos_soc_violations += 1
                    oos_max_soc_violation = max(oos_max_soc_violation, soc_viol)
                end
                E_prev = E_curr
            end

            prev_err = scen.load_error[k]
        end
        push!(oos_load_excess_per_scen, max_le)
        push!(oos_ramp_excess_per_scen, max_re)
    end
    println("done")

    total_checks = n_oos * T
    println(io, "\n  Out-of-sample results ($(n_oos) scenarios × $(T) periods):")
    println(io, @sprintf("    Load violations:  %d / %d  (%.2f%%),  max excess = %.3f MW",
                         oos_load_violations, total_checks, oos_load_violations/total_checks*100, oos_max_load_excess))
    println(io, @sprintf("    Ramp violations:  %d / %d  (%.2f%%),  max excess = %.3f MW/h",
                         oos_ramp_violations, total_checks, oos_ramp_violations/total_checks*100, oos_max_ramp_excess))
    if enable_battery
        println(io, @sprintf("    SOC violations:   %d / %d  (%.2f%%),  max = %.4f (fraction of capacity)",
                             oos_soc_violations, total_checks, oos_soc_violations/total_checks*100, oos_max_soc_violation))
    end

    # Per-scenario summary
    scen_with_load_viol = count(x -> x > 1e-4, oos_load_excess_per_scen)
    scen_with_ramp_viol = count(x -> x > 1e-4, oos_ramp_excess_per_scen)
    println(io, @sprintf("\n    Scenarios with ≥1 load violation: %d / %d (%.1f%%)",
                         scen_with_load_viol, n_oos, scen_with_load_viol/n_oos*100))
    println(io, @sprintf("    Scenarios with ≥1 ramp violation: %d / %d (%.1f%%)",
                         scen_with_ramp_viol, n_oos, scen_with_ramp_viol/n_oos*100))

    # Violation statistics
    load_excesses = filter(x -> x > 1e-4, oos_load_excess_per_scen)
    ramp_excesses = filter(x -> x > 1e-4, oos_ramp_excess_per_scen)
    if !isempty(load_excesses)
        println(io, @sprintf("    Load excess (among violating): mean=%.3f MW, p95=%.3f MW, max=%.3f MW",
                             mean(load_excesses), quantile(load_excesses, 0.95), maximum(load_excesses)))
    end
    if !isempty(ramp_excesses)
        println(io, @sprintf("    Ramp excess (among violating): mean=%.3f MW/h, p95=%.3f MW/h, max=%.3f MW/h",
                             mean(ramp_excesses), quantile(ramp_excesses, 0.95), maximum(ramp_excesses)))
    end

    # Comparison with in-sample risk tolerance
    println(io, "\n  Risk tolerance comparison:")
    in_load_rate = in_load_violations / (n_in * T)
    oos_load_rate = oos_load_violations / total_checks
    println(io, @sprintf("    Load: in-sample=%.2f%%  out-of-sample=%.2f%%  tolerance=%.1f%%",
                         in_load_rate*100, oos_load_rate*100, ic.delta_load_risk*100))
    in_ramp_rate = in_ramp_violations / (n_in * T)
    oos_ramp_rate = oos_ramp_violations / total_checks
    println(io, @sprintf("    Ramp: in-sample=%.2f%%  out-of-sample=%.2f%%  tolerance=%.1f%%",
                         in_ramp_rate*100, oos_ramp_rate*100, ic.delta_ramp_risk*100))

    # Verdict
    load_ok = oos_load_rate <= ic.delta_load_risk * 1.5  # Allow 50% margin
    ramp_ok = oos_ramp_rate <= ic.delta_ramp_risk * 1.5
    if load_ok && ramp_ok
        println(io, "\n  ✓ Out-of-sample violation rates are within acceptable margins")
    else
        if !load_ok
            println(io, "\n  ⚠  Load violation rate ($(round(oos_load_rate*100, digits=2))%) exceeds 1.5× tolerance ($(round(ic.delta_load_risk*150, digits=1))%)")
        end
        if !ramp_ok
            println(io, "  ⚠  Ramp violation rate ($(round(oos_ramp_rate*100, digits=2))%) exceeds 1.5× tolerance ($(round(ic.delta_ramp_risk*150, digits=1))%)")
        end
    end

    return Dict(:in_sample => (load_violations=in_load_violations, ramp_violations=in_ramp_violations,
                               max_load_excess=in_max_load_excess, max_ramp_excess=in_max_ramp_excess),
                :out_of_sample => (load_violations=oos_load_violations, ramp_violations=oos_ramp_violations,
                                   max_load_excess=oos_max_load_excess, max_ramp_excess=oos_max_ramp_excess,
                                   soc_violations=oos_soc_violations, max_soc_violation=oos_max_soc_violation,
                                   scen_with_load_viol=scen_with_load_viol,
                                   scen_with_ramp_viol=scen_with_ramp_viol))
end

# ============================================================================
# Study 1b — Fixed vs Schedulable Load Sensitivity
# ============================================================================

"""
    rebuild_with_base_load_scale(base_data, scale)

Rebuild ProblemDataVCC with base_load scaled by `scale`, adjusting base_load_upper
accordingly. This changes the headroom available for schedulable jobs.
"""
function rebuild_with_base_load_scale(base_data::ProblemDataVCC, scale::Float64)
    scaled_load = base_data.base_load .* scale
    vcc = base_data.vcc
    scaled_upper = [scaled_load[i] - vcc.hardware_limit * vcc.epsilon_min
                    for i in 1:length(scaled_load)]
    ProblemDataVCC(base_data.horizon, base_data.delta_t, scaled_load,
                   scaled_upper, base_data.jobs, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

"""
    rebuild_with_job_scale(base_data, work_scale, rate_scale)

Rebuild ProblemDataVCC with job work and rate_max scaled by the given factors.
"""
function rebuild_with_job_scale(base_data::ProblemDataVCC, work_scale::Float64, rate_scale::Float64)
    scaled_jobs = [Job(j.name, j.release, j.deadline,
                       j.work * work_scale, j.rate_max * rate_scale,
                       j.tardiness_weight)
                   for j in base_data.jobs]
    ProblemDataVCC(base_data.horizon, base_data.delta_t, base_data.base_load,
                   base_data.base_load_upper, scaled_jobs, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

function study1b_fixed_schedulable_sensitivity(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 1b: Fixed Load vs Schedulable Load Sensitivity")
    println(io, "="^90)

    headroom_base = base_data.vcc.hardware_limit - maximum(base_data.base_load_upper)
    sum_rate_max = sum(j.rate_max for j in base_data.jobs)
    println(io, """
    This study varies the ratio of fixed (non-schedulable) to schedulable (job)
    load, which is the most important practical dimension for DC flexibility.

    Approach A: Scale base_load (fixed load) up/down — changes headroom.
    Approach B: Scale job work + rate requirements — changes schedulable demand.

    Base case:
      mean(base_load) = $(round(mean(base_data.base_load), digits=1)) MW
      max(base_load_upper) = $(round(maximum(base_data.base_load_upper), digits=1)) MW
      headroom = P̄^DC - max(base_load_upper) = $(round(headroom_base, digits=1)) MW
      Σ rate_max = $(round(sum_rate_max, digits=1)) MW
    """)

    # --- Part A: Fixed Load Scaling ---
    println(io, "  Part A: Base Load (Fixed Load) Scaling")
    println(io, "  ─────────────────────────────────────────")
    bl_scales = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    bl_results = DataFrame(
        scale=Float64[], mean_base_load=Float64[], headroom=Float64[],
        sched_fraction=Float64[],
        objective=Float64[], worst_revenue=Float64[],
        job_completion=Float64[], mean_dvfs=Float64[],
        total_load_slack=Float64[], solve_time=Float64[])

    for sc in bl_scales
        print("  base_load × $(sc) … ")
        d = rebuild_with_base_load_scale(base_data, sc)
        hr = d.vcc.hardware_limit - maximum(d.base_load_upper)
        sf = hr / d.vcc.hardware_limit  # schedulable fraction of total capacity
        sol, _, _, elapsed = solve_case(d; tag="bl_scale=$sc")
        if sol !== nothing
            mean_a = mean(sol[:vcc].a_factor)
            jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            push!(bl_results, (sc, mean(d.base_load), hr, sf,
                               sol[:objective_value], sol[:worst_case_revenue],
                               jc, mean_a,
                               sol[:slack_violations][:total_load_slack], elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2)),  headroom=$(round(hr, digits=1)) MW")
        else
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("  %-6s  %8s  %9s  %8s  %12s  %12s  %7s  %8s  %9s  %5s",
                         "Scale", "BaseLD", "Headroom", "SchedFr", "Obj(\$)", "WrstRev(\$)",
                         "Job(%)", "mean(a)", "LdSlack", "Time"))
    println(io, "  " * "-"^105)
    for r in eachrow(bl_results)
        println(io, @sprintf("  %-6.2f  %8.1f  %9.1f  %7.1f%%  %12.2f  %12.2f  %6.1f%%  %8.4f  %9.4f  %4.1fs",
                             r.scale, r.mean_base_load, r.headroom, r.sched_fraction*100,
                             r.objective, r.worst_revenue, r.job_completion*100,
                             r.mean_dvfs, r.total_load_slack, r.solve_time))
    end

    # Marginal analysis
    if nrow(bl_results) >= 2
        println(io, "\n  Marginal Δ(objective) / Δ(headroom):")
        for i in 2:nrow(bl_results)
            dhr = bl_results.headroom[i] - bl_results.headroom[i-1]
            dobj = bl_results.objective[i] - bl_results.objective[i-1]
            abs(dhr) < 1e-6 && continue
            println(io, @sprintf("    headroom %.1f→%.1f MW: λ = %+.2f \$/MW-headroom",
                                 bl_results.headroom[i-1], bl_results.headroom[i], dobj/dhr))
        end
    end

    # --- Part B: Job (Schedulable Load) Scaling ---
    println(io, "\n  Part B: Job (Schedulable Load) Scaling")
    println(io, "  ───────────────────────────────────────")
    job_scales = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
    job_results = DataFrame(
        work_scale=Float64[], total_work=Float64[], sum_rate_max=Float64[],
        objective=Float64[], worst_revenue=Float64[],
        job_completion=Float64[], mean_dvfs=Float64[],
        total_load_slack=Float64[], total_hw_slack=Float64[],
        solve_time=Float64[])

    for js in job_scales
        print("  job_scale × $(js) … ")
        d = rebuild_with_job_scale(base_data, js, min(js, 2.0))  # cap rate scale at 2×
        tw = sum(j.work for j in d.jobs)
        sr = sum(j.rate_max for j in d.jobs)
        sol, _, _, elapsed = solve_case(d; tag="job_scale=$js")
        if sol !== nothing
            mean_a = mean(sol[:vcc].a_factor)
            jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            push!(job_results, (js, tw, sr,
                                sol[:objective_value], sol[:worst_case_revenue],
                                jc, mean_a,
                                sol[:slack_violations][:total_load_slack],
                                sol[:slack_violations][:total_hardware_slack], elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2)),  job_comp=$(round(jc*100, digits=1))%")
        else
            println("INFEASIBLE")
        end
    end

    println(io, @sprintf("\n  %-8s  %10s  %9s  %12s  %12s  %7s  %8s  %9s  %9s  %5s",
                         "JobScale", "TotalWork", "ΣRateMax", "Obj(\$)", "WrstRev(\$)",
                         "Job(%)", "mean(a)", "LdSlack", "HWSlack", "Time"))
    println(io, "  " * "-"^110)
    for r in eachrow(job_results)
        println(io, @sprintf("  %-8.2f  %10.1f  %9.1f  %12.2f  %12.2f  %6.1f%%  %8.4f  %9.4f  %9.4f  %4.1fs",
                             r.work_scale, r.total_work, r.sum_rate_max,
                             r.objective, r.worst_revenue, r.job_completion*100,
                             r.mean_dvfs, r.total_load_slack, r.total_hw_slack, r.solve_time))
    end

    return (base_load_scaling=bl_results, job_scaling=job_results)
end

# ============================================================================
# Study 4b — Direct Dual Variable Extraction (LP Relaxation)
# ============================================================================

"""
    extract_lp_duals(model, data)

Fix all binary/integer variables at their MILP-optimal values, re-solve as LP,
and extract dual variables (shadow prices) for the interconnection constraints.

Returns a Dict with per-period and aggregate shadow prices for load_cap and
ramp_cap. The returned duals are already in the objective units because the
scenario weights enter the LP objective; therefore, they should be summed over
scenarios without applying the weights a second time. Works with both HiGHS
and Gurobi.
"""
function extract_lp_duals(model::Model, data::ProblemDataVCC)
    # Ensure we have stored constraint references
    if !haskey(model.ext, :ic_load_constraints)
        return nothing
    end

    Tset = collect(data.horizon)
    T = length(Tset)
    Sset = 1:length(data.scenarios)

    # Fix all binary/integer variables to MILP solution values.
    # IMPORTANT: extract ALL values first, because modifying any variable
    #            invalidates JuMP's solution cache (OptimizeNotCalled error).
    var_vals = Dict{VariableRef, Float64}()
    for var in all_variables(model)
        if is_binary(var) || is_integer(var)
            var_vals[var] = JuMP.value(var)
        end
    end

    n_fixed = length(var_vals)
    for (var, val) in var_vals
        if is_binary(var)
            unset_binary(var)
        end
        if is_integer(var)
            unset_integer(var)
        end
        fix(var, round(val); force=true)
    end

    # Re-solve as LP
    optimize!(model)
    lp_status = termination_status(model)

    if lp_status != OPTIMAL && lp_status != LOCALLY_SOLVED
        return Dict(:status => lp_status, :n_fixed => n_fixed)
    end

    lp_obj = objective_value(model)

    # Read duals from stored interconnection constraint references
    ic_load_cons = model.ext[:ic_load_constraints]
    ic_ramp_up_cons = model.ext[:ic_ramp_up_constraints]
    ic_ramp_down_cons = model.ext[:ic_ramp_down_constraints]

    # Per-period aggregate shadow prices summed across scenarios.
    # Convention: for a ≤ constraint, dual ≤ 0 in max problem.
    # The shadow price λ = -dual gives the value of relaxing the RHS by 1 unit.
    # Do not multiply by scenario weights here: JuMP duals already reflect the
    # weighted objective coefficients in the LP.
    lambda_load_t = Float64[]
    lambda_ramp_up_t = Float64[]
    lambda_ramp_down_t = Float64[]

    for t in Tset
        lam_load = sum(-dual(ic_load_cons[(s, t)]) for s in Sset)
        lam_ramp_up = sum(-dual(ic_ramp_up_cons[(s, t)]) for s in Sset)
        lam_ramp_down = sum(-dual(ic_ramp_down_cons[(s, t)]) for s in Sset)
        push!(lambda_load_t, lam_load)
        push!(lambda_ramp_up_t, lam_ramp_up)
        push!(lambda_ramp_down_t, lam_ramp_down)
    end

    return Dict(
        :status => lp_status,
        :lp_objective => lp_obj,
        :n_fixed => n_fixed,
        :lambda_load_per_period => lambda_load_t,
        :lambda_ramp_up_per_period => lambda_ramp_up_t,
        :lambda_ramp_down_per_period => lambda_ramp_down_t,
        :total_lambda_load => sum(lambda_load_t),
        :total_lambda_ramp_up => sum(lambda_ramp_up_t),
        :total_lambda_ramp_down => sum(lambda_ramp_down_t),
    )
end

function study4b_direct_dual_extraction(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 4b: Direct Dual Variable Extraction (LP Relaxation)")
    println(io, "="^90)
    println(io, """
    Methodology:
      1. Solve the MILP to optimality (discrete DVFS + binary battery modes).
      2. Fix all binary/integer variables at their MILP-optimal values.
      3. Re-solve the resulting LP relaxation.
      4. Read dual variables (shadow prices) λ for interconnection constraints:
           D_t + ε_t ≤ load_cap + slack_load  →  λ_load(s,t)
           ΔD_t + Δε_t ≤ ramp_cap + slack_ramp  →  λ_ramp(s,t)
      5. Aggregate: λ_load(t) = Σ_s w_s · λ_load(s,t)
    Solver: $(DCFlex.solver_name())

    Comparison with Study 4 finite-difference approach validates consistency.
    """)

    # --- 4b-a: Base case duals ---
    println("  4b-a: Base case LP dual extraction …")
    print("  Solving MILP (base case) … ")
    sol, model, _, elapsed_milp = solve_case(base_data; tag="base-dual")
    if sol === nothing
        println(io, "  ERROR: Base case infeasible")
        return nothing
    end
    milp_obj = sol[:objective_value]
    println("done ($(round(elapsed_milp, digits=1))s), obj=\$$(round(milp_obj, digits=2))")

    print("  Extracting LP duals (fixing $(length(collect(all_variables(model)))) vars) … ")
    t0 = time()
    duals_base = extract_lp_duals(model, base_data)
    elapsed_lp = time() - t0
    println("done ($(round(elapsed_lp, digits=1))s)")

    if duals_base === nothing || !haskey(duals_base, :lambda_load_per_period)
        println(io, "  ERROR: Could not extract LP duals (status=$(get(duals_base, :status, "N/A")))")
        return nothing
    end

    println(io, "  Base Case LP Dual Results:")
    println(io, @sprintf("    MILP objective: \$%.2f", milp_obj))
    println(io, @sprintf("    LP objective:   \$%.2f  (gap: \$%.2f)", duals_base[:lp_objective],
                         duals_base[:lp_objective] - milp_obj))
    println(io, @sprintf("    Binary/integer vars fixed: %d", duals_base[:n_fixed]))
    println(io)

    Tset = collect(base_data.horizon)
    T = length(Tset)
    println(io, @sprintf("  %-6s  %12s  %12s  %12s",
                         "Hour", "λ_load(\$/MW)", "λ_ramp_up", "λ_ramp_down"))
    println(io, "  " * "-"^50)
    for (i, t) in enumerate(Tset)
        println(io, @sprintf("  %-6d  %12.4f  %12.4f  %12.4f",
                             t, duals_base[:lambda_load_per_period][i],
                             duals_base[:lambda_ramp_up_per_period][i],
                             duals_base[:lambda_ramp_down_per_period][i]))
    end
    println(io, @sprintf("\n  Totals: λ_load=%.4f  λ_ramp_up=%.4f  λ_ramp_down=%.4f",
                         duals_base[:total_lambda_load],
                         duals_base[:total_lambda_ramp_up],
                         duals_base[:total_lambda_ramp_down]))

    # --- 4b-b: Sweep load_cap and compare finite-diff vs LP duals ---
    println(io, "\n  Comparison: Finite-Difference λ vs LP Dual λ for load_cap sweep")
    println(io, "  ──────────────────────────────────────────────────────────────")

    load_caps = [95.0, 100.0, 103.0, 105.0, 107.0, 110.0]
    fd_results = DataFrame(load_cap=Float64[], milp_obj=Float64[],
                          lp_obj=Float64[], total_lambda_load_lp=Float64[])

    for lc in load_caps
        print("  load_cap=$lc … ")
        d = rebuild_with_interconnection(base_data; load_cap=lc)
        sol_lc, model_lc, _, el = solve_case(d; tag="lc=$lc-dual")
        if sol_lc !== nothing
            mi_obj = sol_lc[:objective_value]
            duals_lc = extract_lp_duals(model_lc, d)
            if duals_lc !== nothing && haskey(duals_lc, :total_lambda_load)
                push!(fd_results, (lc, mi_obj, duals_lc[:lp_objective], duals_lc[:total_lambda_load]))
                println("MILP=\$$(round(mi_obj, digits=2)), LP_λ_load=$(round(duals_lc[:total_lambda_load], digits=2))")
            else
                println("LP duals failed")
            end
        else
            println("INFEASIBLE")
        end
    end

    if nrow(fd_results) >= 2
        println(io, @sprintf("\n  %-12s  %12s  %12s  %14s  %14s",
                             "load_cap(MW)", "MILP Obj(\$)", "FD λ(\$/MW)", "LP λ_load(sum)", "LP Obj(\$)"))
        println(io, "  " * "-"^75)
        for i in 1:nrow(fd_results)
            fd_lambda = if i < nrow(fd_results)
                dc = fd_results.load_cap[i+1] - fd_results.load_cap[i]
                dobj = fd_results.milp_obj[i+1] - fd_results.milp_obj[i]
                abs(dc) < 1e-6 ? 0.0 : dobj / dc
            else
                NaN
            end
            fd_str = isnan(fd_lambda) ? "    —" : @sprintf("%14.2f", fd_lambda)
            println(io, @sprintf("  %-12.1f  %12.2f  %14s  %14.4f  %12.2f",
                                 fd_results.load_cap[i], fd_results.milp_obj[i],
                                 fd_str, fd_results.total_lambda_load_lp[i],
                                 fd_results.lp_obj[i]))
        end
    end

    # --- 4b-c: Sweep ramp_cap similarly ---
    println(io, "\n  Comparison: Finite-Difference λ vs LP Dual λ for ramp_cap sweep")
    println(io, "  ──────────────────────────────────────────────────────────────")

    ramp_caps = [5.0, 10.0, 13.0, 15.0, 20.0, 35.0]
    rd_results = DataFrame(ramp_cap=Float64[], milp_obj=Float64[],
                          lp_obj=Float64[], total_lambda_ramp_up_lp=Float64[],
                          total_lambda_ramp_down_lp=Float64[])

    for rc in ramp_caps
        print("  ramp_cap=$rc … ")
        d = rebuild_with_interconnection(base_data; ramp_cap=rc)
        sol_rc, model_rc, _, el = solve_case(d; tag="rc=$rc-dual")
        if sol_rc !== nothing
            mi_obj = sol_rc[:objective_value]
            duals_rc = extract_lp_duals(model_rc, d)
            if duals_rc !== nothing && haskey(duals_rc, :total_lambda_ramp_up)
                push!(rd_results, (rc, mi_obj, duals_rc[:lp_objective],
                                   duals_rc[:total_lambda_ramp_up], duals_rc[:total_lambda_ramp_down]))
                println("MILP=\$$(round(mi_obj, digits=2)), LP_λ_ramp=$(round(duals_rc[:total_lambda_ramp_up] + duals_rc[:total_lambda_ramp_down], digits=2))")
            else
                println("LP duals failed")
            end
        else
            println("INFEASIBLE")
        end
    end

    if nrow(rd_results) >= 2
        println(io, @sprintf("\n  %-14s  %12s  %12s  %14s  %14s  %12s",
                             "ramp_cap(MW/h)", "MILP Obj(\$)", "FD λ(\$/(MW/h))",
                             "LP λ_ramp_up", "LP λ_ramp_dn", "LP Obj(\$)"))
        println(io, "  " * "-"^90)
        for i in 1:nrow(rd_results)
            fd_lambda = if i < nrow(rd_results)
                dc = rd_results.ramp_cap[i+1] - rd_results.ramp_cap[i]
                dobj = rd_results.milp_obj[i+1] - rd_results.milp_obj[i]
                abs(dc) < 1e-6 ? 0.0 : dobj / dc
            else
                NaN
            end
            fd_str = isnan(fd_lambda) ? "     —" : @sprintf("%12.2f", fd_lambda)
            println(io, @sprintf("  %-14.1f  %12.2f  %12s  %14.4f  %14.4f  %12.2f",
                                 rd_results.ramp_cap[i], rd_results.milp_obj[i],
                                 fd_str, rd_results.total_lambda_ramp_up_lp[i],
                                 rd_results.total_lambda_ramp_down_lp[i],
                                 rd_results.lp_obj[i]))
        end
    end

    println(io, "\n  Note: LP duals are extracted after fixing MILP integer variables.")
    println(io, "  FD λ = Δ(MILP obj)/Δ(param); LP λ = shadow price from LP relaxation at fixed integers.")
    println(io, "  Consistency between the two validates the dual-based approach.")

    return Dict(:base_duals => duals_base,
                :load_cap_sweep => fd_results,
                :ramp_cap_sweep => rd_results)
end

# ============================================================================
# Study 6 — DVFS Level Ablation
# ============================================================================

"""Build, solve, and summarise a VCC instance with custom DVFS levels."""
function solve_case_with_dvfs(data::ProblemDataVCC;
                               enable_battery::Bool=true,
                               tag::String="",
                               quiet::Bool=true,
                               dvfs_levels::Union{Nothing, Vector{Float64}}=nothing,
                               n_dvfs_levels::Int=7)
    kwargs = Dict{Symbol, Any}(:enable_battery => enable_battery)
    if dvfs_levels !== nothing
        kwargs[:dvfs_levels] = dvfs_levels
    else
        kwargs[:n_dvfs_levels] = n_dvfs_levels
    end
    model = build_model_vcc(data; kwargs...)
    set_silent(model)
    set_time_limit_sec(model, 300.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)
    t0 = time()
    optimize!(model)
    elapsed = time() - t0
    status = termination_status(model)
    if !quiet
        println("  [$tag] status=$status  time=$(round(elapsed, digits=1))s")
    end
    if has_values(model)
        sol = solution_summary_vcc(model)
        return sol, model, data, elapsed
    else
        return nothing, model, data, elapsed
    end
end

function study6_dvfs_level_ablation(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 6: DVFS Level Ablation")
    println(io, "="^90)
    println(io, """
    Compares system performance with different DVFS frequency scaling ranges.
    Most data centers operate at fixed frequency (no DVFS). This study quantifies
    the value of enabling DVFS and how the benefit scales with the allowed range.

    Configurations (all use uniform 5% step within the range):
      - No DVFS:  a_t = 1.0 (fixed at reference frequency)  — 1 level
      - ±5%:      a_t ∈ {0.95, 1.00, 1.05}                  — 3 levels
      - ±10%:     a_t ∈ {0.90, 0.95, 1.00, 1.05, 1.10}      — 5 levels
      - ±15%:     a_t ∈ {0.85, …, 1.15}                      — 7 levels
      - ±20%:     a_t ∈ {0.80, …, 1.20}                      — 9 levels

    Base case: a_ref = $(base_data.cost.a_ref), a_min = $(base_data.cost.a_min), a_max = $(base_data.cost.a_max)
    DVFS deviation cost: c₁ = $(base_data.cost.c1) \$/(unit deviation per period)

    Literature on DVFS and system stability/reliability:
    ──────────────────────────────────────────────────────
    DVFS introduces several categories of reliability/stability concerns:

    1. **Soft error susceptibility**: Lower supply voltage (undervolting) increases
       the soft error rate (SER) because reduced noise margins make logic more
       vulnerable to particle strikes. Agiakatsikas et al. (MICRO 2023) measured a
       2.6× increase in SER when voltage drops from nominal to ~85% on server-grade
       Xeon CPUs. Cao et al. (IEEE TVLSI 2022) model soft error probability as
       exponential in voltage reduction: λ_soft ∝ exp(−V/V_ref).

    2. **Frequency transition overhead**: Switching DVFS states incurs a PLL
       stabilization delay (typically 10–100 μs). Gendler et al. (IEEE Micro 2021)
       propose instantaneous frequency switching (I-DVFS) to reduce this to ~1 μs,
       but conventional DVFS still requires voltage/frequency guard bands during
       transitions, limiting how rapidly the system can respond.

    3. **Thermal cycling and electromigration**: Aggressive overclocking (boosting
       above nominal) accelerates wear-out failure mechanisms. Piga et al. (SOSP 2024)
       report that Meta deploys DVFS boosting at data center scale but limits the
       per-server "boost budget" to maintain a target failure rate; sustained operation
       at 115–120% frequency increases the daily hardware failure rate by ~15–30%.

    4. **Power delivery network (PDN) stability**: Rapid load transients from DVFS
       switching can cause voltage droops, potentially triggering machine-check
       exceptions. Chen et al. (IEEE Trans. Industry Applications 2025) design
       distributed power management to mitigate DVFS-induced droop.

    5. **Practical deployment experience**: Krzywda et al. (FGCS 2018) report that
       in production data centers DVFS "rarely reduces energy consumption" due to
       race-to-idle effects. Kuehn & Mashaly (WoNS 2019) show DVFS is most
       beneficial at intermediate load levels (40–70% utilization). Hou et al.
       (IEEE TSG 2023) note that fine-grained DVFS helps edge data centers
       participate in grid services while maintaining stability.

    Our formulation accounts for these effects through:
      - The DVFS deviation cost c₁|a_t − a_o| penalizing departure from nominal
      - Discrete levels preventing continuous fine-grained switching
      - The φ parameter limiting what fraction of load responds to DVFS

    References:
      [1] Agiakatsikas et al., "Impact of voltage scaling on soft errors
          susceptibility of multicore server CPUs," MICRO 2023.
      [2] Cao et al., "Energy and reliability-aware task scheduling for cost
          optimization of DVFS-enabled cloud workflows," IEEE TVLSI, 2022.
      [3] Gendler et al., "I-DVFS: Instantaneous frequency switch during dynamic
          voltage and frequency scaling," IEEE Micro, 2021.
      [4] Piga et al., "Expanding datacenter capacity with DVFS boosting: A safe
          and scalable deployment experience," SOSP 2024.
      [5] Krzywda et al., "Power-performance tradeoffs in data center servers:
          DVFS, CPU pinning, horizontal, and vertical scaling," FGCS, 2018.
      [6] Kuehn & Mashaly, "DVFS-power management and performance engineering of
          data center server clusters," WoNS 2019.
      [7] Hou et al., "Fine-grained online energy management of edge data centers
          using per-core power gating and DVFS," IEEE TSG, 2023.
      [8] Chen & Chen, "An all-digital distributed power management architecture
          for DVFS for multi-core processor," IEEE Trans. Industry App., 2025.
    """)

    configs = [
        ("No DVFS",       [1.0]),
        ("±5%  (3 lvl)",  collect(0.95:0.05:1.05)),
        ("±10% (5 lvl)",  collect(0.90:0.05:1.10)),
        ("±15% (7 lvl)",  collect(0.85:0.05:1.15)),
        ("±20% (9 lvl)",  collect(0.80:0.05:1.20)),
    ]

    results = DataFrame(config=String[], n_levels=Int[], a_range=String[],
                        objective=Float64[], worst_revenue=Float64[],
                        job_completion=Float64[], mean_dvfs=Float64[],
                        min_dvfs=Float64[], max_dvfs=Float64[],
                        total_load_slack=Float64[], total_hw_slack=Float64[],
                        as_revenue=Float64[], dvfs_penalty=Float64[],
                        solve_time=Float64[])

    for (name, levels) in configs
        print("  $name … ")
        sol, _, _, elapsed = solve_case_with_dvfs(base_data; tag=name, dvfs_levels=levels)
        if sol !== nothing
            a_factors = sol[:vcc].a_factor
            mean_a = mean(a_factors)
            min_a = minimum(a_factors)
            max_a = maximum(a_factors)
            jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
            as_rev = sol[:enable_battery] ? sum(
                base_data.pricing.lambda_reserve[i] * sol[:bess][i+1, :reserve] +
                base_data.pricing.lambda_frp_up[i] * sol[:bess][i+1, :frp_up] +
                base_data.pricing.lambda_frp_down[i] * sol[:bess][i+1, :frp_down]
                for i in 1:length(base_data.horizon)) : 0.0
            dvfs_pen = base_data.cost.c1 * sum(abs(a - base_data.cost.a_ref) for a in a_factors)
            n_lvl = length(levels)
            a_range_str = @sprintf("[%.2f, %.2f]", minimum(levels), maximum(levels))
            push!(results, (name, n_lvl, a_range_str,
                            sol[:objective_value], sol[:worst_case_revenue],
                            jc, mean_a, min_a, max_a,
                            sol[:slack_violations][:total_load_slack],
                            sol[:slack_violations][:total_hardware_slack],
                            as_rev, dvfs_pen, elapsed))
            println("obj=\$$(round(sol[:objective_value], digits=2)),  mean(a)=$(round(mean_a, digits=3))")
        else
            println("INFEASIBLE")
        end
    end

    # Print table
    println(io, @sprintf("  %-16s %4s  %-14s  %12s  %12s  %7s  %8s  %6s  %6s  %10s  %9s  %10s  %10s  %6s",
                         "Config", "Lvl", "Range", "Obj(\$)", "WrstRev(\$)", "Job(%)", "mean(a)",
                         "min(a)", "max(a)", "LdSlack", "HWSlack", "AS Rev(\$)", "DVFSPen(\$)", "Time"))
    println(io, "  " * "-"^150)
    for r in eachrow(results)
        println(io, @sprintf("  %-16s %4d  %-14s  %12.2f  %12.2f  %6.1f%%  %8.4f  %6.3f  %6.3f  %10.4f  %9.4f  %10.2f  %10.2f  %5.1fs",
                             r.config, r.n_levels, r.a_range,
                             r.objective, r.worst_revenue, r.job_completion*100,
                             r.mean_dvfs, r.min_dvfs, r.max_dvfs,
                             r.total_load_slack, r.total_hw_slack,
                             r.as_revenue, r.dvfs_penalty, r.solve_time))
    end

    # Value of DVFS (relative to no DVFS)
    if nrow(results) >= 2
        no_dvfs_obj = results[results.config .== "No DVFS", :objective]
        if length(no_dvfs_obj) > 0
            base_obj = no_dvfs_obj[1]
            println(io, @sprintf("\n  Value of DVFS (vs No DVFS baseline at \$%.2f):", base_obj))
            for r in eachrow(results)
                r.config == "No DVFS" && continue
                delta = r.objective - base_obj
                pct = base_obj != 0.0 ? delta / abs(base_obj) * 100 : 0.0
                println(io, @sprintf("    %-16s: Δobj = %+.2f  (%+.2f%%)",
                                     r.config, delta, pct))
            end
        end
    end

    # Marginal analysis of range expansion
    if nrow(results) >= 2
        println(io, "\n  Marginal value of DVFS range expansion:")
        for i in 2:nrow(results)
            dobj = results.objective[i] - results.objective[i-1]
            drev = results.worst_revenue[i] - results.worst_revenue[i-1]
            dpen = results.dvfs_penalty[i] - results.dvfs_penalty[i-1]
            println(io, @sprintf("    %s → %s: Δobj = %+.2f,  Δrev = %+.2f,  Δpenalty = %+.2f",
                                 results.config[i-1], results.config[i], dobj, drev, dpen))
        end
    end

    return results
end

# ============================================================================
# Study 7 — Load Composition Sensitivity (Fixed Total Load)
# ============================================================================

"""
    rebuild_with_load_composition(base_data, total_avg_load, fixed_fraction)

Rebuild ProblemDataVCC with a specified total average load split into fixed
(base_load) and schedulable (job) components according to `fixed_fraction`.

- `total_avg_load`:  target mean(base_load) + mean(job_power) [MW]
- `fixed_fraction`:  fraction of total_avg_load that is fixed (base_load)
"""
function rebuild_with_load_composition(base_data::ProblemDataVCC,
                                       total_avg_load::Float64,
                                       fixed_fraction::Float64)
    T = length(base_data.horizon)
    base_mean = mean(base_data.base_load)
    base_avg_job = sum(j.work for j in base_data.jobs) / T

    # Target fixed and schedulable components
    target_fixed = total_avg_load * fixed_fraction
    target_sched = total_avg_load * (1.0 - fixed_fraction)

    # Scale factors
    bl_scale = target_fixed / base_mean
    job_work_scale = target_sched / base_avg_job
    job_rate_scale = min(job_work_scale, 3.0)  # cap rate scale to avoid infeasibility

    # Rebuild base load
    scaled_load = base_data.base_load .* bl_scale
    vcc = base_data.vcc
    scaled_upper = [scaled_load[i] - vcc.hardware_limit * vcc.epsilon_min
                    for i in 1:length(scaled_load)]

    # Rebuild jobs
    scaled_jobs = [Job(j.name, j.release, j.deadline,
                       j.work * job_work_scale, j.rate_max * job_rate_scale,
                       j.tardiness_weight)
                   for j in base_data.jobs]

    ProblemDataVCC(base_data.horizon, base_data.delta_t, scaled_load,
                   scaled_upper, scaled_jobs, base_data.bess,
                   base_data.pricing, base_data.cost, base_data.interconnection,
                   base_data.scenarios, base_data.vcc)
end

function study7_load_composition_sensitivity(base_data::ProblemDataVCC, io::IO)
    println(io, "\n" * "="^90)
    println(io, "STUDY 7: Load Composition Sensitivity (Fixed Total Load)")
    println(io, "="^90)

    T = length(base_data.horizon)
    base_mean = mean(base_data.base_load)
    base_avg_job = sum(j.work for j in base_data.jobs) / T
    total_base = base_mean + base_avg_job
    base_fixed_frac = base_mean / total_base
    headroom_base = base_data.vcc.hardware_limit - maximum(base_data.base_load_upper)

    println(io, """
    Studies 1b-A and 1b-B varied fixed and schedulable load independently. This
    study holds the total average load constant and sweeps the ratio between fixed
    (always-on services) and schedulable (batch jobs, ML training) components.

    This answers: "If our data center uses X MW on average, what mix of fixed vs
    schedulable load maximizes economic value?"

    Base case:
      mean(base_load) = $(round(base_mean, digits=1)) MW (fixed component)
      mean(job_power) = $(round(base_avg_job, digits=1)) MW (avg schedulable power when 100% scheduled)
      total average   = $(round(total_base, digits=1)) MW
      fixed fraction  = $(round(base_fixed_frac*100, digits=1))%
      headroom (P̄^DC − max P̄^fix) = $(round(headroom_base, digits=1)) MW

    Design: Sweep fixed_fraction ∈ {60%, 70%, 75%, 80%, 85%, 90%, 95%} at three
    total load levels:
      - Light:  $(round(total_base * 0.94, digits=1)) MW (~94% of base)
      - Base:   $(round(total_base, digits=1)) MW (base level)
      - Heavy:  $(round(total_base * 1.05, digits=1)) MW (~105% of base)
    """)

    total_levels = [
        ("Light (~94%)", total_base * 0.94),
        ("Base",         total_base),
        ("Heavy (~105%)", total_base * 1.05),
    ]
    fixed_fractions = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    all_results = DataFrame(
        total_level=String[], total_load=Float64[],
        fixed_frac=Float64[], mean_base=Float64[], avg_job_target=Float64[],
        headroom=Float64[],
        objective=Float64[], worst_revenue=Float64[],
        job_completion=Float64[], mean_dvfs=Float64[],
        total_load_slack=Float64[], total_hw_slack=Float64[],
        solve_time=Float64[])

    for (level_name, total_load) in total_levels
        println(io, "  ── Total Load: $level_name = $(round(total_load, digits=1)) MW ──")
        println(io)

        for ff in fixed_fractions
            tag_str = "$(level_name)_ff=$(round(ff*100))%"
            print("  $tag_str … ")
            d = rebuild_with_load_composition(base_data, total_load, ff)
            hr = d.vcc.hardware_limit - maximum(d.base_load_upper)
            avg_job_target = total_load * (1.0 - ff)
            sol, _, _, elapsed = solve_case(d; tag=tag_str)
            if sol !== nothing
                mean_a = mean(sol[:vcc].a_factor)
                jc = 1.0 - sum(sol[:jobs].unfinished) / sum(sol[:jobs].work)
                push!(all_results, (level_name, total_load, ff,
                                    mean(d.base_load), avg_job_target, hr,
                                    sol[:objective_value], sol[:worst_case_revenue],
                                    jc, mean_a,
                                    sol[:slack_violations][:total_load_slack],
                                    sol[:slack_violations][:total_hardware_slack], elapsed))
                println("obj=\$$(round(sol[:objective_value], digits=2)),  job=$(round(jc*100, digits=1))%,  hr=$(round(hr, digits=1)) MW")
            else
                println("INFEASIBLE")
            end
        end
        println(io)
    end

    # Print consolidated table
    println(io, @sprintf("  %-12s  %8s  %7s  %8s  %8s  %9s  %12s  %12s  %7s  %8s  %9s  %9s  %6s",
                         "Level", "Total", "Fix(%)", "FixMW", "SchedMW", "Headroom",
                         "Obj(\$)", "WrstRev(\$)", "Job(%)", "mean(a)",
                         "LdSlack", "HWSlack", "Time"))
    println(io, "  " * "-"^145)
    for r in eachrow(all_results)
        println(io, @sprintf("  %-12s  %8.1f  %6.0f%%  %8.1f  %8.1f  %9.1f  %12.2f  %12.2f  %6.1f%%  %8.4f  %9.4f  %9.4f  %5.1fs",
                             r.total_level, r.total_load, r.fixed_frac*100,
                             r.mean_base, r.avg_job_target, r.headroom,
                             r.objective, r.worst_revenue, r.job_completion*100,
                             r.mean_dvfs, r.total_load_slack, r.total_hw_slack,
                             r.solve_time))
    end

    # Analysis: optimal fixed fraction per total load level
    println(io, "\n  Optimal fixed fraction per total load level:")
    for (level_name, _) in total_levels
        rows = all_results[all_results.total_level .== level_name, :]
        if nrow(rows) > 0
            best_idx = argmax(rows.objective)
            best = rows[best_idx, :]
            println(io, @sprintf("    %-16s: best f = %.0f%%  (obj = \$%.2f, job = %.1f%%, headroom = %.1f MW)",
                                 level_name, best.fixed_frac*100, best.objective,
                                 best.job_completion*100, best.headroom))
        end
    end

    # Marginal analysis within each level
    for (level_name, _) in total_levels
        rows = all_results[all_results.total_level .== level_name, :]
        if nrow(rows) >= 2
            println(io, "\n  Marginal Δ(objective) / Δ(fixed_fraction) for $level_name:")
            for i in 2:nrow(rows)
                df = rows.fixed_frac[i] - rows.fixed_frac[i-1]
                dobj = rows.objective[i] - rows.objective[i-1]
                abs(df) < 1e-6 && continue
                println(io, @sprintf("    f %.0f%%→%.0f%%: Δobj = %+.2f  (%.1f \$/pp)",
                                     rows.fixed_frac[i-1]*100, rows.fixed_frac[i]*100,
                                     dobj, dobj / (df * 100)))
            end
        end
    end

    return all_results
end
