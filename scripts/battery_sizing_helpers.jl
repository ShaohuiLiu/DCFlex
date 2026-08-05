"""
Reusable battery-sizing helpers for the 100 MW data-center benchmark.

The functions define commercial BESS configurations and solve one configuration
against the calibrated CAISO/PJM case. `cluster_sizing_runner.jl` is the public
multiday entry point.

Study Objective:
- Base case: VCC formulation with real market data (real_market_data_case_vcc.jl)
- Sensitivity parameter: Battery size (power rating and energy capacity)
- Metrics analyzed:
  * Economics: Objective value, revenue components, costs
  * Scheduling: Job completion rates, tardiness, flexibility usage
  * Constraint violations: Load and ramp slack variables
  * VCC-specific: DVFS scaling, hardware capacity usage

Battery Configurations Tested:
1. No Battery: System without BESS (baseline)
2. Tesla Megapack 2XL: 1.9 MW, 3.9 MWh per unit, 92.0% RTE
   - Tested: 1-20 units (1.9-38.0 MW, 3.9-78.0 MWh)
3. Tesla Megapack 3: 1.25 MW, 5 MWh per unit, 91.0% RTE
   - Tested: 1-20 units (1.25-25.0 MW, 5-100 MWh)

Data Sources:
- Market data: CAISO FRP (Nov 2025), PJM Reserve (Nov 2025)
- Load data: hourly_aggregated_utilization.csv (scaled dynamically)
- Base configuration: real_market_data_case_vcc.jl

"""

using DCFlex
using JuMP
using Printf
using Statistics
import MathOptInterface as MOI

# Include the VCC formulation real market data generation functions
include(joinpath(@__DIR__, "real_market_data_case_vcc.jl"))

# ============================================================================
# Battery Configuration Definitions
# ============================================================================

"""
    BatteryConfig

Structure to define a battery configuration for sensitivity analysis.
"""
struct BatteryConfig
    name::String
    description::String
    enable_battery::Bool
    num_units::Int
    power_per_unit::Float64  # MW
    energy_per_unit::Float64  # MWh
    charge_efficiency::Float64
    discharge_efficiency::Float64

    function BatteryConfig(name, desc, enable, units, power, energy, eff_c, eff_d)
        new(name, desc, enable, units, power, energy, eff_c, eff_d)
    end
end

function get_total_power(config::BatteryConfig)
    return config.enable_battery ? config.num_units * config.power_per_unit : 0.0
end

function get_total_energy(config::BatteryConfig)
    return config.enable_battery ? config.num_units * config.energy_per_unit : 0.0
end

function get_rte(config::BatteryConfig)
    return config.charge_efficiency * config.discharge_efficiency
end

"""
    generate_battery_configurations()

Generate all battery configurations to test in the sensitivity analysis.
"""
function generate_battery_configurations()
    configs = BatteryConfig[]

    # Configuration 0: No battery (baseline)
    push!(configs, BatteryConfig(
        "No_Battery",
        "No battery - system baseline",
        false,
        0, 0.0, 0.0, 0.0, 0.0
    ))

    # Tesla Megapack 2XL specifications
    # 1.9 MW power, 3.9 MWh energy, 92.0% RTE
    mp2xl_power = 1.9
    mp2xl_energy = 3.9
    mp2xl_rte = 0.92
    mp2xl_eff = sqrt(mp2xl_rte)  # Assume symmetric charge/discharge efficiency

    for n_units in 1:20
        push!(configs, BatteryConfig(
            "Megapack_2XL_$(n_units)x",
            "Tesla Megapack 2XL × $n_units units",
            true,
            n_units, mp2xl_power, mp2xl_energy, mp2xl_eff, mp2xl_eff
        ))
    end

    # Tesla Megapack 3 specifications
    # 1.25 MW power, 5 MWh energy, 91.0% RTE
    mp3_power = 1.25
    mp3_energy = 5.0
    mp3_rte = 0.91
    mp3_eff = sqrt(mp3_rte)

    for n_units in 1:20
        push!(configs, BatteryConfig(
            "Megapack_3_$(n_units)x",
            "Tesla Megapack 3 × $n_units units",
            true,
            n_units, mp3_power, mp3_energy, mp3_eff, mp3_eff
        ))
    end

    return configs
end

# ============================================================================
# Results Data Structure
# ============================================================================

"""
    SensitivityResult

Structure to store results for one battery configuration.
"""
mutable struct SensitivityResult
    config_name::String
    config_description::String
    enable_battery::Bool
    total_power_mw::Float64
    total_energy_mwh::Float64
    round_trip_efficiency::Float64

    # Optimization status
    termination_status::String
    solve_time_seconds::Float64

    # Economics
    objective_value::Float64
    energy_cost::Float64
    reserve_revenue::Float64
    frp_up_revenue::Float64
    frp_down_revenue::Float64
    total_as_revenue::Float64
    job_penalty_cost::Float64

    # Job scheduling
    total_late_work::Float64
    total_unfinished_work::Float64
    job_completion_rate::Float64

    # Battery usage (if enabled)
    battery_cycles_base::Float64
    battery_cycles_expected::Float64
    total_energy_charged_mwh::Float64
    total_energy_discharged_mwh::Float64

    # Constraint violations
    max_load_slack_mw::Float64
    total_load_slack_mw::Float64
    max_ramp_slack_mw_per_h::Float64
    total_ramp_slack_mw_per_h::Float64

    function SensitivityResult(config::BatteryConfig)
        new(config.name, config.description, config.enable_battery,
            get_total_power(config), get_total_energy(config), get_rte(config),
            "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0)
    end
end

# ============================================================================
# Simulation Runner
# ============================================================================

"""
    run_sensitivity_case(config::BatteryConfig, seed::Int, T::Int; n_cyc_max_override=nothing)

Run optimization for a single battery configuration and return results along with model and data.
Optional `n_cyc_max_override` replaces the default cycling constraint (0.5 cycles/day).
"""
function run_sensitivity_case(config::BatteryConfig, seed::Int=42, T::Int=24; day_index::Int=0, quiet::Bool=false, n_cyc_max_override::Union{Nothing,Float64}=nothing, enforce_terminal_soc::Bool=false, terminal_soc_frac::Float64=0.0)
    if !quiet
        println("\n" * "="^80)
        println("Testing: $(config.description)")
        println("="^80)
    end

    # Generate base problem data using VCC formulation
    data_base = generate_real_market_case_vcc(seed=seed, T=T, use_real_load=true, day_index=day_index, quiet=quiet)

    # Modify BESS parameters based on configuration
    if config.enable_battery
        total_power = get_total_power(config)
        total_energy = get_total_energy(config)

        # Create new BESS parameters with updated values (struct is immutable)
        actual_n_cyc_max = isnothing(n_cyc_max_override) ? data_base.bess.n_cyc_max : n_cyc_max_override
        bess_modified = BESSParameters(
            total_energy,                    # capacity
            data_base.bess.soc_init,        # soc_init
            data_base.bess.soc_min,         # soc_min
            data_base.bess.soc_max,         # soc_max
            total_power,                     # charge_max
            total_power,                     # discharge_max
            config.charge_efficiency,        # eta_charge
            config.discharge_efficiency,     # eta_discharge
            data_base.bess.c_degradation,   # c_degradation
            actual_n_cyc_max                 # n_cyc_max
        )

        # Create new ProblemDataVCC with modified BESS (struct is immutable)
        data = ProblemDataVCC(
            data_base.horizon,
            data_base.delta_t,
            data_base.base_load,
            data_base.base_load_upper,
            data_base.jobs,
            bess_modified,                   # Use modified BESS
            data_base.pricing,
            data_base.cost,
            data_base.interconnection,
            data_base.scenarios,
            data_base.vcc                    # Include VCC parameters
        )

        if !quiet
            println("  Battery: $(total_power) MW / $(total_energy) MWh (RTE: $(round(get_rte(config)*100, digits=1))%)")
            println("  Configuration: $(config.num_units) × $(config.power_per_unit) MW / $(config.energy_per_unit) MWh")
        end
    else
        # Use base data as-is for no-battery case
        data = data_base
        if !quiet
            println("  Battery: Disabled (baseline case)")
        end
    end

    # Build and solve VCC model
    if !quiet; println("  Building VCC model..."); end
    model = build_model_vcc(data, enable_battery=config.enable_battery,
                            enforce_terminal_soc=enforce_terminal_soc,
                            terminal_soc_frac=terminal_soc_frac)
    set_silent(model)
    set_time_limit_sec(model, 300.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)

    if !quiet; println("  Solving..."); end
    t_start = time()
    optimize!(model)
    solve_time = time() - t_start

    # Create result structure
    result = SensitivityResult(config)
    result.termination_status = string(termination_status(model))
    result.solve_time_seconds = solve_time

    if !quiet
        println("  Status: $(result.termination_status)")
        println("  Solve time: $(round(solve_time, digits=2))s")
    end

    if has_values(model)
        # Extract objective value
        result.objective_value = objective_value(model)
        if !quiet; println("  Objective: \$$(round(result.objective_value, digits=2))"); end

        # Extract economic metrics
        for t in data.horizon
            result.energy_cost += data.pricing.lambda_energy[t] * value(model[:D][t]) * data.delta_t
        end

        if config.enable_battery
            for t in data.horizon
                result.reserve_revenue += data.pricing.lambda_reserve[t] * value(model[:P_reserve][t]) * data.delta_t
                result.frp_up_revenue += data.pricing.lambda_frp_up[t] * value(model[:P_frp_up][t]) * data.delta_t
                result.frp_down_revenue += data.pricing.lambda_frp_down[t] * value(model[:P_frp_down][t]) * data.delta_t
            end
            result.total_as_revenue = result.reserve_revenue + result.frp_up_revenue + result.frp_down_revenue
        end

        # Extract job scheduling metrics
        for job in data.jobs
            j_idx = findfirst(x -> x.name == job.name, data.jobs)
            result.total_unfinished_work += value(model[:ℓ][j_idx])
        end

        # Calculate job completion rate
        total_work_required = sum(job.work for job in data.jobs)
        work_completed = total_work_required - result.total_unfinished_work
        result.job_completion_rate = work_completed / total_work_required

        # Extract battery usage metrics (if enabled)
        if config.enable_battery
            for t in data.horizon
                result.total_energy_charged_mwh += value(model[:P_charge][t]) * data.delta_t
                result.total_energy_discharged_mwh += value(model[:P_discharge][t]) * data.delta_t
            end
            result.battery_cycles_base = result.total_energy_discharged_mwh / total_energy

            # Expected equivalent full cycles use the same scenario-weighted
            # throughput convention as the cycle-limit constraint.
            expected_throughput = 0.0
            Tset = collect(data.horizon)
            for scen in data.scenarios
                scen_throughput = 0.0
                for (k, t) in enumerate(Tset)
                    charge_deployed = value(model[:P_charge][t]) + scen.beta_frp_up[k] * value(model[:P_frp_up][t])
                    discharge_deployed = value(model[:P_discharge][t]) +
                                         scen.beta_reserve[k] * value(model[:P_reserve][t]) +
                                         scen.beta_frp_down[k] * value(model[:P_frp_down][t])
                    scen_throughput += (charge_deployed + discharge_deployed) * data.delta_t
                end
                expected_throughput += scen.weight * scen_throughput
            end
            result.battery_cycles_expected = expected_throughput / (2 * total_energy)
        end

        # Extract constraint violations
        for s in 1:length(data.scenarios), t in data.horizon
            load_slack_val = value(model[:load_slack][s, t])
            ramp_up_val = value(model[:ramp_slack_up][s, t])
            ramp_down_val = value(model[:ramp_slack_down][s, t])

            result.max_load_slack_mw = max(result.max_load_slack_mw, load_slack_val)
            result.total_load_slack_mw += load_slack_val * data.scenarios[s].weight

            ramp_slack = max(ramp_up_val, ramp_down_val)
            result.max_ramp_slack_mw_per_h = max(result.max_ramp_slack_mw_per_h, ramp_slack)
            result.total_ramp_slack_mw_per_h += ramp_slack * data.scenarios[s].weight
        end

        if !quiet
            println("  Metrics:")
            println("    - AS Revenue: \$$(round(result.total_as_revenue, digits=2))")
            println("    - Job Completion: $(round(result.job_completion_rate * 100, digits=1))%")
            if config.enable_battery
                println("    - Battery Cycles: $(round(result.battery_cycles_base, digits=3)) base, $(round(result.battery_cycles_expected, digits=3)) expected")
            end
            println("    - Load Slack: $(round(result.max_load_slack_mw, digits=3)) MW max")
        end
    else
        if !quiet; println("  ⚠️  No solution found!"); end
        return result, nothing, nothing
    end

    return result, model, data
end
