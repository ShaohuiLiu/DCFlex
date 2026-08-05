"""
100 MW Data Center with Real Market Data - VCC Formulation

This example demonstrates the revised VCC (Virtual Capacity Curve) formulation
using actual market data from CAISO FRP and PJM Reserve. The key changes from
the original formulation are:

1. **Hardware Limit (P̄^DC)**: Physical facility capacity constraint (100 MW)
2. **DVFS Sensitivity (φ)**: Fraction of non-schedulable load that scales with DVFS
3. **Forecast Error Bounds**: Upper bound P̄^fix_t accounting for uncertainty
4. **Capacity Headroom**: Jobs scheduled within remaining capacity after fixed load

The VCC formulation ensures:
- Flexible jobs scheduled within available headroom: Σ_j p_{j,t} + P̄^fix_t ≤ P̄^DC
- Power balance respects DVFS scaling: P̂_t = P^fix_t(a_t) + Σ_j p_{j,t}
- Robust against forecast errors in non-schedulable load

Data Sources:
- CAISO FRP: November 2025 RTD data (8,645 intervals, 5-min resolution)
- PJM Reserve: November 2025 dispatch reserves
- Load Data: hourly_aggregated_utilization.csv (scaled to ~85 MW base)

Reference: the robust formulation described in the associated paper.
"""

using DCFlex
using Random
using Statistics

# ============================================================================
# Market Data Loading Functions (reuse from real_market_data_case.jl)
# ============================================================================

"""
    load_caiso_frp_stats()

Load CAISO FRP statistics from the analysis results.
"""
function load_caiso_frp_stats()
    frp_up = Dict(:mean => 279.7, :std => 85.0, :min => 65.2, :max => 665.0, :median => 260.0)
    frp_down = Dict(:mean => 305.0, :std => 95.0, :min => 71.3, :max => 875.9, :median => 280.0)
    shortfall = Dict(:frp_up_rate => 0.018, :frp_down_rate => 0.015,
                    :frp_up_avg_deficit => 2.7, :frp_down_avg_deficit => 2.6)
    binding = Dict(:up_instances => 165, :down_instances => 554,
                  :up_rate => 165/593, :down_rate => 554/593)

    hourly_frp_up_factor = [0.7, 0.65, 0.6, 0.6, 0.7, 0.85, 1.1, 1.3, 1.4, 1.3, 1.1, 1.0,
                           0.95, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3, 1.1, 0.95, 0.85, 0.75, 0.7]
    hourly_frp_down_factor = [0.8, 0.85, 0.9, 0.95, 0.9, 0.8, 0.7, 0.75, 0.85, 1.0, 1.2, 1.35,
                             1.4, 1.3, 1.2, 1.0, 0.9, 0.85, 0.8, 0.85, 0.9, 0.9, 0.85, 0.8]

    return Dict(:frp_up => frp_up, :frp_down => frp_down, :shortfall => shortfall,
                :binding => binding, :hourly_frp_up_factor => hourly_frp_up_factor,
                :hourly_frp_down_factor => hourly_frp_down_factor)
end

"""
    load_pjm_reserve_stats()

Load PJM Reserve statistics.
"""
function load_pjm_reserve_stats()
    primary = Dict(:mean => 3066.0, :requirement_mean => 2833.0,
                   :reliability_component => 2643.0, :extended_component => 190.0)
    synchronized = Dict(:mean => 1952.0, :requirement_mean => 1952.0,
                       :reliability_component => 1762.0, :extended_component => 190.0)
    thirty_min = Dict(:mean => 22176.0, :requirement_mean => 3190.0,
                     :reliability_component => 3000.0, :extended_component => 190.0)

    return Dict(:primary => primary, :synchronized => synchronized,
                :thirty_min => thirty_min, :deficit_rate => 0.0)
end

"""
    generate_real_deployment_scenarios_vcc(caiso_stats, pjm_stats, T, n_scenarios)

Generate scenarios with deployment factors based on real CAISO/PJM patterns.
"""
function generate_real_deployment_scenarios_vcc(caiso_stats, pjm_stats, T, n_scenarios)
    scenarios = Scenario[]

    binding_up_rate = caiso_stats[:binding][:up_rate]
    binding_down_rate = caiso_stats[:binding][:down_rate]
    hourly_up_factor = caiso_stats[:hourly_frp_up_factor]
    hourly_down_factor = caiso_stats[:hourly_frp_down_factor]

    for s in 1:n_scenarios
        # Reserve deployment
        beta_reserve = Float64[]
        for t in 1:T
            hour = mod(t - 1, 24)
            base_deploy = if 14 <= hour <= 20
                0.7 + 0.2 * rand()
            elseif 6 <= hour <= 22
                0.5 + 0.2 * rand()
            else
                0.3 + 0.2 * rand()
            end
            push!(beta_reserve, clamp(base_deploy + 0.1 * randn(), 0.0, 1.0))
        end

        # FRP Up deployment
        beta_frp_up = Float64[]
        for t in 1:T
            hour = mod(t - 1, 24)
            hour_factor = hourly_up_factor[hour + 1]
            deploy_prob = binding_up_rate * hour_factor
            deploy_amount = if rand() < deploy_prob
                0.6 + 0.35 * rand()
            else
                0.05 + 0.15 * rand()
            end
            push!(beta_frp_up, clamp(deploy_amount + 0.1 * randn(), 0.0, 1.0))
        end

        # FRP Down deployment
        beta_frp_down = Float64[]
        for t in 1:T
            hour = mod(t - 1, 24)
            hour_factor = hourly_down_factor[hour + 1]
            deploy_prob = binding_down_rate * hour_factor
            deploy_amount = if rand() < deploy_prob
                0.5 + 0.4 * rand()
            else
                0.1 + 0.2 * rand()
            end
            push!(beta_frp_down, clamp(deploy_amount + 0.1 * randn(), 0.0, 1.0))
        end

        # Load forecast errors: ±2-3% at 100 MW scale
        load_error = 2.5 .* randn(T)

        push!(scenarios, Scenario("scenario_$s", beta_reserve, beta_frp_up,
                                   beta_frp_down, load_error, 1.0 / n_scenarios))
    end

    return scenarios
end

# ============================================================================
# Generate VCC Case Data
# ============================================================================

"""
    generate_real_market_case_vcc(; kwargs...)

Generate a 100 MW data center case using the VCC formulation with real market data.

# Key VCC Parameters
- `hardware_limit`: P̄^DC = 100 MW (physical facility capacity)
- `phi`: φ = 0.25 (25% of fixed load is DVFS-sensitive, based on CPU power fraction)
- `epsilon_max`: 0.05 (5% overestimation in forecast)
- `epsilon_min`: -0.03 (3% underestimation in forecast)

The formulation ensures flexible jobs are scheduled within the headroom:
  Σ_j p_{j,t} + P̄^fix_t(a_o) ≤ P̄^DC
"""
function generate_real_market_case_vcc(; seed::Integer=42, T::Int=24, use_real_load::Bool=false, day_index::Int=0, quiet::Bool=false)
    Random.seed!(seed)
    horizon = 1:T
    Δt = 1.0

    caiso_stats = load_caiso_frp_stats()
    pjm_stats = load_pjm_reserve_stats()

    if !quiet
        println("="^80)
        println("VCC Formulation: 100 MW Data Center with Real Market Data")
        println("="^80)
        println()
        println("📊 Loaded CAISO FRP Statistics:")
        println("  FRP Up:   mean=$(caiso_stats[:frp_up][:mean]) MW")
        println("  FRP Down: mean=$(caiso_stats[:frp_down][:mean]) MW")
        println("  Binding rate: Up=$(round(caiso_stats[:binding][:up_rate]*100, digits=1))%, Down=$(round(caiso_stats[:binding][:down_rate]*100, digits=1))%")
        println()
    end

    # Base load profile
    base_load = Float64[]

    if use_real_load
        csv_path = joinpath(@__DIR__, "..", "data", "workload", "hourly_aggregated_utilization.csv")

        if isfile(csv_path)
            lines = readlines(csv_path)
            data_lines = lines[2:end]
            real_data = Float64[]
            for line in data_lines
                parts = split(line, ',')
                if length(parts) >= 2
                    util_mw = parse(Float64, parts[2])
                    push!(real_data, util_mw * 6.1)  # Scale by 6.1x to ~80 MW
                end
            end

            # Select 24-hour slice based on day_index
            # day_index=0: hours 0-23 (first day), day_index=1: hours 24-47, etc.
            n_total_hours = length(real_data)
            n_days = n_total_hours ÷ T
            actual_day = mod(day_index, n_days)  # Wrap around if needed
            start_idx = actual_day * T + 1

            for t in horizon
                idx = start_idx + (t - 1)
                if idx <= n_total_hours
                    push!(base_load, real_data[idx])
                else
                    push!(base_load, real_data[mod1(idx, n_total_hours)])
                end
            end

            if !quiet
                println("📊 Using REAL load data from CSV (scaled by 6.1x), Day $(actual_day + 1)/$(n_days):")
                println("  Hours $(start_idx)-$(start_idx + T - 1) of $(n_total_hours)")
                println("  Load range: $(round(minimum(base_load), digits=1)) - $(round(maximum(base_load), digits=1)) MW")
            end
        else
            @warn "Real load data not found, falling back to synthetic"
            use_real_load = false
        end
    end

    if !use_real_load || isempty(base_load)
        if !quiet
            println("📊 Using SYNTHETIC load data (diurnal pattern):")
        end
        base_mean = 85.0  # Lower base to leave headroom for jobs (75 -> 78,80,85 )

        for t in horizon
            hour = mod(t - 1, 24)
            if 8 <= hour <= 18
                business_factor = 1.0 + 0.12 * sin(π * (hour - 8) / 10)
            elseif 6 <= hour < 8 || 18 < hour <= 22
                business_factor = 0.95
            else
                business_factor = 0.85
            end
            noise = randn() * 5#1.5
            push!(base_load, base_mean * business_factor + noise)
        end
        if !quiet
            println("  Load range: $(round(minimum(base_load), digits=1)) - $(round(maximum(base_load), digits=1)) MW")
        end
    end
    if !quiet
        println()
    end

    # Job portfolio - designed to fit within headroom (~15-25 MW available)
    jobs = [
        Job("batch_analytics_1", 1, 8, 30.0, 6.0, 0.8),    # Lower power to fit headroom
        Job("hpc_simulation", 3, 14, 60.0, 10.0, 2.0),
        Job("ml_training_large", 6, 20, 70.0, 9.0, 2.5),
        Job("data_backup", 1, T, 25.0, 5.0, 0.5),
        Job("batch_analytics_2", 12, 20, 30.0, 7.0, 1.0),
        Job("ml_training_small", 10, 18, 20.0, 5.0, 1.5),
        Job("hpc_evening", 16, T, 40.0, 10.0, 1.8),
        Job("data_processing", 1, 12, 35.0, 4.0, 2.2),
        # Add more jobs if needed, but ensure total max power ≤ 25 MW to fit headroom
        Job("preemptable", 1, T, 30.0, 5.0, 0.1),
        # Retained from the published benchmark to preserve deterministic model
        # ordering under the reported 1% MIP stopping tolerance.
        Job("none", 1, T, 0.0, 0.0, 0.0),
    ]

    # BESS parameters
    bess = BESSParameters(
        36.0,   # Capacity: 36 MWh
        0.6,    # Initial SOC: 60%
        0.15,   # Min SOC: 15%
        0.90,   # Max SOC: 90%
        12.0,   # Charge limit: 12 MW
        12.0,   # Discharge limit: 12 MW
        0.95,   # Charge efficiency
        0.95,   # Discharge efficiency
        45.0,   # c_degradation [$/MWh]
        0.5,    # n_cyc_max [cycles/day]
    )

    # Pricing based on real market patterns
    hourly_frp_up = caiso_stats[:hourly_frp_up_factor]
    hourly_frp_down = caiso_stats[:hourly_frp_down_factor]

    energy_prices = Float64[]
    reserve_prices = Float64[]
    frp_up_prices = Float64[]
    frp_down_prices = Float64[]

    for t in horizon
        hour = mod(t - 1, 24)

        base_price = if 0 <= hour < 6 || 22 <= hour < 24
            45.0 + 10.0 * rand()
        elseif (6 <= hour < 10) || (18 <= hour < 22)
            70.0 + 15.0 * rand()
        else
            100.0 + 20.0 * rand()
        end
        push!(energy_prices, base_price)

        reserve_base = 8.0 + 4.0 * (hour >= 14 && hour <= 20 ? 1.0 : 0.5)
        push!(reserve_prices, reserve_base + 2.0 * rand())

        frp_up_base = 12.0 * hourly_frp_up[hour + 1]
        frp_down_base = 8.0 * hourly_frp_down[hour + 1]
        push!(frp_up_prices, frp_up_base + 3.0 * rand())
        push!(frp_down_prices, frp_down_base + 2.0 * rand())
    end

    pricing = PricingData(energy_prices, reserve_prices, frp_up_prices, frp_down_prices, 1.0)
    # Cost parameters: c1 (frequency deviation), c2 (unfinished work - much larger),
    # c3 (late execution - reasonably large), a_ref, a_min, a_max
    # cost = CostParameters(1000000.0, 2000.0, 100.0, 1.0, 0.8, 1.2)
    cost = CostParameters(1000.0, 2000.0, 100.0, 1.0, 0.8, 1.2)

    interconnection = InterconnectionLimits(
        100.0,  # Load capacity (slightly above hardware limit for battery headroom)
        15.0,   # Ramp limit
        0.05,   # Load risk tolerance
        0.05,   # Ramp risk tolerance
        base_load[1], # initial net load
        200.0,  # big M load
        100.0   # big M ramp
    )

    scenarios = generate_real_deployment_scenarios_vcc(caiso_stats, pjm_stats, T, 50)

    # Create base ProblemData
    base_data = ProblemData(horizon, Δt, base_load, jobs, bess, pricing,
                            cost, interconnection, scenarios)

    # VCC parameters
    # - hardware_limit: 100 MW physical facility capacity
    # - phi: 0.25 (CPU accounts for ~25% of server power, which scales with DVFS)
    # - epsilon_max/min: forecast error bounds (±3-5% typical for day-ahead)
    vcc_params = VCCParameters(
        115.0,   # P̄^DC: 100 MW hardware limit
        0.25;    # φ: 25% DVFS sensitivity
        forecast_error_sigma=0.95,
        epsilon_max=0.05,    # 5% overestimate
        epsilon_min=-0.03,   # 3% underestimate
        enable_vcc_variable=false,
        idle_resource_cost=10.0
    )

    # Convert to VCC problem data
    data_vcc = ProblemDataVCC(base_data, vcc_params)

    return data_vcc
end
