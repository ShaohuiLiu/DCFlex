"""
types_vcc.jl - Extended types for the VCC (Virtual Capacity Curve) formulation.

This file defines additional types for the revised problem formulation that
introduces the Virtual Capacity Curve (VCC) concept for co-located data centers
and BESS. The key changes from the original formulation are:

1. Hardware limit P̄^DC (physical facility capacity constraint)
2. DVFS sensitivity fraction φ ∈ (0,1) for non-schedulable loads
3. Upper bound on fixed load P̄^fix_t(a_t) accounting for forecast errors
4. VCC V_t as schedulable capacity envelope (optional)

Reference: the robust formulation described in the associated paper.
"""

"""
    VCCParameters

Parameters for the VCC (Virtual Capacity Curve) formulation following the
revised formulation in Appendix A of the main document.

Fields
- `hardware_limit` : P̄^DC [MW], the physical facility power capacity (hard limit).
- `phi` : φ ∈ (0,1), fraction of non-schedulable load that scales with DVFS.
  Empirical studies suggest φ ≈ 0.25 (CPU ≈ 25% of server power).
- `forecast_error_sigma` : σ, confidence level for forecast error bounds.
  The upper bound P̄^fix_t is computed using the (1+σ)/2 quantile.
- `epsilon_max` : ε^DC_{max}, maximum normalized forecast error (upper quantile).
- `epsilon_min` : ε^DC_{min}, minimum normalized forecast error (lower quantile).
- `enable_vcc_variable` : If true, introduce V_t as an explicit decision variable
  for schedulable capacity envelope. If false, use implicit formulation.
- `idle_resource_cost` : c_V [USD/MW-h], cost of leaving hardware capacity idle
  (opportunity cost penalty).

The forecast error bounds are computed as:
  ε^DC_{min} = F^{-1}((1-σ)/2), ε^DC_{max} = F^{-1}((1+σ)/2)
where F is the CDF of the normalized forecast error distribution.
"""
struct VCCParameters
    hardware_limit::Float64       # P̄^DC [MW]
    phi::Float64                  # φ fraction (0,1)
    forecast_error_sigma::Float64 # σ confidence level
    epsilon_max::Float64          # ε^DC_{max}
    epsilon_min::Float64          # ε^DC_{min}
    enable_vcc_variable::Bool     # Use explicit V_t variable
    idle_resource_cost::Float64   # c_V [USD/MW-h]
end

"""
    VCCParameters(hardware_limit, phi; kwargs...)

Convenience constructor with sensible defaults for VCC parameters.

# Arguments
- `hardware_limit::Float64`: P̄^DC [MW], physical facility capacity
- `phi::Float64=0.25`: DVFS sensitivity fraction (default based on empirical data)

# Keyword Arguments
- `forecast_error_sigma::Float64=0.95`: Confidence level for forecast bounds
- `epsilon_max::Float64=0.05`: Max normalized forecast error (5% overestimate)
- `epsilon_min::Float64=-0.03`: Min normalized forecast error (3% underestimate)
- `enable_vcc_variable::Bool=false`: Use implicit VCC formulation by default
- `idle_resource_cost::Float64=10.0`: USD/MW-h idle penalty
"""
function VCCParameters(hardware_limit::Float64, phi::Float64=0.25;
                       forecast_error_sigma::Float64=0.95,
                       epsilon_max::Float64=0.05,
                       epsilon_min::Float64=-0.03,
                       enable_vcc_variable::Bool=false,
                       idle_resource_cost::Float64=10.0)
    @assert 0.0 < phi < 1.0 "phi must be in (0,1)"
    @assert hardware_limit > 0.0 "hardware_limit must be positive"
    @assert 0.0 < forecast_error_sigma < 1.0 "forecast_error_sigma must be in (0,1)"
    @assert epsilon_max > epsilon_min "epsilon_max must be > epsilon_min"

    VCCParameters(hardware_limit, phi, forecast_error_sigma, epsilon_max, epsilon_min,
                  enable_vcc_variable, idle_resource_cost)
end

"""
    ProblemDataVCC

Aggregates all inputs for the VCC formulation, extending ProblemData with
VCC-specific parameters. This follows the revised formulation in Appendix A.

The key differences from ProblemData are:
1. Includes VCCParameters for hardware limit, DVFS sensitivity, and forecast bounds
2. Power balance uses P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t
3. Capacity constraint: ∑_j p_{j,t} + P̄^fix_t(a_o) ≤ P̄^DC
"""
struct ProblemDataVCC
    horizon::UnitRange{Int}
    delta_t::Float64
    base_load::Vector{Float64}           # P̂⁰_t: forecasted non-schedulable load at a_o
    base_load_upper::Vector{Float64}     # P̄^DC_t: upper confidence bound
    jobs::Vector{Job}
    bess::BESSParameters
    pricing::PricingData
    cost::CostParameters
    interconnection::InterconnectionLimits
    scenarios::Vector{Scenario}
    vcc::VCCParameters
end

"""
    ProblemDataVCC(data::ProblemData, vcc::VCCParameters)

Create VCC problem data from standard problem data by adding VCC parameters
and computing the upper confidence bounds for base load.

The upper bound is computed as:
  P̄^DC_t = P̂⁰_t - P̄^DC * ε^DC_{min}
where ε^DC_{min} < 0 corresponds to underestimation (actual > forecast).
"""
function ProblemDataVCC(data::ProblemData, vcc::VCCParameters)
    # Compute upper confidence bounds for base load
    # P̄^DC_t := P̂⁰_t - P^DC_max * ε^DC_{min}
    # Since ε^DC_{min} < 0 (underestimate), this gives P̄^DC_t > P̂⁰_t
    base_load_upper = [data.base_load[i] - vcc.hardware_limit * vcc.epsilon_min
                       for i in 1:length(data.base_load)]

    ProblemDataVCC(
        data.horizon,
        data.delta_t,
        data.base_load,
        base_load_upper,
        data.jobs,
        data.bess,
        data.pricing,
        data.cost,
        data.interconnection,
        data.scenarios,
        vcc
    )
end

"""
    nperiods(data::ProblemDataVCC)

Return the number of time steps in the planning horizon.
"""
nperiods(data::ProblemDataVCC) = length(data.horizon)

"""
    validate!(data::ProblemDataVCC)

Perform consistency checks on the VCC problem data.
"""
function validate_vcc!(data::ProblemDataVCC)
    T = nperiods(data)
    _check_length(data.base_load, T, "base_load")
    _check_length(data.base_load_upper, T, "base_load_upper")
    _check_length(data.pricing.lambda_energy, T, "pricing.lambda_energy")
    _check_length(data.pricing.lambda_reserve, T, "pricing.lambda_reserve")
    _check_length(data.pricing.lambda_frp_up, T, "pricing.lambda_frp_up")
    _check_length(data.pricing.lambda_frp_down, T, "pricing.lambda_frp_down")

    for (idx, scen) in enumerate(data.scenarios)
        _check_length(scen.beta_reserve, T, "scenario[$idx].beta_reserve")
        _check_length(scen.beta_frp_up, T, "scenario[$idx].beta_frp_up")
        _check_length(scen.beta_frp_down, T, "scenario[$idx].beta_frp_down")
        _check_length(scen.load_error, T, "scenario[$idx].load_error")
        scen.weight >= 0 || error("scenario[$idx] has negative weight")
    end
    sum(s.weight for s in data.scenarios) > 0 || error("Scenario weights must sum to positive")

    for (idx, job) in enumerate(data.jobs)
        job.release >= first(data.horizon) || error("Job $(job.name) release earlier than horizon")
        job.deadline <= last(data.horizon) || error("Job $(job.name) deadline exceeds horizon")
        job.release <= job.deadline || error("Job $(job.name) has release > deadline")
        job.work >= 0 || error("Job $(job.name) has negative work")
        job.rate_max >= 0 || error("Job $(job.name) has negative rate_max")
    end

    b = data.bess
    0.0 <= b.soc_init <= 1.0 || error("Initial SOC must be in [0,1]")
    0.0 <= b.soc_min <= b.soc_max <= 1.0 || error("SOC bounds inconsistent")
    b.capacity > 0 || error("BESS capacity must be positive")
    b.charge_max >= 0 || error("BESS charging power must be non-negative")
    b.discharge_max >= 0 || error("BESS discharging power must be non-negative")

    vcc = data.vcc
    vcc.hardware_limit > 0 || error("Hardware limit must be positive")
    0 < vcc.phi < 1 || error("phi must be in (0,1)")

    # Check that hardware limit is consistent with base loads
    max_base_upper = maximum(data.base_load_upper)
    if max_base_upper > vcc.hardware_limit
        @warn "Upper bound of base load ($max_base_upper MW) exceeds hardware limit ($(vcc.hardware_limit) MW)"
    end

end
