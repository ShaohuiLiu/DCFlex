import Base: length

"""
    Job

Describes a schedulable workload in the data center following V1 formulation.
Each job j ∈ J has execution rate w_{j,t} and tardiness ℓ_j variables.

Fields
- `name` : unique identifier used in reporting.
- `release` : release time s_j (first time period when the job can start, 1-based index).
- `deadline` : deadline d_j (last time period when the job should be finished).
- `work` : total work requirement W_j (in compatible units with Δt).
- `rate_max` : maximum executable rate r_j^{max} per time period.
- `tardiness_weight` : weight κ_j used in the objective for deadline shortfall ℓ_j.
"""
struct Job
    name::String
    release::Int
    deadline::Int
    work::Float64
    rate_max::Float64
    tardiness_weight::Float64
end

"""
    BESSParameters

Technical description of the co-located battery system.
The BESS can operate in mutually exclusive charging/discharging modes with
directional capacity constraints for co-provision of energy and ancillary services.

Fields
- `capacity` : energy capacity E^{cap} [MWh].
- `soc_init` : initial state of charge fraction in [0,1].
- `soc_min`  : minimum allowed SOC fraction.
- `soc_max`  : maximum allowed SOC fraction.
- `charge_max` : maximum charging power P_{α}^{max} [MW] for energy + FRP up.
- `discharge_max` : maximum discharging power P_{β}^{max} [MW] for energy + reserve + FRP down.
- `eta_charge` : charging efficiency η_{α}.
- `eta_discharge` : discharging efficiency η_{β}.
- `c_degradation` : degradation cost c^{deg} [\$/MWh] representing amortized battery replacement cost.
- `n_cyc_max` : preferred daily cycle limit N^{cyc}_{max} (e.g., 0.5 cycles/day) for warranty compliance.
"""
struct BESSParameters
    capacity::Float64
    soc_init::Float64
    soc_min::Float64
    soc_max::Float64
    charge_max::Float64
    discharge_max::Float64
    eta_charge::Float64
    eta_discharge::Float64
    c_degradation::Float64
    n_cyc_max::Float64
end

# Provide backward-compatible constructor that sets default degradation parameters
function BESSParameters(capacity, soc_init, soc_min, soc_max, charge_max, discharge_max, eta_charge, eta_discharge)
    # Default degradation cost: ~$50/MWh based on typical Li-ion battery replacement costs
    # Default cycle limit: 0.5 cycles/day (typical warranty condition)
    BESSParameters(capacity, soc_init, soc_min, soc_max, charge_max, discharge_max,
                   eta_charge, eta_discharge, 50.0, 0.5)
end

"""
    PricingData

Holds market price time series and the risk-aversion weight applied to the
robust revenue component.
"""
struct PricingData
    lambda_energy::Vector{Float64}
    lambda_reserve::Vector{Float64}
    lambda_frp_up::Vector{Float64}
    lambda_frp_down::Vector{Float64}
    lambda_weight::Float64
end

"""
    CostParameters

Coefficients appearing in the flexibility cost function f = c3∑_{j,t>d_j}κ_jv_{j,t} + c2∑κ_jℓ_j + c1∑|a_t−a_o|
and operating bounds for power-intensity scaling.

Fields
- `c1` : coefficient for power intensity deviation penalty |a_t−a_o|.
- `c2` : coefficient for unfinished job portion penalty κ_jℓ_j (much larger, reflects complete job loss).
- `c3` : coefficient for late execution penalty κ_jv_{j,t} for t>d_j (reasonably large, reflects realistic conditions).
- `a_ref` : reference power intensity a_o.
- `a_min` : minimum allowable power intensity (underbar{a}).
- `a_max` : maximum allowable power intensity (overbar{a}).
"""
struct CostParameters
    c1::Float64
    c2::Float64
    c3::Float64
    a_ref::Float64
    a_min::Float64
    a_max::Float64
end

"""
    InterconnectionLimits

Encodes capacity and ramping limits together with risk tolerances for the
chance constraints.
"""
struct InterconnectionLimits
    load_cap::Float64
    ramp_cap::Float64
    delta_load_risk::Float64
    delta_ramp_risk::Float64
    initial_net_load::Float64
    big_m_load::Float64
    big_m_ramp::Float64
end

"""
    Scenario

Represents a realization ξ ∈ Ξ of uncertain reserve/FRP deployment factors
β(ξ) and load forecast errors following V1 formulation. The uncertainty
affects actual utilization in reserve and FRP services.
All vectors must have length equal to the number of time periods.
"""
struct Scenario
    name::String
    beta_reserve::Vector{Float64}
    beta_frp_up::Vector{Float64}
    beta_frp_down::Vector{Float64}
    load_error::Vector{Float64}
    weight::Float64
end

"""
    ProblemData

Aggregates all inputs necessary to build the scenario-based robust JuMP model.
The model optimizes
co-located data center and BESS operation under grid interconnection requirements.
"""
struct ProblemData
    horizon::UnitRange{Int}
    delta_t::Float64
    base_load::Vector{Float64}
    jobs::Vector{Job}
    bess::BESSParameters
    pricing::PricingData
    cost::CostParameters
    interconnection::InterconnectionLimits
    scenarios::Vector{Scenario}
end

"""
    nperiods(data)

Return the number of time steps in the planning horizon.
"""
nperiods(data::ProblemData) = length(data.horizon)

function _check_length(vec::AbstractVector, T::Int, label::AbstractString)
    length(vec) == T || error("$label must have length $T, found $(length(vec))")
end

"""
    validate!(data)

Perform a set of consistency checks on the problem data. Throws an error if any
requirement is violated.
"""
function validate!(data::ProblemData)
    T = nperiods(data)
    _check_length(data.base_load, T, "base_load")
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
    sum(s.weight for s in data.scenarios) > 0 || error("Scenario weights must sum to a positive value")

    for (idx, job) in enumerate(data.jobs)
        job.release >= first(data.horizon) || error("Job $(job.name) release earlier than horizon start")
        job.deadline <= last(data.horizon) || error("Job $(job.name) deadline exceeds horizon end")
        job.release <= job.deadline || error("Job $(job.name) has release > deadline")
        job.work >= 0 || error("Job $(job.name) has negative work requirement")
        job.rate_max >= 0 || error("Job $(job.name) has negative rate upper bound")
    end

    b = data.bess
    0.0 <= b.soc_init <= 1.0 || error("Initial SOC must be in [0,1]")
    0.0 <= b.soc_min <= b.soc_max <= 1.0 || error("SOC bounds inconsistent")
    b.capacity > 0 || error("BESS capacity must be positive")
    b.charge_max >= 0 || error("BESS charging power must be non-negative")
    b.discharge_max >= 0 || error("BESS discharging power must be non-negative")
    b.eta_charge > 0 || error("Charging efficiency must be positive")
    b.eta_discharge > 0 || error("Discharging efficiency must be positive")
    b.c_degradation >= 0 || error("Degradation cost must be non-negative")
    b.n_cyc_max > 0 || error("Maximum cycle limit must be positive")

    ic = data.interconnection
    ic.big_m_load > 0 || error("big_m_load must be positive")
    ic.big_m_ramp > 0 || error("big_m_ramp must be positive")
    ic.load_cap >= 0 || error("load_cap must be non-negative")
    ic.ramp_cap >= 0 || error("ramp_cap must be non-negative")
    0 <= ic.delta_load_risk <= 1 || error("delta_load_risk must lie in [0,1]")
    0 <= ic.delta_ramp_risk <= 1 || error("delta_ramp_risk must lie in [0,1]")

    c = data.cost
    c.a_min <= c.a_ref <= c.a_max || error("a_ref must lie within [a_min, a_max]")
    println("Data validation finished. If no errors were thrown, the data is consistent.")
end

length(data::ProblemData) = nperiods(data)
