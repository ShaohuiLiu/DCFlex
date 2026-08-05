"""
model_vcc.jl - VCC (Virtual Capacity Curve) formulation for co-located DC + BESS.

This file implements the revised problem formulation from Appendix A of the main
document. The key changes from the original formulation are:

1. **Hardware limit**: P̄^DC as a hard physical capacity constraint
2. **DVFS-sensitive fraction**: φ ∈ (0,1) of non-schedulable load scales with a_t
3. **Forecast error bounds**: P̄^fix_t(a_t) accounts for uncertainty in base load
4. **Capacity constraint**: Flexible jobs scheduled within remaining headroom

The revised power balance is:
  P̂_t = P^fix_t(a_t) + Σ_j p_{j,t}

where:
  P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t

and the capacity constraint ensures:
  Σ_j p_{j,t} + P̄^fix_t(a_o) ≤ P̄^DC

Reference: Appendix A "Revised Problem Formulation for Co-located Data Center and BESS"
"""

"""
    build_model_vcc(data::ProblemDataVCC; kwargs...)

Construct the JuMP model for the VCC formulation following the revised problem
formulation in Appendix A. This extends the base model with:

1. Hardware capacity constraint: Σ_j p_{j,t} + P̄^fix_t(a_o) ≤ P̄^DC
2. DVFS-sensitive fixed load: P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t
3. Revised power balance: P̂_t = P^fix_t(a_t) + Σ_j p_{j,t}

# DVFS Handling

The bilinear term p_{j,t} = a_t × w_{j,t} can be handled in two ways:

- `dvfs_mode = :discrete` (default): Discretize a_t into L fixed DVFS frequency
  levels. For each level l, a separate copy w^l_{j,t} of the execution rate is
  created. The power becomes p_{j,t} = Σ_l a^l · w^l_{j,t}, which is exact and
  linear. This avoids any relaxation gap at the cost of additional binary variables.

- `dvfs_mode = :mccormick`: Use McCormick envelope relaxation of the bilinear
  term. This is a convex relaxation that may produce solutions infeasible in the
  original problem without iterative tightening.

# Keyword Arguments
- `dvfs_mode::Symbol=:discrete`: DVFS handling method (`:discrete` or `:mccormick`)
- `n_dvfs_levels::Int=7`: Number of discrete DVFS levels (only used when `dvfs_mode=:discrete`)
- `dvfs_levels::Union{Nothing,Vector{Float64}}=nothing`: Custom DVFS level values.
  If `nothing`, levels are evenly spaced between a_min and a_max (with a_ref
  guaranteed to be included).

# Battery LP Relaxation
- `relax_battery_binary::Bool=false`: When `true`, the BESS charge/discharge mode
  indicators `α_t, β_t` are modeled as continuous variables in `[0,1]` instead of
  binaries. This yields the standard LP relaxation of the charge/discharge
  complementarity (`α_t + β_t ≤ 1`). Combined with `dvfs_mode=:mccormick`, the
  whole model becomes a linear program. The relaxation may permit partial
  simultaneous charge and discharge (e.g. stacking FRP-up against reserve), which
  is unphysical and should be checked against the MILP solution.

# DVFS LP Relaxation (direct relaxation of the discrete formulation)
- `relax_dvfs_binary::Bool=false`: Only valid with `dvfs_mode=:discrete`. When
  `true`, the level-selection indicators `z_{l,t}` are modeled as continuous
  variables in `[0,1]` instead of binaries, i.e. the *direct LP relaxation of the
  exact MILP formulation* (as opposed to the `:mccormick` reformulation, which
  replaces the discrete formulation entirely). With `Σ_l z_{l,t} = 1` the DVFS
  factor becomes a convex combination `a_t = Σ_l a^l z_{l,t}` and the job power a
  shared-multiplier combination `p_{j,t} = Σ_l a^l w^l_{j,t}`, so `p_{j,t} = a_t·w_{j,t}`
  no longer holds exactly at fractional `z`. Combined with
  `relax_battery_binary=true`, the whole model becomes a linear program.

# Minimum Ancillary-Service Participation (must-offer obligation)
- `min_as_reserve::Float64=0.0`: Minimum spinning-reserve offer `P^res_t` [MW] per period.
- `min_as_frp_up::Float64=0.0`: Minimum FRP-up offer `P^{frp,up}_t` [MW] per period.
- `min_as_frp_down::Float64=0.0`: Minimum FRP-down offer `P^{frp,dn}_t` [MW] per period.
- `min_as_total::Float64=0.0`: Minimum total AS offer `P^res_t + P^{frp,up}_t + P^{frp,dn}_t` [MW] per period.
- `as_window::Union{Nothing,AbstractVector{Int}}=nothing`: Set of periods (1-based
  time indices) where the minimum-participation floors apply. If `nothing`, the
  floors apply to every period. Only active when `enable_battery=true`.

The `penalty_style` and other options follow the base model.
"""
function build_model_vcc(data::ProblemDataVCC;
                         optimizer=optimizer_factory(),
                         penalty_style::Symbol=:l1,
                         sense::Symbol=:max,
                         enable_chance_constraints::Bool=false,
                         enable_battery::Bool=true,
                         enable_degradation_penalty::Bool=true,
                         enforce_terminal_soc::Bool=false,
                         terminal_soc_frac::Float64=0.0,
                         dvfs_mode::Symbol=:discrete,
                         n_dvfs_levels::Int=7,
                         dvfs_levels::Union{Nothing, Vector{Float64}}=nothing,
                         relax_battery_binary::Bool=false,
                         relax_dvfs_binary::Bool=false,
                         min_as_reserve::Float64=0.0,
                         min_as_frp_up::Float64=0.0,
                         min_as_frp_down::Float64=0.0,
                         min_as_total::Float64=0.0,
                         as_window::Union{Nothing, AbstractVector{Int}}=nothing,
                         interconnection_slack_penalty::Float64=10000.0,
                         hardware_slack_penalty::Float64=10000.0,
                         allow_interconnection_slack::Bool=true)
    @assert dvfs_mode in (:discrete, :mccormick) "dvfs_mode must be :discrete or :mccormick, got :$dvfs_mode"
    @assert !(relax_dvfs_binary && dvfs_mode != :discrete) "relax_dvfs_binary=true requires dvfs_mode=:discrete (it relaxes the discrete level-selection binaries)"
    @assert n_dvfs_levels >= 2 "n_dvfs_levels must be >= 2, got $n_dvfs_levels"
    @assert min_as_reserve >= 0 && min_as_frp_up >= 0 && min_as_frp_down >= 0 && min_as_total >= 0 "Minimum AS participation floors must be non-negative"

    validate_vcc!(data)
    model = Model(optimizer)

    Tset = collect(data.horizon)
    Jset = collect(1:length(data.jobs))
    Sset = collect(1:length(data.scenarios))
    Δt = data.delta_t

    # Scenario weights (normalized to sum to one)
    weights = [s.weight for s in data.scenarios]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # VCC parameters
    vcc = data.vcc
    φ = vcc.phi
    P_DC_max = vcc.hardware_limit

    # Pre-compute constants
    s_max = sum(job.rate_max for job in data.jobs)
    a_min = data.cost.a_min
    a_max = data.cost.a_max
    a_ref = data.cost.a_ref

    # ========================================================================
    # Compute discrete DVFS levels (if applicable)
    # ========================================================================
    use_discrete = (dvfs_mode == :discrete)
    a_levels = Float64[]
    Lset = Int[]

    if use_discrete
        if dvfs_levels !== nothing
            # User-provided levels
            a_levels = sort(dvfs_levels)
            @assert all(a_min .<= a_levels .<= a_max) "All dvfs_levels must be in [a_min=$a_min, a_max=$a_max]"
        else
            # Generate evenly spaced levels, ensuring a_ref is included
            raw_levels = collect(LinRange(a_min, a_max, n_dvfs_levels))
            # Snap nearest level to a_ref if it's not already exact
            if !any(l -> isapprox(l, a_ref; atol=1e-10), raw_levels)
                # Replace the closest level with a_ref
                _, idx = findmin(abs.(raw_levels .- a_ref))
                raw_levels[idx] = a_ref
            end
            a_levels = sort(unique(raw_levels))
        end
        Lset = collect(1:length(a_levels))
    end

    # ========================================================================
    # Decision Variables
    # ========================================================================

    # Var: Job scheduling
    @variable(model, 0 <= w[Jset, Tset])      # execution rate w_{j,t} (aggregate across levels)
    @variable(model, 0 <= p[Jset, Tset])      # power p_{j,t} = a_t * w_{j,t}
    @variable(model, ℓ[Jset] >= 0)            # unfinished work ℓ_j
    @variable(model, a_min <= a[Tset] <= a_max)  # DVFS factor
    @variable(model, 0 <= s[Tset])            # total execution rate Σ_j w_{j,t}
    @variable(model, 0 <= job_power[Tset])    # total job power Σ_j p_{j,t}

    # Discrete DVFS variables (only created if dvfs_mode == :discrete)
    if use_discrete
        if relax_dvfs_binary
            # Direct LP relaxation: level selectors become convex-combination weights
            @variable(model, 0 <= z[Lset, Tset] <= 1)  # z[l,t]: fractional level weight
        else
            @variable(model, z[Lset, Tset], Bin)       # z[l,t]: DVFS level l selected at time t
        end
        @variable(model, 0 <= w_level[Jset, Lset, Tset]) # w^l_{j,t}: execution rate at level l
    end

    # Var: Fixed load (DVFS-sensitive)
    # P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t
    @variable(model, 0 <= P_fix[Tset])        # P^fix_t(a_t)

    # Var: BESS (only created if battery is enabled)
    if enable_battery
        @variable(model, 0 <= P_charge[Tset])
        @variable(model, 0 <= P_discharge[Tset])
        @variable(model, 0 <= P_reserve[Tset])
        @variable(model, 0 <= P_frp_up[Tset])
        @variable(model, 0 <= P_frp_down[Tset])
        if relax_battery_binary
            # LP relaxation: continuous mode indicators in [0,1]
            @variable(model, 0 <= α[Tset] <= 1)
            @variable(model, 0 <= β[Tset] <= 1)
        else
            @variable(model, α[Tset], Bin)
            @variable(model, β[Tset], Bin)
        end
        @variable(model, E[Tset])
    end

    # Var: net load
    @variable(model, D[Tset])
    @variable(model, ΔD[Tset])

    @variable(model, worst_revenue)

    # Soft constraint slack variables
    @variable(model, load_slack[Sset, Tset] >= 0)
    @variable(model, ramp_slack_up[Sset, Tset] >= 0)
    @variable(model, ramp_slack_down[Sset, Tset] >= 0)

    # Hardware capacity slack (new in VCC)
    @variable(model, hardware_slack[Tset] >= 0)

    # Cycle slack for the degradation penalty.
    if enable_battery && enable_degradation_penalty
        @variable(model, cycle_slack[Sset] >= 0)
    end

    # Chance constraint violation indicators
    if enable_chance_constraints
        @variable(model, load_violation[Sset, Tset], Bin)
        @variable(model, ramp_violation_up[Sset, Tset], Bin)
        @variable(model, ramp_violation_down[Sset, Tset], Bin)
    end

    # ========================================================================
    # Job Scheduling Constraints (eq:dc-schedule1 in Appendix)
    # ========================================================================

    if use_discrete
        # ---- Discrete DVFS constraints ----
        # Exactly one DVFS level per time period
        for t in Tset
            @constraint(model, sum(z[l, t] for l in Lset) == 1)
        end

        # Link a[t] to the selected level
        for t in Tset
            @constraint(model, a[t] == sum(a_levels[l] * z[l, t] for l in Lset))
        end

        for (j_idx, job) in enumerate(data.jobs)
            for t in Tset
                for l in Lset
                    # Level activation: w^l_{j,t} can only be positive if level l is selected
                    @constraint(model, w_level[j_idx, l, t] <= job.rate_max * z[l, t])
                    # Release constraint at each level
                    t < job.release && @constraint(model, w_level[j_idx, l, t] == 0)
                end

                # Aggregate execution rate: w_{j,t} = Σ_l w^l_{j,t}
                @constraint(model, w[j_idx, t] == sum(w_level[j_idx, l, t] for l in Lset))

                # Exact power linking: p_{j,t} = Σ_l a^l · w^l_{j,t}
                # This is LINEAR (no relaxation gap!)
                @constraint(model, p[j_idx, t] == sum(a_levels[l] * w_level[j_idx, l, t] for l in Lset))

                # Rate constraint on aggregate
                @constraint(model, w[j_idx, t] <= job.rate_max)
            end

            # Work completion with DVFS scaling (eq:deadline-soft-mod1, eq:work-complete-mod1)
            # Actual work done: Σ_t (p_{j,t}/a_o) * Δt
            @constraint(model, sum(p[j_idx, t] / a_ref * Δt for t in Tset if t <= job.deadline) +
                              sum(p[j_idx, t] / a_ref * Δt for t in Tset if t > job.deadline) >= job.work - ℓ[j_idx])
            @constraint(model, sum(p[j_idx, t] / a_ref * Δt for t in Tset) <= job.work)
        end
    else
        # ---- McCormick relaxation constraints ----
        for (j_idx, job) in enumerate(data.jobs)
            for t in Tset
                # Release and rate constraints
                t < job.release && @constraint(model, w[j_idx, t] == 0)
                @constraint(model, w[j_idx, t] <= job.rate_max)

                # McCormick linearization for p_{j,t} = a_t * w_{j,t} (eq:p-link1)
                @constraint(model, p[j_idx, t] >= a_min * w[j_idx, t])
                @constraint(model, p[j_idx, t] >= a_max * w[j_idx, t] + job.rate_max * a[t] - a_max * job.rate_max)
                @constraint(model, p[j_idx, t] <= a_max * w[j_idx, t])
                @constraint(model, p[j_idx, t] <= a_min * w[j_idx, t] + job.rate_max * a[t] - a_min * job.rate_max)
            end

            # Work completion with DVFS scaling (eq:deadline-soft-mod1, eq:work-complete-mod1)
            # Actual work done: Σ_t (p_{j,t}/a_o) * Δt
            @constraint(model, sum(p[j_idx, t] / a_ref * Δt for t in Tset if t <= job.deadline) +
                              sum(p[j_idx, t] / a_ref * Δt for t in Tset if t > job.deadline) >= job.work - ℓ[j_idx])
            @constraint(model, sum(p[j_idx, t] / a_ref * Δt for t in Tset) <= job.work)
        end
    end

    # Total execution rate and job power
    for t in Tset
        @constraint(model, s[t] == sum(w[j_idx, t] for j_idx in Jset))
        @constraint(model, job_power[t] == sum(p[j_idx, t] for j_idx in Jset))
    end

    # ========================================================================
    # Fixed Load with DVFS Sensitivity (eq:pred-fixed-load-appendix)
    # P^fix_t(a_t) = (1-φ)P̂⁰_t + φ(a_t/a_o)P̂⁰_t
    # ========================================================================

    for (idx, t) in enumerate(Tset)
        base = data.base_load[idx]

        if !iszero(base)
            if use_discrete
                # With discrete levels, use the binary selection directly:
                # P^fix_t = Σ_l [(1-φ)*base + φ*base/a_ref * a^l] * z[l,t]
                P_fix_at_level = [base * (1 - φ) + (φ * base / a_ref) * a_levels[l] for l in Lset]
                @constraint(model, P_fix[t] == sum(P_fix_at_level[l] * z[l, t] for l in Lset))
            else
                # P^fix_t(a_t) = base*(1-φ) + (φ*base/a_ref)*a_t
                # This is linear in a_t, no McCormick needed!
                @constraint(model, P_fix[t] == base * (1 - φ) + (φ * base / a_ref) * a[t])
            end
        else
            @constraint(model, P_fix[t] == 0)
        end
    end

    # ========================================================================
    # Hardware Capacity Constraint (eq:capacity-appendix)
    # Σ_j p_{j,t} + P̄^fix_t(a_o) ≤ P̄^DC + hardware_slack[t]
    # where P̄^fix_t(a_o) is the upper bound of fixed load at reference frequency
    # ========================================================================

    for (idx, t) in enumerate(Tset)
        # P̄^fix_t(a_o) from data.base_load_upper (precomputed)
        P_fix_upper = data.base_load_upper[idx]
        @constraint(model, job_power[t] + P_fix_upper <= P_DC_max + hardware_slack[t])
    end

    # ========================================================================
    # Power Balance (eq:power-balance-appendix)
    # D_t = P_charge_t - P_discharge_t + P_fix_t + job_power_t
    # ========================================================================

    for t in Tset
        if enable_battery
            @constraint(model, D[t] == P_charge[t] - P_discharge[t] + P_fix[t] + job_power[t])
        else
            @constraint(model, D[t] == P_fix[t] + job_power[t])
        end
    end

    # ========================================================================
    # BESS Operation Constraints (same as base model)
    # ========================================================================

    if enable_battery
        b = data.bess
        E0 = b.capacity * b.soc_init
        soc_min = b.soc_min * b.capacity
        soc_max = b.soc_max * b.capacity

        for (i, t) in enumerate(Tset)
            @constraint(model, α[t] + β[t] <= 1)
            @constraint(model, P_charge[t] + P_frp_up[t] <= α[t] * b.charge_max)
            @constraint(model, P_discharge[t] + P_reserve[t] + P_frp_down[t] <= β[t] * b.discharge_max)

            # Energy balance (base schedule)
            inflow = (P_charge[t] * b.eta_charge - P_discharge[t] / b.eta_discharge) * Δt
            if i == 1
                @constraint(model, E[t] == E0 + inflow)
            else
                t_prev = Tset[i-1]
                @constraint(model, E[t] == E[t_prev] + inflow)
            end
        end

        # ====================================================================
        # Minimum Ancillary-Service Participation (must-offer obligation)
        # The resource is contractually obligated to offer a minimum amount of
        # ancillary services in each period of the participation window.
        # ====================================================================
        has_min_as = (min_as_reserve > 0.0) || (min_as_frp_up > 0.0) ||
                     (min_as_frp_down > 0.0) || (min_as_total > 0.0)
        if has_min_as
            as_periods = as_window === nothing ? Tset : [t for t in Tset if t in as_window]
            for t in as_periods
                if min_as_reserve > 0.0
                    @constraint(model, P_reserve[t] >= min_as_reserve)
                end
                if min_as_frp_up > 0.0
                    @constraint(model, P_frp_up[t] >= min_as_frp_up)
                end
                if min_as_frp_down > 0.0
                    @constraint(model, P_frp_down[t] >= min_as_frp_down)
                end
                if min_as_total > 0.0
                    @constraint(model, P_reserve[t] + P_frp_up[t] + P_frp_down[t] >= min_as_total)
                end
            end
        end
    end

    # Ramp rate calculation
    for (i, t) in enumerate(Tset)
        if i == 1
            @constraint(model, ΔD[t] == D[t] - data.interconnection.initial_net_load)
        else
            t_prev = Tset[i-1]
            @constraint(model, ΔD[t] == D[t] - D[t_prev])
        end
    end

    # Scenario-based SOC constraints
    if enable_battery
        b = data.bess
        E0 = b.capacity * b.soc_init
        soc_min = b.soc_min * b.capacity
        soc_max = b.soc_max * b.capacity

        for t in Tset
            @constraint(model, soc_min <= E[t] <= soc_max)
        end

        # Terminal SOC constraint: battery must end at or above a specified SOC
        # Prevents spurious "free energy" from initial SOC depletion
        if enforce_terminal_soc
            soc_target = terminal_soc_frac > 0.0 ? terminal_soc_frac * b.capacity : E0
            @constraint(model, E[Tset[end]] >= soc_target)
        end

        for (s_idx, scen) in enumerate(data.scenarios)
            E_scen = E0
            for (k, t) in enumerate(Tset)
                charge_deployed = P_charge[t] + scen.beta_frp_up[k] * P_frp_up[t]
                discharge_deployed = P_discharge[t] + scen.beta_reserve[k] * P_reserve[t] + scen.beta_frp_down[k] * P_frp_down[t]
                inflow_scen = (charge_deployed * b.eta_charge - discharge_deployed / b.eta_discharge) * Δt

                E_scen_t = (k == 1) ? E0 + inflow_scen : E_scen + inflow_scen
                @constraint(model, E_scen_t >= soc_min)
                @constraint(model, E_scen_t <= soc_max)

                if k < length(Tset)
                    E_scen = E_scen_t
                end
            end
        end

        # Degradation penalty constraints
        if enable_degradation_penalty
            for (s_idx, scen) in enumerate(data.scenarios)
                throughput_expr = @expression(model, sum(
                    Δt * (
                        (P_charge[t] + scen.beta_frp_up[k] * P_frp_up[t]) +
                        (P_discharge[t] + scen.beta_reserve[k] * P_reserve[t] + scen.beta_frp_down[k] * P_frp_down[t])
                    )
                    for (k, t) in enumerate(Tset)
                ))
                @constraint(model, throughput_expr <= 2 * b.capacity * (b.n_cyc_max + cycle_slack[s_idx]))
            end
        end
    end

    # ========================================================================
    # Interconnection Constraints
    # ========================================================================

    ic = data.interconnection

    if enable_chance_constraints
        bigM_load = ic.big_m_load
        bigM_ramp = ic.big_m_ramp

        for (s_idx, scen) in enumerate(data.scenarios)
            prev_error = 0.0
            for (k, t) in enumerate(Tset)
                load_expr = D[t] + scen.load_error[k]
                @constraint(model, load_expr - ic.load_cap <= bigM_load * load_violation[s_idx, t])

                error_diff = scen.load_error[k] - prev_error
                ramp_expr = ΔD[t] + error_diff
                @constraint(model, ramp_expr - ic.ramp_cap <= bigM_ramp * ramp_violation_up[s_idx, t])
                @constraint(model, -ramp_expr - ic.ramp_cap <= bigM_ramp * ramp_violation_down[s_idx, t])

                prev_error = scen.load_error[k]
            end
        end

        for t in Tset
            @constraint(model, sum(weights[s_idx] * load_violation[s_idx, t] for s_idx in Sset) <= ic.delta_load_risk)
            @constraint(model, sum(weights[s_idx] * ramp_violation_up[s_idx, t] for s_idx in Sset) <= ic.delta_ramp_risk)
            @constraint(model, sum(weights[s_idx] * ramp_violation_down[s_idx, t] for s_idx in Sset) <= ic.delta_ramp_risk)
        end
    else
        ic_load_cons = Dict{Tuple{Int,Int}, ConstraintRef}()
        ic_ramp_up_cons = Dict{Tuple{Int,Int}, ConstraintRef}()
        ic_ramp_down_cons = Dict{Tuple{Int,Int}, ConstraintRef}()

        for (s_idx, scen) in enumerate(data.scenarios)
            prev_error = 0.0
            for (k, t) in enumerate(Tset)
                load_expr = D[t] + scen.load_error[k]
                c_load = if allow_interconnection_slack
                    @constraint(model, load_expr <= ic.load_cap + load_slack[s_idx, t])
                else
                    @constraint(model, load_expr <= ic.load_cap)
                end
                ic_load_cons[(s_idx, t)] = c_load

                error_diff = scen.load_error[k] - prev_error
                ramp_expr = ΔD[t] + error_diff
                c_ramp_up = if allow_interconnection_slack
                    @constraint(model, ramp_expr <= ic.ramp_cap + ramp_slack_up[s_idx, t])
                else
                    @constraint(model, ramp_expr <= ic.ramp_cap)
                end
                c_ramp_down = if allow_interconnection_slack
                    @constraint(model, -ramp_expr <= ic.ramp_cap + ramp_slack_down[s_idx, t])
                else
                    @constraint(model, -ramp_expr <= ic.ramp_cap)
                end
                ic_ramp_up_cons[(s_idx, t)] = c_ramp_up
                ic_ramp_down_cons[(s_idx, t)] = c_ramp_down

                prev_error = scen.load_error[k]
            end
        end

        model.ext[:ic_load_constraints] = ic_load_cons
        model.ext[:ic_ramp_up_constraints] = ic_ramp_up_cons
        model.ext[:ic_ramp_down_constraints] = ic_ramp_down_cons
    end

    # ========================================================================
    # Revenue Expressions
    # ========================================================================

    if enable_battery
        reserve_revenue = @expression(model, sum(data.pricing.lambda_reserve[idx] * P_reserve[t] * Δt for (idx, t) in enumerate(Tset)))
        frp_revenue = @expression(model, sum((data.pricing.lambda_frp_up[idx] * P_frp_up[t] + data.pricing.lambda_frp_down[idx] * P_frp_down[t]) * Δt for (idx, t) in enumerate(Tset)))
    else
        reserve_revenue = @expression(model, 0.0)
        frp_revenue = @expression(model, 0.0)
    end

    scenario_energy_cost = Dict{Int, JuMP.AffExpr}()
    for (s_idx, scen) in enumerate(data.scenarios)
        expr = JuMP.AffExpr(0.0)
        for (idx, t) in enumerate(Tset)
            load_expr = D[t] + scen.load_error[idx]
            JuMP.add_to_expression!(expr, data.pricing.lambda_energy[idx] * Δt * load_expr)
            if enable_battery
                JuMP.add_to_expression!(expr, -data.pricing.lambda_energy[idx] * Δt * scen.beta_reserve[idx] * P_reserve[t])
                JuMP.add_to_expression!(expr, data.pricing.lambda_energy[idx] * Δt * scen.beta_frp_up[idx] * P_frp_up[t])
                JuMP.add_to_expression!(expr, -data.pricing.lambda_energy[idx] * Δt * scen.beta_frp_down[idx] * P_frp_down[t])
            end
        end
        scenario_energy_cost[s_idx] = expr

        revenue_expr = data.pricing.lambda_weight * (reserve_revenue + frp_revenue - expr)
        if enable_battery && enable_degradation_penalty
            degradation_penalty = data.bess.c_degradation * (2 * data.bess.capacity * cycle_slack[s_idx])
            @constraint(model, worst_revenue <= revenue_expr - degradation_penalty)
        else
            @constraint(model, worst_revenue <= revenue_expr)
        end
    end

    # ========================================================================
    # Objective Function (eq:obj-affine1 with VCC additions)
    # ========================================================================

    # Late execution penalty: c3 * Σ_{j,t>d_j} κ_j(t-d_j)(p_{j,t}/a_o)
    late_execution_cost = @expression(model, data.cost.c3 * sum(
        data.jobs[j_idx].tardiness_weight * (t - data.jobs[j_idx].deadline) * p[j_idx, t] / a_ref
        for j_idx in Jset for t in Tset if t > data.jobs[j_idx].deadline
    ))

    # Unfinished work penalty
    unfinished_cost = @expression(model, data.cost.c2 * sum(data.jobs[j_idx].tardiness_weight * ℓ[j_idx] for j_idx in Jset))

    # Slack penalties
    capacity_violation_cost = @expression(model, interconnection_slack_penalty * sum(
        weights[s_idx] * (load_slack[s_idx, t] + ramp_slack_up[s_idx, t] + ramp_slack_down[s_idx, t])
        for s_idx in Sset for t in Tset
    ))

    # Hardware capacity violation cost (new in VCC)
    hardware_violation_cost = @expression(model, hardware_slack_penalty * sum(hardware_slack[t] for t in Tset))

    # DVFS deviation penalty
    deviation_cost = zero(JuMP.AffExpr)
    dev_vars = nothing

    penalty_style_normalized = penalty_style
    if penalty_style == :absolute
        penalty_style_normalized = :l_infinity
    elseif penalty_style == :quadratic
        penalty_style_normalized = :l2
    end

    if penalty_style_normalized == :l_infinity
        max_dev = @variable(model, max_dev >= 0)
        dev_vars = max_dev
        @constraint(model, max_dev .>= a .- a_ref)
        @constraint(model, max_dev .>= -(a .- a_ref))
        deviation_cost = @expression(model, data.cost.c1 * max_dev)
    elseif penalty_style_normalized == :l1
        dev_pos = @variable(model, dev_pos[Tset] >= 0)
        dev_neg = @variable(model, dev_neg[Tset] >= 0)
        dev_vars = (dev_pos=dev_pos, dev_neg=dev_neg)
        @constraint(model, a .- a_ref .== dev_pos .- dev_neg)
        deviation_cost = @expression(model, data.cost.c1 * sum(dev_pos .+ dev_neg))
    elseif penalty_style_normalized == :l2
        if optimizer == HiGHS.Optimizer
            @warn "L2 penalty creates MIQP. HiGHS doesn't support MIQP."
        end
        deviation_cost = @expression(model, data.cost.c1 * sum((a[t] - a_ref)^2 for t in Tset))
    elseif penalty_style_normalized == :none
        deviation_cost = 0.0
    else
        error("Unsupported penalty_style $(penalty_style)")
    end

    total_flexibility_cost = late_execution_cost + unfinished_cost + deviation_cost +
                             capacity_violation_cost + hardware_violation_cost

    if sense == :max
        @objective(model, Max, worst_revenue - total_flexibility_cost)
    else
        @objective(model, Min, total_flexibility_cost - worst_revenue)
    end

    # Store metadata
    model.ext[:dc_data] = data

    vars_dict = Dict{Symbol, Any}(
        :w => w, :p => p, :ℓ => ℓ, :a => a, :s => s,
        :job_power => job_power, :P_fix => P_fix,
        :D => D, :ΔD => ΔD, :worst_revenue => worst_revenue,
        :dev_vars => dev_vars, :load_slack => load_slack,
        :ramp_slack_up => ramp_slack_up, :ramp_slack_down => ramp_slack_down,
        :hardware_slack => hardware_slack
    )

    if use_discrete
        vars_dict[:z] = z
        vars_dict[:w_level] = w_level
    end

    if enable_battery
        vars_dict[:P_charge] = P_charge
        vars_dict[:P_discharge] = P_discharge
        vars_dict[:P_reserve] = P_reserve
        vars_dict[:P_frp_up] = P_frp_up
        vars_dict[:P_frp_down] = P_frp_down
        vars_dict[:α] = α
        vars_dict[:β] = β
        vars_dict[:E] = E
        if enable_degradation_penalty
            vars_dict[:cycle_slack] = cycle_slack
        end
    end

    if enable_chance_constraints
        vars_dict[:load_violation] = load_violation
        vars_dict[:ramp_violation_up] = ramp_violation_up
        vars_dict[:ramp_violation_down] = ramp_violation_down
    end

    model.ext[:dc_vars] = (; (Symbol(k) => v for (k, v) in vars_dict)...)
    model.ext[:dc_settings] = (; penalty_style=penalty_style_normalized, sense, weights,
                                enable_chance_constraints, enable_battery, enable_degradation_penalty,
                                formulation=:vcc, dvfs_mode=dvfs_mode,
                                interconnection_slack_penalty, hardware_slack_penalty,
                                allow_interconnection_slack,
                                relax_battery_binary=relax_battery_binary,
                                relax_dvfs_binary=relax_dvfs_binary,
                                min_as_reserve=min_as_reserve, min_as_frp_up=min_as_frp_up,
                                min_as_frp_down=min_as_frp_down, min_as_total=min_as_total,
                                as_window=as_window,
                                dvfs_levels=use_discrete ? a_levels : nothing,
                                n_dvfs_levels=use_discrete ? length(a_levels) : 0)

    return model
end


"""
    mccormick_feasibility_check(model)

Check the relaxation gap of a McCormick-mode solution: for every (j,t) pair,
compute |p_{j,t} - a_t * w_{j,t}|.  Returns a Dict with:
  - :max_gap        maximum absolute gap across all (j,t)
  - :mean_gap       mean absolute gap
  - :gap_matrix     Matrix{Float64}(J, T) of per-element gaps
  - :is_feasible    true when max_gap < atol (default 1e-3)
"""
function mccormick_feasibility_check(model::Model; atol::Float64=1e-3)
    data = model.ext[:dc_data]
    vars = model.ext[:dc_vars]
    Tset = collect(data.horizon)
    Jset = 1:length(data.jobs)

    gap_matrix = zeros(length(Jset), length(Tset))
    for (ti, t) in enumerate(Tset)
        a_val = JuMP.value(vars.a[t])
        for j in Jset
            w_val = JuMP.value(vars.w[j, t])
            p_val = JuMP.value(vars.p[j, t])
            gap_matrix[j, ti] = abs(p_val - a_val * w_val)
        end
    end

    max_gap = maximum(gap_matrix)
    mean_gap = sum(gap_matrix) / length(gap_matrix)

    return Dict(
        :max_gap => max_gap,
        :mean_gap => mean_gap,
        :gap_matrix => gap_matrix,
        :is_feasible => max_gap < atol,
    )
end


"""
    mccormick_project_solution(model)

Post-solve projection for McCormick solutions.  For each (j,t), re-computes the
"true" power as `p̃_{j,t} = a_t * w_{j,t}` and returns corrected values.  This
guarantees the bilinear identity holds exactly, at the cost of small deviations
in constraints that depend on p_{j,t}.

Returns a Dict with:
  - :projected_p       Dict((j,t) => projected power)
  - :projected_job_power  Dict(t => total projected job power)
  - :projected_net_load   Dict(t => corrected net load D_t)
  - :max_correction    largest |p_proj - p_relax| across (j,t)
  - :corrections       Dict((j,t) => p_proj - p_relax)
"""
function mccormick_project_solution(model::Model)
    data = model.ext[:dc_data]
    vars = model.ext[:dc_vars]
    settings = model.ext[:dc_settings]
    Tset = collect(data.horizon)
    Jset = 1:length(data.jobs)
    enable_battery = get(settings, :enable_battery, true)

    projected_p = Dict{Tuple{Int,Int}, Float64}()
    corrections = Dict{Tuple{Int,Int}, Float64}()
    projected_job_power = Dict{Int, Float64}()
    projected_net_load = Dict{Int, Float64}()

    max_correction = 0.0

    for (ti, t) in enumerate(Tset)
        a_val = JuMP.value(vars.a[t])
        jp = 0.0
        for j in Jset
            w_val = JuMP.value(vars.w[j, t])
            p_relax = JuMP.value(vars.p[j, t])
            p_proj = a_val * w_val
            projected_p[(j, t)] = p_proj
            corrections[(j, t)] = p_proj - p_relax
            max_correction = max(max_correction, abs(p_proj - p_relax))
            jp += p_proj
        end
        projected_job_power[t] = jp

        P_fix_val = JuMP.value(vars.P_fix[t])
        if enable_battery
            D_corr = JuMP.value(vars.P_charge[t]) - JuMP.value(vars.P_discharge[t]) + P_fix_val + jp
        else
            D_corr = P_fix_val + jp
        end
        projected_net_load[t] = D_corr
    end

    return Dict(
        :projected_p => projected_p,
        :projected_job_power => projected_job_power,
        :projected_net_load => projected_net_load,
        :max_correction => max_correction,
        :corrections => corrections,
    )
end


"""
    build_model_vcc_mccormick_tightened(data::ProblemDataVCC; max_iterations=3, kwargs...)

Iterative bound-tightening wrapper around `build_model_vcc` in McCormick mode.

After the initial solve, the procedure:
1. Reads the optimal a_t values.
2. Tightens the bounds on a_t to [a_t - ε, a_t + ε] per period (where ε shrinks
   each iteration).
3. Re-solves with the tighter bounds, producing a progressively smaller McCormick
   relaxation gap.

This converges to zero gap when a_t is fixed, and in practice eliminates >95%
of the relaxation gap in 2–3 iterations.

Returns `(model, gap_history)`.
"""
function build_model_vcc_mccormick_tightened(data::ProblemDataVCC;
        max_iterations::Int=3, gap_tol::Float64=1e-4, kwargs...)

    gap_history = Float64[]

    # Initial solve with full bounds
    model = build_model_vcc(data; dvfs_mode=:mccormick, kwargs...)
    set_silent(model)
    optimize!(model)

    if termination_status(model) ∉ [MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.LOCALLY_SOLVED]
        @warn "Initial McCormick solve did not find optimal solution"
        return model, gap_history
    end

    check = mccormick_feasibility_check(model)
    push!(gap_history, check[:max_gap])

    if check[:is_feasible]
        println("McCormick gap already below tolerance after initial solve: $(check[:max_gap])")
        return model, gap_history
    end

    a_min_orig = data.cost.a_min
    a_max_orig = data.cost.a_max
    Tset = collect(data.horizon)

    for iter in 1:max_iterations
        # Read a_t from the previous solve
        vars = model.ext[:dc_vars]
        a_solved = [JuMP.value(vars.a[t]) for t in Tset]

        # Shrink bounds: use progressively tighter interval around the solved a_t
        # Shrink factor: 0.5^iter of the original range
        range_half = (a_max_orig - a_min_orig) / 2.0 * (0.5^iter)

        # Rebuild with tightened per-period bounds
        # We encode this by adding post-hoc constraints to a fresh model
        model = build_model_vcc(data; dvfs_mode=:mccormick, kwargs...)
        set_silent(model)

        new_vars = model.ext[:dc_vars]
        for (ti, t) in enumerate(Tset)
            lb = max(a_min_orig, a_solved[ti] - range_half)
            ub = min(a_max_orig, a_solved[ti] + range_half)
            JuMP.set_lower_bound(new_vars.a[t], lb)
            JuMP.set_upper_bound(new_vars.a[t], ub)
        end

        optimize!(model)

        if termination_status(model) ∉ [MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.LOCALLY_SOLVED]
            @warn "McCormick tightening iteration $iter did not converge"
            break
        end

        check = mccormick_feasibility_check(model)
        push!(gap_history, check[:max_gap])
        println("McCormick tightening iteration $iter: max gap = $(round(check[:max_gap], sigdigits=4))")

        if check[:max_gap] < gap_tol
            println("McCormick gap below tolerance ($gap_tol) after $iter tightening iterations")
            break
        end
    end

    return model, gap_history
end


"""
    solution_summary_vcc(model)

Collect key solution statistics for the VCC formulation and return them as a
dictionary of DataFrames/values.
"""
function solution_summary_vcc(model::Model)
    data = get(model.ext, :dc_data, nothing)
    vars = get(model.ext, :dc_vars, nothing)
    settings = get(model.ext, :dc_settings, nothing)
    data === nothing && error("Model does not carry dc_data metadata.")
    vars === nothing && error("Model does not carry dc_vars metadata.")
    settings === nothing && error("Model does not carry dc_settings metadata.")

    enable_battery = get(settings, :enable_battery, true)

    status = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    obj = JuMP.objective_value(model)

    Tset = collect(data.horizon)
    a_ref = data.cost.a_ref

    # Job summary using p_{j,t} instead of v_{j,t}
    job_table = DataFrame(job=String[], release=Int[], deadline=Int[], work=Float64[],
                         completed=Float64[], completed_by_deadline=Float64[],
                         late_work=Float64[], late_penalty=Float64[], unfinished=Float64[])
    for (j_idx, job) in enumerate(data.jobs)
        # Actual work = Σ_t (p_{j,t}/a_o) * Δt
        completed = sum(JuMP.value(vars.p[j_idx, t]) / a_ref * data.delta_t for t in Tset)
        completed_by_deadline = sum(JuMP.value(vars.p[j_idx, t]) / a_ref * data.delta_t for t in Tset if t <= job.deadline)
        late_work = sum(JuMP.value(vars.p[j_idx, t]) / a_ref * data.delta_t for t in Tset if t > job.deadline; init=0.0)
        late_penalty = job.tardiness_weight * sum(
            (t - job.deadline) * JuMP.value(vars.p[j_idx, t]) / a_ref * data.delta_t
            for t in Tset if t > job.deadline; init=0.0
        )
        unfinished = JuMP.value(vars.ℓ[j_idx])
        push!(job_table, (job.name, job.release, job.deadline, job.work, completed,
                         completed_by_deadline, late_work, late_penalty, unfinished))
    end

    # BESS table
    bess_table = DataFrame(time=Int[], charge=Float64[], discharge=Float64[],
                          reserve=Float64[], frp_up=Float64[], frp_down=Float64[],
                          energy=Float64[], soc=Float64[], net_load=Float64[])

    if enable_battery
        E0 = data.bess.capacity * data.bess.soc_init
        initial_load = data.base_load[1]
        push!(bess_table, (0, 0.0, 0.0, 0.0, 0.0, 0.0, E0, data.bess.soc_init, initial_load))

        for (idx, t) in enumerate(Tset)
            charge = JuMP.value(vars.P_charge[t])
            discharge = JuMP.value(vars.P_discharge[t])
            reserve = JuMP.value(vars.P_reserve[t])
            frp_up = JuMP.value(vars.P_frp_up[t])
            frp_down = JuMP.value(vars.P_frp_down[t])
            energy = JuMP.value(vars.E[t])
            soc = energy / data.bess.capacity
            net_load = JuMP.value(vars.D[t])
            push!(bess_table, (t, charge, discharge, reserve, frp_up, frp_down, energy, soc, net_load))
        end
    else
        initial_load = data.base_load[1]
        push!(bess_table, (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initial_load))
        for (idx, t) in enumerate(Tset)
            net_load = JuMP.value(vars.D[t])
            push!(bess_table, (t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, net_load))
        end
    end

    # VCC-specific metrics
    vcc_table = DataFrame(time=Int[], P_fix=Float64[], job_power=Float64[],
                         a_factor=Float64[], hardware_slack=Float64[],
                         P_fix_upper=Float64[], headroom=Float64[])

    for (idx, t) in enumerate(Tset)
        P_fix_val = JuMP.value(vars.P_fix[t])
        job_power_val = JuMP.value(vars.job_power[t])
        a_val = JuMP.value(vars.a[t])
        hw_slack = JuMP.value(vars.hardware_slack[t])
        P_fix_upper = data.base_load_upper[idx]
        headroom = data.vcc.hardware_limit - P_fix_upper - job_power_val
        push!(vcc_table, (t, P_fix_val, job_power_val, a_val, hw_slack, P_fix_upper, headroom))
    end

    # Scenario revenues
    weights = settings.weights
    scen_revenues = Float64[]
    if enable_battery
        reserve_revenue = sum(data.pricing.lambda_reserve[idx] * JuMP.value(vars.P_reserve[t]) * data.delta_t
                             for (idx, t) in enumerate(Tset))
        frp_revenue = sum((data.pricing.lambda_frp_up[idx] * JuMP.value(vars.P_frp_up[t]) +
                          data.pricing.lambda_frp_down[idx] * JuMP.value(vars.P_frp_down[t])) * data.delta_t
                         for (idx, t) in enumerate(Tset))
    else
        reserve_revenue = 0.0
        frp_revenue = 0.0
    end

    for (s_idx, scen) in enumerate(data.scenarios)
        actual_load = [JuMP.value(vars.D[t]) + scen.load_error[idx] for (idx, t) in enumerate(Tset)]
        if enable_battery
            scen_energy = sum(data.pricing.lambda_energy[idx] * data.delta_t *
                             (actual_load[idx] - scen.beta_reserve[idx] * JuMP.value(vars.P_reserve[t]) +
                              scen.beta_frp_up[idx] * JuMP.value(vars.P_frp_up[t]) -
                              scen.beta_frp_down[idx] * JuMP.value(vars.P_frp_down[t]))
                             for (idx, t) in enumerate(Tset))
        else
            scen_energy = sum(data.pricing.lambda_energy[idx] * data.delta_t * actual_load[idx]
                             for (idx, t) in enumerate(Tset))
        end
        revenue = data.pricing.lambda_weight * (reserve_revenue + frp_revenue - scen_energy)
        push!(scen_revenues, revenue)
    end
    scenario_df = DataFrame(name=[s.name for s in data.scenarios], weight=[s.weight for s in data.scenarios],
                           revenue=scen_revenues)

    # Slack violations
    slack_violations = Dict{Symbol, Float64}()
    slack_violations[:total_load_slack] = sum(weights[s_idx] * JuMP.value(vars.load_slack[s_idx, t])
                                              for s_idx in 1:length(data.scenarios) for t in Tset)
    slack_violations[:total_ramp_slack_up] = sum(weights[s_idx] * JuMP.value(vars.ramp_slack_up[s_idx, t])
                                                 for s_idx in 1:length(data.scenarios) for t in Tset)
    slack_violations[:total_ramp_slack_down] = sum(weights[s_idx] * JuMP.value(vars.ramp_slack_down[s_idx, t])
                                                   for s_idx in 1:length(data.scenarios) for t in Tset)
    slack_violations[:total_hardware_slack] = sum(JuMP.value(vars.hardware_slack[t]) for t in Tset)
    slack_violations[:max_hardware_slack] = maximum(JuMP.value(vars.hardware_slack[t]) for t in Tset)

    # DVFS mode information
    dvfs_mode = get(settings, :dvfs_mode, :mccormick)
    dvfs_info = Dict{Symbol, Any}(
        :mode => dvfs_mode,
    )

    if dvfs_mode == :discrete
        dvfs_levels = get(settings, :dvfs_levels, nothing)
        n_levels = get(settings, :n_dvfs_levels, 0)
        dvfs_info[:levels] = dvfs_levels
        dvfs_info[:n_levels] = n_levels

        # Report which level was selected at each time period
        if hasproperty(vars, :z)
            selected_levels = Int[]
            for t in Tset
                for l in 1:n_levels
                    if JuMP.value(vars.z[l, t]) > 0.5
                        push!(selected_levels, l)
                        break
                    end
                end
            end
            dvfs_info[:selected_level_indices] = selected_levels
            dvfs_info[:selected_level_values] = [dvfs_levels[l] for l in selected_levels]
        end
    end

    return Dict(
        :termination_status => status,
        :primal_status => primal,
        :objective_value => obj,
        :worst_case_revenue => JuMP.value(vars.worst_revenue),
        :jobs => job_table,
        :bess => bess_table,
        :vcc => vcc_table,
        :scenarios => scenario_df,
        :slack_violations => slack_violations,
        :enable_battery => enable_battery,
        :hardware_limit => data.vcc.hardware_limit,
        :phi => data.vcc.phi,
        :dvfs_info => dvfs_info,
    )
end
