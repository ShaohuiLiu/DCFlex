using DCFlex
using JuMP
using HiGHS
using Printf
using Statistics
import MathOptInterface as MOI

include(joinpath(@__DIR__, "real_market_data_case_vcc.jl"))

function copy_with_params(data::ProblemDataVCC; load_cap=data.interconnection.load_cap,
                          ramp_cap=data.interconnection.ramp_cap,
                          c1=data.cost.c1, c2=data.cost.c2, c3=data.cost.c3,
                          cdeg=data.bess.c_degradation, ncyc=data.bess.n_cyc_max,
                          energy_spread=1.0, as_scale=1.0)
    ic0 = data.interconnection
    ic = InterconnectionLimits(load_cap, ramp_cap, ic0.delta_load_risk, ic0.delta_ramp_risk,
                               ic0.initial_net_load, ic0.big_m_load, ic0.big_m_ramp)
    cost = CostParameters(c1, c2, c3, data.cost.a_ref, data.cost.a_min, data.cost.a_max)
    b0 = data.bess
    bess = BESSParameters(b0.capacity, b0.soc_init, b0.soc_min, b0.soc_max,
                          b0.charge_max, b0.discharge_max, b0.eta_charge,
                          b0.eta_discharge, cdeg, ncyc)

    λe0 = data.pricing.lambda_energy
    μe = mean(λe0)
    λe = [max(1.0, μe + energy_spread * (x - μe)) for x in λe0]
    pricing = PricingData(λe,
                          as_scale .* data.pricing.lambda_reserve,
                          as_scale .* data.pricing.lambda_frp_up,
                          as_scale .* data.pricing.lambda_frp_down,
                          data.pricing.lambda_weight)

    ProblemDataVCC(data.horizon, data.delta_t, data.base_load, data.base_load_upper,
                   data.jobs, bess, pricing, cost, ic, data.scenarios, data.vcc)
end

function solve_case(data::ProblemDataVCC; battery::Bool, terminal_soc::Bool=false,
                    slack_penalty::Float64=10000.0, allow_slack::Bool=true)
    model = build_model_vcc(data; optimizer=optimizer_factory(),
                            enable_battery=battery,
                            enforce_terminal_soc=terminal_soc,
                            terminal_soc_frac=data.bess.soc_init,
                            interconnection_slack_penalty=slack_penalty,
                            allow_interconnection_slack=allow_slack)
    set_silent(model)
    set_time_limit_sec(model, 120.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)
    optimize!(model)
    return model
end

function metrics(model, data::ProblemDataVCC)
    vars = model.ext[:dc_vars]
    Tset = collect(data.horizon)
    obj = objective_value(model)
    P_fix = [value(vars.P_fix[t]) for t in Tset]
    job_power = [value(vars.job_power[t]) for t in Tset]
    dc_load = P_fix .+ job_power
    net_load = [value(vars.D[t]) for t in Tset]
    a = [value(vars.a[t]) for t in Tset]
    unfinished = sum(value(vars.ℓ[j]) for j in 1:length(data.jobs))
    load_slack = sum(value(vars.load_slack[s, t]) for s in 1:length(data.scenarios) for t in Tset) / length(data.scenarios)
    ramp_slack = sum(value(vars.ramp_slack_up[s, t]) + value(vars.ramp_slack_down[s, t])
                     for s in 1:length(data.scenarios) for t in Tset) / length(data.scenarios)

    charge = 0.0
    discharge = 0.0
    reserve = 0.0
    frp_up = 0.0
    frp_down = 0.0
    if haskey(model.ext[:dc_settings], :enable_battery) && model.ext[:dc_settings].enable_battery
        charge = sum(value(vars.P_charge[t]) for t in Tset)
        discharge = sum(value(vars.P_discharge[t]) for t in Tset)
        reserve = sum(value(vars.P_reserve[t]) for t in Tset)
        frp_up = sum(value(vars.P_frp_up[t]) for t in Tset)
        frp_down = sum(value(vars.P_frp_down[t]) for t in Tset)
    end

    return (obj=obj, max_dc=maximum(dc_load), max_net=maximum(net_load),
            max_ramp=maximum(abs.(diff([data.interconnection.initial_net_load; net_load]))),
            amin=minimum(a), amax=maximum(a), unfinished=unfinished,
            load_slack=load_slack, ramp_slack=ramp_slack,
            charge=charge, discharge=discharge, reserve=reserve,
            frp_up=frp_up, frp_down=frp_down)
end

configs = [
    (name="base100", load_cap=100.0, ramp_cap=15.0, c1=1000.0, c2=2000.0, cdeg=45.0, ncyc=0.5, espread=1.0, as_scale=1.0, terminal=false),
    (name="tight98", load_cap=98.0, ramp_cap=15.0, c1=1000.0, c2=5000.0, cdeg=20.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="tight95", load_cap=95.0, ramp_cap=15.0, c1=1000.0, c2=5000.0, cdeg=20.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="ramp8", load_cap=100.0, ramp_cap=8.0, c1=1000.0, c2=5000.0, cdeg=20.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="tight98_ramp8", load_cap=98.0, ramp_cap=8.0, c1=1000.0, c2=5000.0, cdeg=20.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="high_dvfs_cost", load_cap=100.0, ramp_cap=15.0, c1=12000.0, c2=5000.0, cdeg=20.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="tight98_high_dvfs", load_cap=98.0, ramp_cap=12.0, c1=12000.0, c2=8000.0, cdeg=15.0, ncyc=1.0, espread=1.0, as_scale=1.0, terminal=false),
    (name="arb_spread_terminal", load_cap=100.0, ramp_cap=15.0, c1=5000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true),
    (name="tight98_arb_terminal", load_cap=98.0, ramp_cap=10.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true),
    (name="as_plus_arb", load_cap=98.0, ramp_cap=10.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true),
    (name="cap101_arb", load_cap=101.0, ramp_cap=15.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true),
    (name="cap102_arb", load_cap=102.0, ramp_cap=15.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true),
    (name="cap102_as_arb", load_cap=102.0, ramp_cap=15.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true),
    (name="cap102_ramp12_as_arb", load_cap=102.0, ramp_cap=12.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true),
    (name="cap102_high_dvfs_as_arb", load_cap=102.0, ramp_cap=15.0, c1=15000.0, c2=10000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true),
    (name="tight98_arb_highpen", load_cap=98.0, ramp_cap=10.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true, slack=1000000.0, hard=false),
    (name="tight98_arb_hard", load_cap=98.0, ramp_cap=10.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.0, terminal=true, slack=10000.0, hard=true),
    (name="cap102_as_arb_highpen", load_cap=102.0, ramp_cap=15.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true, slack=1000000.0, hard=false),
    (name="cap102_as_arb_hard", load_cap=102.0, ramp_cap=15.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true, slack=10000.0, hard=true),
    (name="cap102_ramp12_as_arb_highpen", load_cap=102.0, ramp_cap=12.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true, slack=1000000.0, hard=false),
    (name="cap102_ramp12_as_arb_hard", load_cap=102.0, ramp_cap=12.0, c1=8000.0, c2=8000.0, cdeg=10.0, ncyc=1.5, espread=2.0, as_scale=1.5, terminal=true, slack=10000.0, hard=true),
]

if abspath(PROGRAM_FILE) == @__FILE__
    days = length(ARGS) > 0 ? parse.(Int, ARGS) : [21, 22, 23, 2, 18]

    println("day,config,obj_batt,obj_nb,value_add,max_dc_b,max_net_b,max_dc_nb,max_net_nb,unfinished_b,unfinished_nb,a_b_min,a_b_max,a_nb_min,a_nb_max,charge,discharge,reserve,frp_up,frp_down,load_slack_b,ramp_slack_b")
    for day in days
        base = generate_real_market_case_vcc(seed=42, T=24, use_real_load=true, day_index=day, quiet=true)
        for cfg in configs
            data = copy_with_params(base; load_cap=cfg.load_cap, ramp_cap=cfg.ramp_cap,
                                    c1=cfg.c1, c2=cfg.c2, cdeg=cfg.cdeg, ncyc=cfg.ncyc,
                                    energy_spread=cfg.espread, as_scale=cfg.as_scale)
            slack_penalty = haskey(cfg, :slack) ? cfg.slack : 10000.0
            allow_slack = haskey(cfg, :hard) ? !cfg.hard : true
            mb = solve_case(data; battery=true, terminal_soc=cfg.terminal,
                            slack_penalty=slack_penalty, allow_slack=allow_slack)
            mn = solve_case(data; battery=false, terminal_soc=false,
                            slack_penalty=slack_penalty, allow_slack=allow_slack)
            if !has_values(mb) || !has_values(mn)
                @printf("%d,%s,NO_VALUES\n", day, cfg.name)
                continue
            end
            xb = metrics(mb, data)
            xn = metrics(mn, data)
            @printf("%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f\n",
                    day, cfg.name, xb.obj, xn.obj, xb.obj - xn.obj,
                    xb.max_dc, xb.max_net, xn.max_dc, xn.max_net,
                    xb.unfinished, xn.unfinished, xb.amin, xb.amax, xn.amin, xn.amax,
                    xb.charge, xb.discharge, xb.reserve, xb.frp_up, xb.frp_down,
                    xb.load_slack, xb.ramp_slack)
        end
    end
end
