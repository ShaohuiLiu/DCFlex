using DCFlex
using JuMP
using HiGHS
using Printf
using Statistics

include(joinpath(@__DIR__, "representative_day_tuning.jl"))

function clone_with_vcc(data::ProblemDataVCC; phi=data.vcc.phi,
                        hardware_limit=data.vcc.hardware_limit,
                        job_scale::Float64=1.0, kwargs...)
    tuned = copy_with_params(data; kwargs...)
    jobs = [
        Job(job.name, job.release, job.deadline, job.work * job_scale,
            job.rate_max, job.tardiness_weight)
        for job in tuned.jobs
    ]
    vcc = VCCParameters(hardware_limit, phi;
                        forecast_error_sigma=tuned.vcc.forecast_error_sigma,
                        epsilon_max=tuned.vcc.epsilon_max,
                        epsilon_min=tuned.vcc.epsilon_min,
                        enable_vcc_variable=tuned.vcc.enable_vcc_variable,
                        idle_resource_cost=tuned.vcc.idle_resource_cost)
    base_load_upper = [tuned.base_load[i] - vcc.hardware_limit * vcc.epsilon_min
                       for i in eachindex(tuned.base_load)]
    return ProblemDataVCC(tuned.horizon, tuned.delta_t, tuned.base_load, base_load_upper,
                          jobs, tuned.bess, tuned.pricing, tuned.cost,
                          tuned.interconnection, tuned.scenarios, vcc)
end

function score_case(day::Int, data::ProblemDataVCC; terminal::Bool=true)
    mb = solve_case(data; battery=true, terminal_soc=terminal, allow_slack=false)
    mn = solve_case(data; battery=false, terminal_soc=false, allow_slack=false)
    if !has_values(mb) || !has_values(mn)
        return nothing
    end

    xb = metrics(mb, data)
    xn = metrics(mn, data)
    vars_b = mb.ext[:dc_vars]
    vars_n = mn.ext[:dc_vars]
    Tset = collect(data.horizon)
    a_b = [value(vars_b.a[t]) for t in Tset]
    a_n = [value(vars_n.a[t]) for t in Tset]
    net_b = [value(vars_b.D[t]) for t in Tset]
    net_n = [value(vars_n.D[t]) for t in Tset]
    P_fix_b = [value(vars_b.P_fix[t]) for t in Tset]
    job_b = [value(vars_b.job_power[t]) for t in Tset]
    dc_b = P_fix_b .+ job_b
    diff = abs.(a_b .- a_n)
    return (
        day=day,
        value_add=xb.obj - xn.obj,
        dvfs_l1=sum(diff),
        dvfs_max=maximum(diff),
        dvfs_hours=sum(diff .> 0.02),
        a_b_min=minimum(a_b),
        a_b_max=maximum(a_b),
        a_n_min=minimum(a_n),
        a_n_max=maximum(a_n),
        max_dc_b=xb.max_dc,
        max_net_b=xb.max_net,
        max_net_n=xn.max_net,
        net_l1=sum(abs.(net_b .- net_n)),
        unfinished_b=xb.unfinished,
        unfinished_n=xn.unfinished,
        charge=xb.charge,
        discharge=xb.discharge,
        as_total=xb.reserve + xb.frp_up + xb.frp_down,
        obj_b=xb.obj,
        obj_n=xn.obj,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    days = length(ARGS) > 0 ? parse.(Int, ARGS) : [21]
    results = []

    for day in days
        base = generate_real_market_case_vcc(seed=42, T=24, use_real_load=true,
                                             day_index=day, quiet=true)
        for load_cap in [101.0, 102.0]
            for ramp_cap in [15.0]
                for phi in [0.25, 0.35]
                    for c1 in [15000.0, 30000.0]
                        for c2 in [8000.0, 15000.0]
                            for job_scale in [1.0, 1.08]
                                data = clone_with_vcc(base; load_cap=load_cap,
                                                      ramp_cap=ramp_cap,
                                                      c1=c1, c2=c2,
                                                      cdeg=10.0, ncyc=1.5,
                                                      energy_spread=2.0,
                                                      as_scale=1.5,
                                                      phi=phi,
                                                      job_scale=job_scale)
                                r = score_case(day, data)
                                isnothing(r) && continue
                                if r.discharge >= 8.0 && r.charge >= 8.0 &&
                                   r.as_total >= 100.0 &&
                                   r.max_dc_b >= load_cap + 0.5 &&
                                   r.unfinished_b <= 1e-3
                                    push!(results, merge(r, (load_cap=load_cap,
                                                             ramp_cap=ramp_cap,
                                                             phi=phi,
                                                             c1=c1, c2=c2,
                                                             job_scale=job_scale)))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    sort!(results, by=r -> (-r.dvfs_l1, -r.value_add))

    println("day,load_cap,ramp_cap,phi,c1,c2,job_scale,value_add,dvfs_l1,dvfs_max,dvfs_hours,a_b_range,a_nb_range,max_dc_b,max_net_b,max_net_nb,net_l1,unfinished_b,unfinished_nb,charge,discharge,as_total")
    for r in Iterators.take(results, 30)
        @printf("%d,%.0f,%.0f,%.2f,%.0f,%.0f,%.2f,%.2f,%.3f,%.3f,%d,%.3f-%.3f,%.3f-%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                r.day, r.load_cap, r.ramp_cap, r.phi, r.c1, r.c2, r.job_scale,
                r.value_add, r.dvfs_l1, r.dvfs_max, r.dvfs_hours,
                r.a_b_min, r.a_b_max, r.a_n_min, r.a_n_max,
                r.max_dc_b, r.max_net_b, r.max_net_n, r.net_l1,
                r.unfinished_b, r.unfinished_n, r.charge, r.discharge, r.as_total)
    end
end
