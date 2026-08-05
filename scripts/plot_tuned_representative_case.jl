using DCFlex
using CSV
using DataFrames
using JuMP
using Statistics
import MathOptInterface as MOI

include(joinpath(@__DIR__, "dvfs_contrast_scan.jl"))

function tuned_data(day::Int; load_cap=100.0, ramp_cap=15.0, phi=0.35,
                    job_scale=1.08, c1=15000.0, c2=15000.0)
    data = generate_real_market_case_vcc(seed=42, T=24, use_real_load=true,
                                         day_index=day, quiet=true)
    return clone_with_vcc(data; load_cap=load_cap, ramp_cap=ramp_cap,
                          c1=c1, c2=c2, cdeg=10.0, ncyc=1.5,
                          energy_spread=2.0, as_scale=1.5,
                          phi=phi, job_scale=job_scale)
end

function solve_tuned(data::ProblemDataVCC; battery::Bool)
    model = build_model_vcc(data; optimizer=optimizer_factory(),
                            enable_battery=battery,
                            enforce_terminal_soc=battery,
                            terminal_soc_frac=data.bess.soc_init,
                            allow_interconnection_slack=false)
    set_silent(model)
    set_time_limit_sec(model, 120.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)
    optimize!(model)
    return model
end

day = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 21
load_cap = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 100.0
ramp_cap = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 15.0
phi = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.35
job_scale = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.08
c1 = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 15000.0
c2 = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 15000.0

data = tuned_data(day; load_cap=load_cap, ramp_cap=ramp_cap,
                  phi=phi, job_scale=job_scale, c1=c1, c2=c2)
mb = solve_tuned(data; battery=true)
mn = solve_tuned(data; battery=false)

outdir = joinpath(@__DIR__, "..", "results", "representative")
mkpath(outdir)

Tset = collect(data.horizon)
vars_b = mb.ext[:dc_vars]
vars_n = mn.ext[:dc_vars]

has_values(mb) || error("Representative BESS solve returned no solution")
has_values(mn) || error("Representative no-BESS solve returned no solution")

fixed_b = [value(vars_b.P_fix[t]) for t in Tset]
jobs_b = [value(vars_b.job_power[t]) for t in Tset]
fixed_n = [value(vars_n.P_fix[t]) for t in Tset]
jobs_n = [value(vars_n.job_power[t]) for t in Tset]
charge = [value(vars_b.P_charge[t]) for t in Tset]
discharge = [value(vars_b.P_discharge[t]) for t in Tset]
reserve = [value(vars_b.P_reserve[t]) for t in Tset]
frp_up = [value(vars_b.P_frp_up[t]) for t in Tset]
frp_down = [value(vars_b.P_frp_down[t]) for t in Tset]

dispatch = DataFrame(
    hour=Tset,
    fixed_load_bess=fixed_b,
    schedulable_load_bess=jobs_b,
    dc_load_bess=fixed_b .+ jobs_b,
    net_load_bess=[value(vars_b.D[t]) for t in Tset],
    fixed_load_no_bess=fixed_n,
    schedulable_load_no_bess=jobs_n,
    dc_load_no_bess=fixed_n .+ jobs_n,
    net_load_no_bess=[value(vars_n.D[t]) for t in Tset],
    dvfs_bess=[value(vars_b.a[t]) for t in Tset],
    dvfs_no_bess=[value(vars_n.a[t]) for t in Tset],
    charge=charge,
    discharge=discharge,
    reserve=reserve,
    frp_up=frp_up,
    frp_down=frp_down,
    load_cap=fill(data.interconnection.load_cap, length(Tset)),
)
CSV.write(joinpath(outdir, "dispatch.csv"), dispatch)

b = data.bess
soc_scenarios = zeros(length(data.scenarios), length(Tset) + 1)
for (s, scenario) in enumerate(data.scenarios)
    energy = b.capacity * b.soc_init
    soc_scenarios[s, 1] = b.soc_init
    for (k, _) in enumerate(Tset)
        deployed_charge = charge[k] + scenario.beta_frp_up[k] * frp_up[k]
        deployed_discharge = discharge[k] + scenario.beta_reserve[k] * reserve[k] +
                             scenario.beta_frp_down[k] * frp_down[k]
        energy += (deployed_charge * b.eta_charge -
                   deployed_discharge / b.eta_discharge) * data.delta_t
        soc_scenarios[s, k + 1] = energy / b.capacity
    end
end

soc = DataFrame(
    hour=0:length(Tset),
    scheduled_soc=[b.soc_init; [value(vars_b.E[t]) / b.capacity for t in Tset]],
    scenario_mean=vec(mean(soc_scenarios, dims=1)),
    scenario_std=vec(std(soc_scenarios, dims=1)),
    soc_min=fill(b.soc_min, length(Tset) + 1),
    soc_max=fill(b.soc_max, length(Tset) + 1),
)
CSV.write(joinpath(outdir, "soc.csv"), soc)

summary = DataFrame(
    metric=["solver", "day", "with_bess_objective", "no_bess_objective", "value_added"],
    value=[DCFlex.solver_name(), string(day), string(objective_value(mb)),
           string(objective_value(mn)), string(objective_value(mb) - objective_value(mn))],
)
CSV.write(joinpath(outdir, "summary.csv"), summary)

println("Saved representative result tables to: ", outdir)
println("with_bess_objective=", objective_value(mb))
println("no_bess_objective=", objective_value(mn))
println("value_added=", objective_value(mb) - objective_value(mn))
