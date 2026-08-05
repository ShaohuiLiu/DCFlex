"""
Cluster runner for optimal battery sizing with parameterized cycling constraint.

Usage:
    julia --project=. --threads=auto scripts/cluster_sizing_runner.jl <n_cyc_max> [n_days]

Examples:
    julia --project=. --threads=auto scripts/cluster_sizing_runner.jl 0.5
    julia --project=. --threads=auto scripts/cluster_sizing_runner.jl 1.0
    julia --project=. --threads=auto scripts/cluster_sizing_runner.jl 1.5
    julia --project=. --threads=auto scripts/cluster_sizing_runner.jl 2.0

Each invocation runs 31 days × 41 configs = 1,271 optimizations for a single
n_cyc_max value, saving results to a JLD2-compatible CSV for later aggregation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuMP, Printf, Statistics

# Include dependencies
include(joinpath(@__DIR__, "battery_sizing_helpers.jl"))

# ============================================================================
# Parse command-line argument
# ============================================================================

if length(ARGS) < 1
    error("Usage: julia cluster_sizing_runner.jl <n_cyc_max> [n_days] [terminal_soc_frac]")
end

const N_CYC_MAX = parse(Float64, ARGS[1])
const N_DAYS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 31
const TERMINAL_SOC_FRAC = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
const ENFORCE_TERMINAL_SOC = TERMINAL_SOC_FRAC > 0.0
const RESULT_DIR = joinpath(@__DIR__, "..", "results", "battery_sizing", "csv")

println("="^80)
println("  CLUSTER SIZING RUNNER — n_cyc_max = $N_CYC_MAX cycles/day")
if ENFORCE_TERMINAL_SOC
    println("  Terminal SOC constraint: ≥ $(TERMINAL_SOC_FRAC * 100)%")
end
println("="^80)
println("  Days: $N_DAYS | Threads: $(Threads.nthreads())")
println()

# ============================================================================
# Run simulation with threading per day
# ============================================================================

function run_cycling_simulation(n_cyc_max::Float64; n_days::Int=31)
    configs = generate_battery_configurations()
    n_configs = length(configs)
    total_runs = n_days * n_configs

    # Pre-allocate thread-safe storage
    # Each day gets its own vector (no contention between days)
    day_results = Vector{Vector{NamedTuple}}(undef, n_days)
    for d in 1:n_days
        day_results[d] = Vector{NamedTuple}(undef, n_configs)
    end

    t_start = time()
    completed = Threads.Atomic{Int}(0)

    # Parallelize over days (each day is independent)
    Threads.@threads for day_idx in 0:(n_days - 1)
        day_num = day_idx + 1
        seed = 42 + day_idx * 7

        for (ci, config) in enumerate(configs)
            result, model, data = run_sensitivity_case(config, seed, 24;
                                                        day_index=day_idx, quiet=true,
                                                        n_cyc_max_override=n_cyc_max,
                                                        enforce_terminal_soc=ENFORCE_TERMINAL_SOC,
                                                        terminal_soc_frac=TERMINAL_SOC_FRAC)

            # Extract extended metrics
            dvfs_dev = 0.0
            mean_dvfs = 1.0
            total_load = 0.0
            energy_cost = 0.0
            final_soc_frac = 0.0
            avg_energy_price = 0.0

            if !isnothing(model) && has_values(model)
                a_ref = data.cost.a_ref
                for t in data.horizon
                    a_val = value(model[:a][t])
                    dvfs_dev += abs(a_val - a_ref)
                    d_val = value(model[:D][t])
                    total_load += d_val * data.delta_t
                    energy_cost += data.pricing.lambda_energy[t] * d_val * data.delta_t
                end
                a_vals = [value(model[:a][t]) for t in data.horizon]
                mean_dvfs = mean(a_vals)
                avg_energy_price = mean(data.pricing.lambda_energy)

                # Extract final SOC for accounting correction
                if config.enable_battery
                    Tset = collect(data.horizon)
                    E_final = value(model[:E][Tset[end]])
                    final_soc_frac = E_final / data.bess.capacity
                else
                    final_soc_frac = 0.0
                end
            end

            day_results[day_num][ci] = (
                day = day_num,
                config_name = config.name,
                n_cyc_max = n_cyc_max,
                objective_value = result.objective_value,
                as_revenue = result.total_as_revenue,
                reserve_revenue = result.reserve_revenue,
                frp_up_revenue = result.frp_up_revenue,
                frp_down_revenue = result.frp_down_revenue,
                energy_cost = energy_cost,
                job_completion_rate = result.job_completion_rate,
                battery_cycles_base = result.battery_cycles_base,
                battery_cycles_expected = result.battery_cycles_expected,
                dvfs_deviation_sum = dvfs_dev,
                mean_dvfs_factor = mean_dvfs,
                total_load_mwh = total_load,
                final_soc_frac = final_soc_frac,
                avg_energy_price = avg_energy_price,
                status = result.termination_status,
            )

            Threads.atomic_add!(completed, 1)
        end

        elapsed = time() - t_start
        c = completed[]
        eta = c > 0 ? elapsed / c * (total_runs - c) / 60 : 0
        println("  Day $day_num done ($(c)/$total_runs, ETA: $(round(eta, digits=1))min)")
    end

    total_time = time() - t_start
    println("\n  Total: $(round(total_time / 60, digits=1)) min ($(completed[]) runs)")

    return day_results, configs
end

# ============================================================================
# Save results to CSV
# ============================================================================

function save_results_csv(day_results, output_path::String)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        # Header
        println(io, "day,config_name,n_cyc_max,objective_value,as_revenue,reserve_revenue,frp_up_revenue,frp_down_revenue,energy_cost,job_completion_rate,battery_cycles_base,battery_cycles_expected,dvfs_deviation_sum,mean_dvfs_factor,total_load_mwh,final_soc_frac,avg_energy_price,status")

        for day_vec in day_results
            for r in day_vec
                println(io, join([
                    r.day, r.config_name, r.n_cyc_max,
                    r.objective_value, r.as_revenue, r.reserve_revenue,
                    r.frp_up_revenue, r.frp_down_revenue, r.energy_cost,
                    r.job_completion_rate, r.battery_cycles_base, r.battery_cycles_expected,
                    r.dvfs_deviation_sum, r.mean_dvfs_factor, r.total_load_mwh,
                    r.final_soc_frac, r.avg_energy_price,
                    r.status
                ], ","))
            end
        end
    end

    println("  Results saved to: $output_path")
end

# ============================================================================
# Main
# ============================================================================

mkpath(RESULT_DIR)

day_results, configs = run_cycling_simulation(N_CYC_MAX, n_days=N_DAYS)

cyc_label = replace(string(N_CYC_MAX), "." => "p")
soc_label = ENFORCE_TERMINAL_SOC ? "_soc$(Int(TERMINAL_SOC_FRAC * 100))" : ""
csv_path = joinpath(RESULT_DIR, "sizing_ncyc_$(cyc_label)$(soc_label).csv")
save_results_csv(day_results, csv_path)

println("\n  DONE: n_cyc_max=$N_CYC_MAX → $csv_path")
