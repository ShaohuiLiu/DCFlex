using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const N_DAYS = parse(Int, get(ENV, "N_DAYS", "31"))

const EXPECTED_ROWS_PER_DAY = Dict(
    "study1_phi_sensitivity.csv" => 13,
    "study1b_fixed_schedulable.csv" => 16,
    "study2_battery_ablation.csv" => 5,
    "study3_extreme_cases.csv" => 10,
    "study4_ic_sensitivity.csv" => 16,
    "study4b_lp_duals.csv" => 24,
    "study4c_ic_composition.csv" => 48,
    "study5_oos_robustness.csv" => 1,
    "study6_dvfs_level_ablation.csv" => 5,
    "study7_load_composition.csv" => 28,
    "study8_stressed_bess_value.csv" => 34,
    "studyA_lp_relaxation.csv" => 6,
    "studyA_negprice_probe.csv" => 3,
    "studyB_min_as_participation.csv" => 20,
)

function verify_csv(path::String, expected_rows::Int)
    isfile(path) || error("Missing result file: $path")
    df = CSV.read(path, DataFrame)
    nrow(df) == expected_rows || error(
        "Unexpected row count in $path: got $(nrow(df)), expected $expected_rows")
    "day" in names(df) && length(unique(df.day)) == N_DAYS || error(
        "Unexpected day coverage in $path")
end

ablation_dir = joinpath(ROOT, "results", "ablation", "csv")
for (name, rows_per_day) in EXPECTED_ROWS_PER_DAY
    verify_csv(joinpath(ablation_dir, name), rows_per_day * N_DAYS)
end

sizing_dir = joinpath(ROOT, "results", "battery_sizing", "csv")
for label in ("0p5", "1p0", "1p5", "2p0")
    verify_csv(joinpath(sizing_dir, "sizing_ncyc_$(label).csv"), 41 * N_DAYS)
end

required_figures = [
    joinpath(ROOT, "results", "representative", "load.png"),
    joinpath(ROOT, "results", "representative", "battery.png"),
    joinpath(ROOT, "results", "figures", "fig_combined_value.pdf"),
    joinpath(ROOT, "results", "figures", "fig_optimal_cycling_combined.pdf"),
    joinpath(ROOT, "results", "figures", "fig_study4c_ic_composition.pdf"),
    joinpath(ROOT, "results", "figures", "fig_study8_stressed_bess_value.pdf"),
]
for path in required_figures
    isfile(path) || error("Missing generated figure: $path")
    filesize(path) > 0 || error("Generated figure is empty: $path")
end

println("Verified all result tables for $N_DAYS days and key manuscript figures.")
