using Test
using DCFlex
using HiGHS
using JuMP
import MathOptInterface as MOI

include(joinpath(@__DIR__, "..", "scripts", "real_market_data_case_vcc.jl"))

@testset "DCFlex benchmark model" begin
    data = generate_real_market_case_vcc(
        seed=42,
        T=24,
        use_real_load=true,
        day_index=0,
        quiet=true,
    )

    @test validate_vcc!(data) === nothing
    @test length(data.base_load) == 24
    @test length(data.scenarios) == 50
    @test sum(job.work for job in data.jobs) == 340.0

    model = build_model_vcc(data; optimizer=HiGHS.Optimizer)
    set_time_limit_sec(model, 60.0)
    set_attribute(model, MOI.RelativeGapTolerance(), 0.01)
    set_silent(model)
    optimize!(model)

    @test termination_status(model) in (MOI.OPTIMAL, MOI.TIME_LIMIT)
    @test has_values(model)

    summary = solution_summary_vcc(model)
    @test isfinite(summary[:objective_value])
    @test size(summary[:jobs], 1) == length(data.jobs)
    @test size(summary[:bess], 1) == length(data.horizon) + 1
end
