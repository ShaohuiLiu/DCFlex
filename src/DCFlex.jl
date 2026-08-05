"""
    DCFlex

Robust day-ahead co-optimization of data-center workloads, discrete DVFS, and
co-located battery storage under grid interconnection and ancillary-service
requirements.
"""
module DCFlex

using DataFrames
using Gurobi
using HiGHS
using JuMP
using LinearAlgebra
using Statistics
import MathOptInterface as MOI

export Job, BESSParameters, PricingData, CostParameters, InterconnectionLimits,
       Scenario, ProblemData, VCCParameters, ProblemDataVCC,
       build_model_vcc, solution_summary_vcc, validate!, validate_vcc!,
       mccormick_feasibility_check, mccormick_project_solution,
       build_model_vcc_mccormick_tightened, optimizer_factory

include("types.jl")
include("types_vcc.jl")
include("solver.jl")
include("model_vcc.jl")

end # module
