const _SELECTED_SOLVER = Ref{Symbol}(:unset)
const _GUROBI_ENVS = Dict{Int,Any}()
const _GUROBI_ENV_LOCK = ReentrantLock()

function _gurobi_env()
    tid = Threads.threadid()
    return lock(_GUROBI_ENV_LOCK) do
        get!(_GUROBI_ENVS, tid) do
            Gurobi.Env(output_flag=0)
        end
    end
end

_gurobi_optimizer() = Gurobi.Optimizer(_gurobi_env())

"""Return the configured solver name (`Gurobi` or `HiGHS`).

Set `DCFLEX_SOLVER` to `gurobi`, `highs`, or `auto` (the default). In auto
mode, Gurobi 13 is used when a valid license is available; otherwise the
open-source HiGHS solver is used.
"""
function solver_name()
    if _SELECTED_SOLVER[] == :unset
        requested = lowercase(get(ENV, "DCFLEX_SOLVER", "auto"))
        requested in ("auto", "gurobi", "highs") || error(
            "DCFLEX_SOLVER must be one of: auto, gurobi, highs (got '$requested')")

        if requested == "highs"
            _SELECTED_SOLVER[] = :highs
        else
            try
                _gurobi_env()
                _SELECTED_SOLVER[] = :gurobi
            catch err
                requested == "gurobi" && error(
                    "Gurobi was requested but no usable Gurobi 13 license was found. " *
                    "Configure a license or rerun with DCFLEX_SOLVER=highs. " *
                    "Original error: $(sprint(showerror, err))")
                @warn "Gurobi is unavailable; using HiGHS" exception=(err, catch_backtrace())
                _SELECTED_SOLVER[] = :highs
            end
        end
    end

    return _SELECTED_SOLVER[] == :gurobi ? "Gurobi 13" : "HiGHS"
end

"""Return a JuMP-compatible optimizer factory for the configured solver."""
function optimizer_factory()
    solver_name()
    return _SELECTED_SOLVER[] == :gurobi ? _gurobi_optimizer : HiGHS.Optimizer
end
