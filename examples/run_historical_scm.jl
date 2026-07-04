using GeometricBoundaryLayer

function main()
    default_paths = [
        normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "processed", "gabls3_scm_cabauw_obs_v33.nc")),
        normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "gabs3", "gabls3_scm_cabauw_obs_v33.nc")),
        normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "sheba", "processed", "sheba_input.nc")),
    ]

    if !isempty(ARGS)
        pushfirst!(default_paths, normpath(ARGS[1]))
    end

    nc_candidates = unique(default_paths)
    nc_existing = filter(isfile, nc_candidates)

    if isempty(nc_existing)
        println("No NetCDF forcing file found for historical SCM run.")
        println("Checked:")
        for p in nc_candidates
            println("  - ", p)
        end
        println("Usage:")
        println("  julia --project=. examples/run_historical_scm.jl /absolute/path/to/profile.nc")
        return nothing
    end

    input_path = first(nc_existing)
    out_path = normpath(joinpath(@__DIR__, "outputs", "historical_scm_diagnostics.csv"))

    println("Using forcing file: ", input_path)
    rows = run_historical_scm(input_path; max_snapshots=120, nsteps_per_snapshot=2, dt=5.0)
    write_historical_diagnostics_csv(out_path, rows)

    println("Wrote diagnostics rows: ", length(rows))
    println("CSV output: ", out_path)

    return nothing
end

main()
