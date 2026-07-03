using GeometricBoundaryLayer

# Reuse local sibling data layout if available.
default_data_path = normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "sheba", "processed", "sheba_input.nc"))
fallback_gabls_path = normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "gabs3", "gabls3_scm_cabauw_obs_v33.nc"))
fallback_processed_path = normpath(joinpath(@__DIR__, "..", "..", "SpectralBL-Analytics", "data", "processed", "gabls3_scm_cabauw_obs_v33.nc"))
cli_path = isempty(ARGS) ? nothing : ARGS[1]

candidate_paths = String[]
if cli_path !== nothing
    push!(candidate_paths, normpath(cli_path))
end
append!(candidate_paths, [default_data_path, fallback_gabls_path, fallback_processed_path])

existing = filter(isfile, candidate_paths)

if isempty(existing)
    println("No local SHEBA/GABLS NetCDF file found.")
    println("Checked:")
    for p in candidate_paths
        println("  - ", p)
    end
    println("Provide your file path and run:")
    println("  samples = load_intrinsic_trajectory(\"/absolute/path/to/file.nc\")")
    exit(0)
end

path = first(existing)
println("Using data file: ", path)

samples = load_intrinsic_trajectory(path)
println("Loaded intrinsic samples: ", length(samples))

if isempty(samples)
    println("No valid intrinsic samples could be constructed from this dataset.")
    exit(0)
end

embedding = MostEmbedding()
ambient = KinematicFluxMetric([1.0, 0.25, 1.0, 1.0])
pm = PullbackMetric(embedding, ambient)

min_fold = Inf
argmin_u = [0.0, 0.0]
for s in samples
    u = [s.zeta, s.shear]
    fp = fold_proximity(pm, u)
    if fp < min_fold
        min_fold = fp
        argmin_u = u
    end
end

println("Minimum fold proximity along observed trajectory: ", min_fold)
println("At intrinsic coordinate u = ", argmin_u)
println("Metric condition number there: ", metric_condition_number(pm, argmin_u))

curve = trace_fold_curve(pm, argmin_u; ds=0.03, nsteps=30, target_det=0.05)
println("Continuation points traced: ", size(curve, 1))
