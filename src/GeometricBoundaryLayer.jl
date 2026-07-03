module GeometricBoundaryLayer

using LinearAlgebra
using ForwardDiff

include("metric_engine.jl")
include("most_embedding.jl")
include("diagnostics.jl")
include("data_ingestion.jl")
include("continuation.jl")
include("numerics.jl")
include("historical_scm.jl")

export AbstractAmbientMetric
export KinematicFluxMetric
export PullbackMetric
export evaluate_metric_tensor
export metric_condition_number

export MostEmbedding

export fold_proximity
export gaussian_curvature_proxy

export IntrinsicSample
export ProfileSnapshot
export load_intrinsic_trajectory
export load_profile_snapshots
export interpolate_snapshot_to_grid
export samples_to_intrinsic_matrix

export trace_fold_curve

export DECMesh1D
export exterior_derivative_0_to_1
export hodge_stars
export intrinsic_faces
export metric_capacity
export hodge_laplacian
export step_diffusion!

export HistoricalDiagnosticsRow
export run_historical_scm
export write_historical_diagnostics_csv

end # module
