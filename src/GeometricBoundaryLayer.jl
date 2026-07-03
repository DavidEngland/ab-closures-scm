module GeometricBoundaryLayer

using LinearAlgebra
using ForwardDiff

include("metric_engine.jl")
include("most_embedding.jl")
include("diagnostics.jl")
include("data_ingestion.jl")
include("continuation.jl")

export AbstractAmbientMetric
export KinematicFluxMetric
export PullbackMetric
export evaluate_metric_tensor
export metric_condition_number

export MostEmbedding

export fold_proximity
export gaussian_curvature_proxy

export IntrinsicSample
export load_intrinsic_trajectory
export samples_to_intrinsic_matrix

export trace_fold_curve

end # module
