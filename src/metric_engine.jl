abstract type AbstractAmbientMetric end

"""
    KinematicFluxMetric(scales)

Diagonal ambient metric for mixed ABL coordinates.
`scales` is a 4-vector with characteristic magnitudes for `(zeta, Ri, Rw, Ro_l)`.
"""
struct KinematicFluxMetric <: AbstractAmbientMetric
    scales::NTuple{4,Float64}
end

function KinematicFluxMetric(scales::AbstractVector{<:Real})
    length(scales) == 4 || throw(ArgumentError("Expected 4 scales for (zeta, Ri, Rw, Ro_l)."))
    return KinematicFluxMetric((Float64.(scales)...,))
end

function (metric::KinematicFluxMetric)(::AbstractVector)
    return Diagonal(1.0 ./ (collect(metric.scales) .^ 2))
end

"""
    PullbackMetric(manifold_map, ambient_metric)

Container for a manifold embedding map `u -> X(u)` and ambient metric `G`.
"""
struct PullbackMetric{M,AM<:AbstractAmbientMetric}
    manifold_map::M
    ambient_metric::AM
end

"""
    evaluate_metric_tensor(pm, u)

Compute intrinsic metric: `g = J' * G * J`, where `J = dX/du`.
"""
function evaluate_metric_tensor(pm::PullbackMetric, u::AbstractVector{<:Real})
    J = ForwardDiff.jacobian(pm.manifold_map, u)
    X = pm.manifold_map(u)
    G = pm.ambient_metric(X)
    return Symmetric(J' * G * J)
end

"""
    metric_condition_number(pm, u)

Condition number of the intrinsic metric. Large values indicate coordinate folding
or near-loss of regularity in the embedding.
"""
function metric_condition_number(pm::PullbackMetric, u::AbstractVector{<:Real})
    g = evaluate_metric_tensor(pm, u)
    return cond(Matrix(g))
end
