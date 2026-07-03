"""
    fold_proximity(pm, u)

Returns `sqrt(det(g))` where `g` is the intrinsic metric. Values near zero indicate
that the embedding Jacobian is losing rank (fold proximity).
"""
function fold_proximity(pm::PullbackMetric, u::AbstractVector{<:Real})
    g = evaluate_metric_tensor(pm, u)
    return sqrt(max(det(Matrix(g)), 0.0))
end

"""
    gaussian_curvature_proxy(pm, u)

For a 2D intrinsic manifold, this returns a practical scalar proxy based on the
determinant of the Hessian of `log(det(g))`, useful for identifying high-curvature
regions in state-space scans.
"""
function gaussian_curvature_proxy(pm::PullbackMetric, u::AbstractVector{<:Real})
    length(u) == 2 || throw(ArgumentError("Curvature proxy currently supports 2D intrinsic coordinates."))

    f(v) = begin
        g = evaluate_metric_tensor(pm, v)
        d = max(det(Matrix(g)), eps(Float64))
        log(d)
    end

    H = ForwardDiff.hessian(f, u)
    return det(H)
end
