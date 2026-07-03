"""
    trace_fold_curve(pm, u0; kwargs...)

Pseudo-arc-length continuation for the fold condition
`F(u) = det(g(u)) - target_det = 0`, where `g(u)` is the intrinsic metric.

Returns an `N x 2` matrix of continuation points.
"""
function trace_fold_curve(
    pm::PullbackMetric,
    u0::AbstractVector{<:Real};
    ds::Float64=0.05,
    nsteps::Int=80,
    newton_max::Int=12,
    tol::Float64=1e-8,
    target_det::Float64=1e-4,
)
    length(u0) == 2 || throw(ArgumentError("trace_fold_curve currently supports 2D intrinsic coordinates."))

    F(u) = det(Matrix(evaluate_metric_tensor(pm, u))) - target_det

    # Find initial point on the constraint via scalar Newton along y with x fixed.
    u_curr = copy(Float64.(u0))
    u_curr[2] = solve_scalar_along_y(F, u_curr; tol=tol, maxiter=50)

    points = Matrix{Float64}(undef, nsteps + 1, 2)
    points[1, :] .= u_curr

    # Initial tangent from implicit relation F(x, y) = 0: y' = -Fx/Fy.
    tvec = normalized_tangent(F, u_curr)

    for k in 1:nsteps
        u_pred = u_curr .+ ds .* tvec
        u_next, ok = corrector_step(F, u_pred, u_curr, tvec; tol=tol, maxiter=newton_max)
        if !ok
            points = points[1:k, :]
            return points
        end

        points[k + 1, :] .= u_next

        # Secant update for tangent orientation continuity.
        tnew = u_next .- u_curr
        nt = norm(tnew)
        nt <= 0 && break
        tvec = tnew / nt

        u_curr = u_next
    end

    return points
end

function solve_scalar_along_y(F, u; tol=1e-8, maxiter=50)
    y = u[2]
    x = u[1]
    for _ in 1:maxiter
        f = F([x, y])
        abs(f) < tol && return y
        h = 1e-6 * (1 + abs(y))
        df = (F([x, y + h]) - F([x, y - h])) / (2h)
        abs(df) < eps(Float64) && break
        y -= f / df
    end
    return y
end

function normalized_tangent(F, u)
    h = 1e-6
    fx = (F([u[1] + h, u[2]]) - F([u[1] - h, u[2]])) / (2h)
    fy = (F([u[1], u[2] + h]) - F([u[1], u[2] - h])) / (2h)
    abs(fy) < eps(Float64) && return [1.0, 0.0]
    dydx = -fx / fy
    t = [1.0, dydx]
    return t / norm(t)
end

function corrector_step(F, u_pred, u_ref, tvec; tol=1e-8, maxiter=12)
    u = copy(u_pred)

    for _ in 1:maxiter
        r1 = F(u)
        r2 = dot(u .- u_pred, tvec)
        if norm([r1, r2]) < tol
            return u, true
        end

        h = 1e-6
        dFdx = (F([u[1] + h, u[2]]) - F([u[1] - h, u[2]])) / (2h)
        dFdy = (F([u[1], u[2] + h]) - F([u[1], u[2] - h])) / (2h)

        J = [dFdx dFdy; tvec[1] tvec[2]]
        rhs = -[r1, r2]

        if abs(det(J)) < 1e-14
            return u, false
        end

        du = J \ rhs
        u .+= du
    end

    return u, false
end
