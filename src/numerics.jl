using SparseArrays

"""
    DECMesh1D(n_cells, z_primal, z_dual, cell_volume, face_spacing)

1D staggered mesh for DEC-inspired SCM numerics.
- 0-forms are stored at primal cell centers (`z_primal`, length `n_cells`).
- 1-forms/fluxes are stored at dual faces (`z_dual`, length `n_cells + 1`).
"""
struct DECMesh1D
    n_cells::Int
    z_primal::Vector{Float64}
    z_dual::Vector{Float64}
    cell_volume::Vector{Float64}
    face_spacing::Vector{Float64}
end

"""
    DECMesh1D(n_cells, z_top; stretching=1.4)

Build a vertically stretched mesh. `stretching > 1` clusters levels near the surface.
"""
function DECMesh1D(n_cells::Int, z_top::Float64; stretching::Float64=1.4)
    n_cells >= 3 || throw(ArgumentError("n_cells must be >= 3."))
    z_top > 0 || throw(ArgumentError("z_top must be positive."))
    stretching > 0 || throw(ArgumentError("stretching must be positive."))

    eta_edges = collect(range(0.0, 1.0; length=n_cells + 1))
    z_edges = z_top .* (eta_edges .^ stretching)

    z_primal = @. 0.5 * (z_edges[1:end-1] + z_edges[2:end])
    z_dual = copy(z_edges)

    cell_volume = diff(z_dual)

    face_spacing = Vector{Float64}(undef, n_cells + 1)
    face_spacing[1] = cell_volume[1]
    for i in 2:n_cells
        face_spacing[i] = z_primal[i] - z_primal[i - 1]
    end
    face_spacing[end] = cell_volume[end]

    return DECMesh1D(n_cells, z_primal, z_dual, cell_volume, face_spacing)
end

"""
    exterior_derivative_0_to_1(mesh)

Discrete exterior derivative mapping 0-forms (cell centers) to 1-forms (faces).
Boundary rows are zero to encode no-flux boundaries by default.
"""
function exterior_derivative_0_to_1(mesh::DECMesh1D)
    n = mesh.n_cells

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for face in 2:n
        dz = mesh.z_primal[face] - mesh.z_primal[face - 1]
        push!(rows, face); push!(cols, face - 1); push!(vals, -1.0 / dz)
        push!(rows, face); push!(cols, face);     push!(vals,  1.0 / dz)
    end

    return sparse(rows, cols, vals, n + 1, n)
end

"""
    hodge_stars(mesh)

Return `(star0, star0_inv, star1)` with:
- `star0`: maps primal 0-forms to dual 1-cells via cell volumes
- `star1`: face metric for 1-forms (identity baseline here)
"""
function hodge_stars(mesh::DECMesh1D)
    star0 = spdiagm(0 => mesh.cell_volume)
    star0_inv = spdiagm(0 => 1.0 ./ mesh.cell_volume)
    star1 = spdiagm(0 => ones(mesh.n_cells + 1))
    return star0, star0_inv, star1
end

"""
    intrinsic_faces(U, Theta, zeta_profile, mesh)

Estimate face-level intrinsic coordinates `(zeta, shear)`.
If `zeta_profile` is provided at cell centers, face values are interpolated from it.
Otherwise `zeta` is estimated from a local Richardson proxy.
"""
function intrinsic_faces(
    U::AbstractVector{<:Real},
    Theta::AbstractVector{<:Real},
    zeta_profile::Union{Nothing,AbstractVector{<:Real}},
    mesh::DECMesh1D,
)
    n = mesh.n_cells
    length(U) == n || throw(ArgumentError("U must have length n_cells."))
    length(Theta) == n || throw(ArgumentError("Theta must have length n_cells."))
    zeta_profile !== nothing && length(zeta_profile) != n && throw(ArgumentError("zeta_profile must be length n_cells."))

    shear = zeros(Float64, n + 1)
    for face in 2:n
        dz = mesh.z_primal[face] - mesh.z_primal[face - 1]
        shear[face] = abs((U[face] - U[face - 1]) / dz)
    end
    shear[1] = shear[2]
    shear[end] = shear[end - 1]

    zeta = zeros(Float64, n + 1)
    if zeta_profile === nothing
        g = 9.81
        for face in 2:n
            dz = mesh.z_primal[face] - mesh.z_primal[face - 1]
            dudz = (U[face] - U[face - 1]) / dz
            dthdz = (Theta[face] - Theta[face - 1]) / dz
            th_ref = max(abs(0.5 * (Theta[face] + Theta[face - 1])), 200.0)
            ri = (g / th_ref) * dthdz / (dudz^2 + eps(Float64))
            zeta[face] = max(ri, 0.0)
        end
        zeta[1] = zeta[2]
        zeta[end] = zeta[end - 1]
    else
        zeta[1] = Float64(zeta_profile[1])
        zeta[end] = Float64(zeta_profile[end])
        for face in 2:n
            zeta[face] = 0.5 * (Float64(zeta_profile[face - 1]) + Float64(zeta_profile[face]))
        end
    end

    return zeta, shear
end

"""
    metric_capacity(pm, zeta_face, shear_face; kappa_alpha=0.02, cap_min=0.05, cap_max=1.0)

Convert metric condition number into transport capacity multipliers on faces.
Large condition numbers reduce transport capacity near folds.
"""
function metric_capacity(
    pm::PullbackMetric,
    zeta_face::AbstractVector{<:Real},
    shear_face::AbstractVector{<:Real};
    kappa_alpha::Float64=0.02,
    cap_min::Float64=0.05,
    cap_max::Float64=1.0,
)
    nfaces = length(zeta_face)
    length(shear_face) == nfaces || throw(ArgumentError("zeta_face and shear_face lengths must match."))

    cap = zeros(Float64, nfaces)
    for i in eachindex(cap)
        u = [Float64(zeta_face[i]), Float64(shear_face[i])]
        κ = metric_condition_number(pm, u)
        c = 1.0 / (1.0 + kappa_alpha * max(κ - 1.0, 0.0))
        cap[i] = clamp(c, cap_min, cap_max)
    end

    return cap
end

"""
    hodge_laplacian(mesh, K_face)

Build DEC-inspired diffusion operator `L` over 0-forms.
`K_face` is the face eddy diffusivity/capacity vector (`n_cells + 1`).
"""
function hodge_laplacian(mesh::DECMesh1D, K_face::AbstractVector{<:Real})
    n = mesh.n_cells
    length(K_face) == n + 1 || throw(ArgumentError("K_face must have length n_cells + 1."))

    D01 = exterior_derivative_0_to_1(mesh)
    _, star0_inv, _ = hodge_stars(mesh)
    Kstar1 = spdiagm(0 => Float64.(K_face))

    # L = -⋆0^{-1} d^T (K ⋆1) d  (with ⋆1 absorbed into Kstar1)
    return -star0_inv * transpose(D01) * Kstar1 * D01
end

"""
    step_diffusion!(U, V, Theta, mesh, pm, dt; kwargs...)

Advance one implicit Crank-Nicolson step with metric-aware DEC diffusion and
optional Coriolis coupling.
"""
function step_diffusion!(
    U::Vector{Float64},
    V::Vector{Float64},
    Theta::Vector{Float64},
    mesh::DECMesh1D,
    pm::PullbackMetric,
    dt::Float64;
    zeta_profile::Union{Nothing,AbstractVector{<:Real}}=nothing,
    K0_u::Float64=1.0,
    K0_v::Float64=1.0,
    K0_th::Float64=0.7,
    coriolis_f::Float64=0.0,
    Ug::Float64=0.0,
    Vg::Float64=0.0,
)
    n = mesh.n_cells
    length(U) == n || throw(ArgumentError("U length must match mesh.n_cells."))
    length(V) == n || throw(ArgumentError("V length must match mesh.n_cells."))
    length(Theta) == n || throw(ArgumentError("Theta length must match mesh.n_cells."))
    dt > 0 || throw(ArgumentError("dt must be positive."))

    zeta_face, shear_face = intrinsic_faces(U, Theta, zeta_profile, mesh)
    cap = metric_capacity(pm, zeta_face, shear_face)

    L_u = hodge_laplacian(mesh, K0_u .* cap)
    L_v = hodge_laplacian(mesh, K0_v .* cap)
    L_t = hodge_laplacian(mesh, K0_th .* cap)

    I0 = spdiagm(0 => ones(n))

    A_u = I0 - 0.5 * dt * L_u
    B_u = I0 + 0.5 * dt * L_u

    A_v = I0 - 0.5 * dt * L_v
    B_v = I0 + 0.5 * dt * L_v

    A_t = I0 - 0.5 * dt * L_t
    B_t = I0 + 0.5 * dt * L_t

    rhs_u = B_u * U
    rhs_v = B_v * V

    if coriolis_f != 0.0
        rhs_u .+= dt .* (coriolis_f .* (V .- Vg))
        rhs_v .+= dt .* (-coriolis_f .* (U .- Ug))
    end

    rhs_t = B_t * Theta

    U .= A_u \ rhs_u
    V .= A_v \ rhs_v
    Theta .= A_t \ rhs_t

    return U, V, Theta
end
