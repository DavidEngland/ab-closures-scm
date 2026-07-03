"""
    MostEmbedding(; kwargs...)

Concrete 2D intrinsic -> 4D ambient embedding:
`u = (zeta, shear)` maps to `X = (zeta, Ri, Rw, Ro_l)`.

This map extends MOST through rational closures that permit fold-like geometry
in strongly stable regimes.
"""
struct MostEmbedding
    beta_m::Float64
    beta_h::Float64
    gamma_m::Float64
    gamma_h::Float64
    c_wave::Float64
    c_shear::Float64
    c_rossby::Float64
    eps::Float64
end

function MostEmbedding(; beta_m=4.7, beta_h=7.8, gamma_m=0.18, gamma_h=0.12,
    c_wave=0.3, c_shear=0.8, c_rossby=1.0, eps=1e-8)
    return MostEmbedding(beta_m, beta_h, gamma_m, gamma_h, c_wave, c_shear, c_rossby, eps)
end

# Rational forms create geometric nonlinearity and fold-prone regions near large zeta.
function phi_m(emb::MostEmbedding, zeta::Real)
    return 1 + emb.beta_m * zeta / (1 - emb.gamma_m * zeta)
end

function phi_h(emb::MostEmbedding, zeta::Real)
    return 1 + emb.beta_h * zeta / (1 - emb.gamma_h * zeta)
end

function (emb::MostEmbedding)(u::AbstractVector{<:Real})
    length(u) == 2 || throw(ArgumentError("Expected u = [zeta, shear]."))
    zeta = u[1]
    shear = u[2]

    pm = phi_m(emb, zeta)
    ph = phi_h(emb, zeta)

    # Local gradient Richardson surrogate from MOST-like closure.
    Ri = zeta * ph / (pm^2 + emb.eps)

    # Non-local wave/radiation surrogate and local Rossby alignment parameter.
    Rw = emb.c_wave * zeta - emb.c_shear * shear^2
    Ro_l = emb.c_rossby * shear / (1 + abs(zeta))

    return [zeta, Ri, Rw, Ro_l]
end
