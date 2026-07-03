using NCDatasets
using Dates

"""
    IntrinsicSample

Time-indexed intrinsic state sample for geometric diagnostics.
"""
struct IntrinsicSample
    source_file::String
    time_value::Float64
    zeta::Float64
    shear::Float64
    ri_local::Float64
end

"""
    ProfileSnapshot

One time-indexed vertical profile from a NetCDF file.
"""
struct ProfileSnapshot
    source_file::String
    time_value::Float64
    z::Vector{Float64}
    u::Vector{Float64}
    theta::Vector{Float64}
    zeta::Float64
end

"""
    load_intrinsic_trajectory(path; kwargs...)

Load a NetCDF profile file and construct intrinsic trajectory samples
`u = (zeta, shear)` with optional local Richardson estimate.

Variable lookup uses candidate lists and falls back gracefully when
some fields are unavailable.
"""
function load_intrinsic_trajectory(
    path::String;
    z_candidates::Vector{Symbol}=[:zh, :zf, :zt, :z, :height],
    u_candidates::Vector{Symbol}=[:uw, :u, :U],
    theta_candidates::Vector{Symbol}=[:theta_v, :thetav, :theta, :th, :T],
    zl_candidates::Vector{Symbol}=[:zL, :ZL, :zeta, :z_over_L],
    time_candidates::Vector{Symbol}=[:time, :Time],
)
    isfile(path) || throw(ArgumentError("Missing netCDF file: $(path)"))

    ds = NCDataset(path)
    try
        z_vals = read_variable_any(ds, z_candidates)
        u_vals = read_variable_any(ds, u_candidates)
        time_vals = read_variable_any(ds, time_candidates)

        z_mat = to_level_time_matrix(z_vals)
        u_mat = to_level_time_matrix(u_vals)
        t_vec = vec(to_float_array(time_vals))

        theta_mat = try_read_level_time_matrix(ds, theta_candidates)
        zl_vec = try_read_vector(ds, zl_candidates)

        n_time = minimum((size(z_mat, 2), size(u_mat, 2), length(t_vec)))
        out = IntrinsicSample[]

        for t in 1:n_time
            z_col = Vector{Float64}(z_mat[:, t])
            u_col = Vector{Float64}(u_mat[:, t])

            valid = isfinite.(z_col) .& isfinite.(u_col)
            count(valid) < 2 && continue

            z_valid = z_col[valid]
            u_valid = u_col[valid]
            z_sorted, u_sorted = sort_pairs(z_valid, u_valid)

            shear = bulk_shear(z_sorted, u_sorted)
            (!isfinite(shear) || shear <= 0) && continue

            zeta = compute_zeta(t, zl_vec, theta_mat, z_mat, u_mat)
            !isfinite(zeta) && continue

            ri_local = compute_local_ri(t, theta_mat, z_mat, u_mat)
            if !isfinite(ri_local)
                ri_local = NaN
            end

            push!(out, IntrinsicSample(path, normalize_time_value(t_vec[t]), zeta, shear, ri_local))
        end

        return out
    finally
        close(ds)
    end
end

"""
    samples_to_intrinsic_matrix(samples)

Convert samples into an `N x 2` matrix where each row is `[zeta, shear]`.
"""
function samples_to_intrinsic_matrix(samples::Vector{IntrinsicSample})
    out = Matrix{Float64}(undef, length(samples), 2)
    for i in eachindex(samples)
        out[i, 1] = samples[i].zeta
        out[i, 2] = samples[i].shear
    end
    return out
end

"""
    load_profile_snapshots(path; kwargs...)

Load a sequence of vertical profile snapshots from a NetCDF file.
Each snapshot contains sorted finite `(z, u, theta)` vectors and a scalar `zeta`.
"""
function load_profile_snapshots(
    path::String;
    z_candidates::Vector{Symbol}=[:zh, :zf, :zt, :z, :height],
    u_candidates::Vector{Symbol}=[:uw, :u, :U],
    theta_candidates::Vector{Symbol}=[:theta_v, :thetav, :theta, :th, :T],
    zl_candidates::Vector{Symbol}=[:zL, :ZL, :zeta, :z_over_L],
    time_candidates::Vector{Symbol}=[:time, :Time],
    max_steps::Union{Nothing,Int}=nothing,
)
    isfile(path) || throw(ArgumentError("Missing netCDF file: $(path)"))

    ds = NCDataset(path)
    try
        z_vals = read_variable_any(ds, z_candidates)
        u_vals = read_variable_any(ds, u_candidates)
        time_vals = read_variable_any(ds, time_candidates)

        z_mat = to_level_time_matrix(z_vals)
        u_mat = to_level_time_matrix(u_vals)
        t_vec = vec(to_float_array(time_vals))

        theta_mat = try_read_level_time_matrix(ds, theta_candidates)
        zl_vec = try_read_vector(ds, zl_candidates)

        n_time = minimum((size(z_mat, 2), size(u_mat, 2), length(t_vec)))
        if max_steps !== nothing
            n_time = min(n_time, max_steps)
        end

        out = ProfileSnapshot[]
        for t in 1:n_time
            z_col = Vector{Float64}(z_mat[:, t])
            u_col = Vector{Float64}(u_mat[:, t])

            if theta_mat === nothing || t > size(theta_mat, 2)
                th_col = fill(265.0, length(z_col))
            else
                th_col = Vector{Float64}(theta_mat[:, t])
            end

            valid = isfinite.(z_col) .& isfinite.(u_col) .& isfinite.(th_col)
            count(valid) < 2 && continue

            z_valid = z_col[valid]
            u_valid = u_col[valid]
            th_valid = th_col[valid]

            z_sorted, u_sorted, th_sorted = sort_triplets(z_valid, u_valid, th_valid)

            zeta = compute_zeta(t, zl_vec, theta_mat, z_mat, u_mat)
            if !isfinite(zeta)
                ri = compute_local_ri(t, theta_mat, z_mat, u_mat)
                zeta = isfinite(ri) ? max(ri, 0.0) : 0.0
            end

            push!(out, ProfileSnapshot(path, normalize_time_value(t_vec[t]), z_sorted, u_sorted, th_sorted, zeta))
        end

        return out
    finally
        close(ds)
    end
end

"""
    interpolate_snapshot_to_grid(snapshot, target_z)

Resample a profile snapshot to arbitrary vertical target heights using
piecewise-linear interpolation with clamped boundary extrapolation.
"""
function interpolate_snapshot_to_grid(snapshot::ProfileSnapshot, target_z::AbstractVector{<:Real})
    z_tgt = Float64.(target_z)
    u_interp = interpolate_profile_1d(snapshot.z, snapshot.u, z_tgt)
    th_interp = interpolate_profile_1d(snapshot.z, snapshot.theta, z_tgt)
    zeta_profile = fill(snapshot.zeta, length(z_tgt))
    return u_interp, th_interp, zeta_profile
end

function read_variable_any(ds::NCDataset, candidates::Vector{Symbol})
    for c in candidates
        name = String(c)
        if haskey(ds, name)
            return ds[name][:]
        end
    end
    throw(ArgumentError("Missing variable. Tried: $(join(String.(candidates), ", "))"))
end

function try_read_level_time_matrix(ds::NCDataset, candidates::Vector{Symbol})
    for c in candidates
        name = String(c)
        if haskey(ds, name)
            return to_level_time_matrix(ds[name][:])
        end
    end
    return nothing
end

function try_read_vector(ds::NCDataset, candidates::Vector{Symbol})
    for c in candidates
        name = String(c)
        if haskey(ds, name)
            return vec(to_float_array(ds[name][:]))
        end
    end
    return nothing
end

function to_float_array(values)
    raw = Array(values)
    return map(v -> ismissing(v) ? NaN : Float64(v), raw)
end

function to_level_time_matrix(values)
    arr = to_float_array(values)
    if ndims(arr) == 1
        return reshape(arr, :, 1)
    elseif ndims(arr) == 2
        return arr
    else
        throw(ArgumentError("Expected 1D or 2D profile variable, got rank $(ndims(arr))."))
    end
end

function sort_pairs(z::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    p = sortperm(z)
    z_sorted = Float64.(z[p])
    u_sorted = Float64.(u[p])

    dedup_z = Float64[]
    dedup_u = Float64[]

    i = 1
    while i <= length(z_sorted)
        zi = z_sorted[i]
        j = i
        acc = 0.0
        n = 0
        while j <= length(z_sorted) && z_sorted[j] == zi
            acc += u_sorted[j]
            n += 1
            j += 1
        end
        push!(dedup_z, zi)
        push!(dedup_u, acc / n)
        i = j
    end

    return dedup_z, dedup_u
end

function sort_triplets(
    z::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    th::AbstractVector{<:Real},
)
    p = sortperm(z)
    z_sorted = Float64.(z[p])
    u_sorted = Float64.(u[p])
    th_sorted = Float64.(th[p])

    dedup_z = Float64[]
    dedup_u = Float64[]
    dedup_th = Float64[]

    i = 1
    while i <= length(z_sorted)
        zi = z_sorted[i]
        j = i
        acc_u = 0.0
        acc_th = 0.0
        n = 0
        while j <= length(z_sorted) && z_sorted[j] == zi
            acc_u += u_sorted[j]
            acc_th += th_sorted[j]
            n += 1
            j += 1
        end
        push!(dedup_z, zi)
        push!(dedup_u, acc_u / n)
        push!(dedup_th, acc_th / n)
        i = j
    end

    return dedup_z, dedup_u, dedup_th
end

function interpolate_profile_1d(z::Vector{Float64}, values::Vector{Float64}, target_z::Vector{Float64})
    out = Vector{Float64}(undef, length(target_z))
    for (i, zt) in enumerate(target_z)
        if zt <= z[1]
            out[i] = values[1]
            continue
        elseif zt >= z[end]
            out[i] = values[end]
            continue
        end

        idx = searchsortedlast(z, zt)
        z_l = z[idx]
        z_r = z[idx + 1]
        v_l = values[idx]
        v_r = values[idx + 1]
        w = (zt - z_l) / (z_r - z_l)
        out[i] = (1.0 - w) * v_l + w * v_r
    end

    return out
end

function bulk_shear(z::Vector{Float64}, u::Vector{Float64})
    dz = z[end] - z[1]
    dz <= 0 && return NaN
    return abs(u[end] - u[1]) / dz
end

# Prefer explicit z/L variable when available; otherwise estimate from local Ri.
function compute_zeta(t::Int, zl_vec, theta_mat, z_mat, u_mat)
    if zl_vec !== nothing && t <= length(zl_vec) && isfinite(zl_vec[t])
        return zl_vec[t]
    end

    ri = compute_local_ri(t, theta_mat, z_mat, u_mat)
    if isfinite(ri)
        # Stable branch proxy mapping for diagnostics when direct z/L is absent.
        return max(ri, 0.0)
    end

    return NaN
end

function compute_local_ri(t::Int, theta_mat, z_mat, u_mat)
    theta_mat === nothing && return NaN
    t > size(theta_mat, 2) && return NaN
    t > size(z_mat, 2) && return NaN
    t > size(u_mat, 2) && return NaN

    z = Vector{Float64}(z_mat[:, t])
    u = Vector{Float64}(u_mat[:, t])
    th = Vector{Float64}(theta_mat[:, t])

    valid = isfinite.(z) .& isfinite.(u) .& isfinite.(th)
    count(valid) < 2 && return NaN

    z_valid = z[valid]
    u_valid = u[valid]
    th_valid = th[valid]

    p = sortperm(z_valid)
    z_sorted = z_valid[p]
    u_sorted = u_valid[p]
    th_sorted = th_valid[p]

    dz = z_sorted[end] - z_sorted[1]
    dz <= 0 && return NaN

    dudz = (u_sorted[end] - u_sorted[1]) / dz
    dthdz = (th_sorted[end] - th_sorted[1]) / dz
    abs(dudz) < eps(Float64) && return NaN

    th_ref = max(abs(th_sorted[1]), 200.0)
    g = 9.81

    return (g / th_ref) * dthdz / (dudz^2)
end

function normalize_time_value(t)
    if t isa Number
        return Float64(t)
    elseif t isa DateTime
        return Dates.datetime2unix(t)
    else
        s = string(t)
        dt = tryparse(DateTime, s)
        if dt !== nothing
            return Dates.datetime2unix(dt)
        end
        num = tryparse(Float64, s)
        if num !== nothing
            return num
        end
        throw(ArgumentError("Unsupported time value type $(typeof(t))"))
    end
end
