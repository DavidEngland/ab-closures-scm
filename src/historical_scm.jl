"""
    HistoricalDiagnosticsRow

Container for one exported diagnostic row.
"""
struct HistoricalDiagnosticsRow
    time::Float64
    height::Float64
    zeta::Float64
    shear::Float64
    metric_det::Float64
    condition_number::Float64
    capacity_damping::Float64
    u_tendency::Float64
end

"""
    run_historical_scm(path; kwargs...)

Run a metric-aware DEC single-column simulation driven by ingested profile
snapshots and return diagnostic rows suitable for CSV export.
"""
function run_historical_scm(
    path::String;
    n_cells::Int=64,
    z_top::Float64=250.0,
    stretching::Float64=1.8,
    dt::Float64=5.0,
    nsteps_per_snapshot::Int=1,
    max_snapshots::Union{Nothing,Int}=200,
    coriolis_f::Float64=1e-4,
    Ug::Float64=6.0,
    Vg::Float64=1.0,
    K0_u::Float64=0.8,
    K0_v::Float64=0.8,
    K0_th::Float64=0.5,
)
    mesh = DECMesh1D(n_cells, z_top; stretching=stretching)
    snapshots = load_profile_snapshots(path; max_steps=max_snapshots)
    isempty(snapshots) && throw(ArgumentError("No usable profile snapshots found in $(path)."))

    embedding = MostEmbedding()
    ambient = KinematicFluxMetric([1.0, 0.25, 1.0, 1.0])
    pm = PullbackMetric(embedding, ambient)

    U, Theta, zeta_profile = interpolate_snapshot_to_grid(first(snapshots), mesh.z_primal)
    V = fill(Vg, mesh.n_cells)

    rows = HistoricalDiagnosticsRow[]

    for snap in snapshots
        U_target, Theta_target, zeta_target = interpolate_snapshot_to_grid(snap, mesh.z_primal)

        # Light nudging toward observed profile keeps the SCM anchored to historical forcing.
        U .= 0.7 .* U .+ 0.3 .* U_target
        Theta .= 0.7 .* Theta .+ 0.3 .* Theta_target
        zeta_profile .= zeta_target

        U_prev = copy(U)
        for _ in 1:nsteps_per_snapshot
            step_diffusion!(
                U,
                V,
                Theta,
                mesh,
                pm,
                dt;
                zeta_profile=zeta_profile,
                K0_u=K0_u,
                K0_v=K0_v,
                K0_th=K0_th,
                coriolis_f=coriolis_f,
                Ug=Ug,
                Vg=Vg,
            )
        end

        zeta_face, shear_face = intrinsic_faces(U, Theta, zeta_profile, mesh)
        cap = metric_capacity(pm, zeta_face, shear_face)

        for i in 1:mesh.n_cells
            zeta_i = zeta_profile[i]
            shear_i = abs(U[i] - (i == 1 ? U[i] : U[i - 1])) / (i == 1 ? mesh.cell_volume[1] : (mesh.z_primal[i] - mesh.z_primal[i - 1]))
            u_state = [zeta_i, shear_i]

            g = evaluate_metric_tensor(pm, u_state)
            metric_det = det(Matrix(g))
            condg = metric_condition_number(pm, u_state)

            face_idx = clamp(i, 1, length(cap))
            cap_i = cap[face_idx]
            u_tendency = (U[i] - U_prev[i]) / (dt * nsteps_per_snapshot)

            push!(rows, HistoricalDiagnosticsRow(
                snap.time_value,
                mesh.z_primal[i],
                zeta_i,
                shear_i,
                metric_det,
                condg,
                cap_i,
                u_tendency,
            ))
        end
    end

    return rows
end

"""
    write_historical_diagnostics_csv(path, rows)

Write diagnostics table with columns:
`time,height,zeta,shear,metric_det,condition_number,capacity_damping,u_tendency`.
"""
function write_historical_diagnostics_csv(path::String, rows::Vector{HistoricalDiagnosticsRow})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "time,height,zeta,shear,metric_det,condition_number,capacity_damping,u_tendency")
        for r in rows
            println(io,
                string(
                    r.time, ",",
                    r.height, ",",
                    r.zeta, ",",
                    r.shear, ",",
                    r.metric_det, ",",
                    r.condition_number, ",",
                    r.capacity_damping, ",",
                    r.u_tendency,
                )
            )
        end
    end
    return path
end
