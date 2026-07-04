using GeometricBoundaryLayer
using LinearAlgebra

# Synthetic fallback for validating CSV export without external NetCDF files.
# We build pseudo snapshots in-memory and route them through the same DEC diagnostics path.

function synthetic_rows()
    mesh = DECMesh1D(24, 120.0; stretching=1.6)
    pm = PullbackMetric(MostEmbedding(), KinematicFluxMetric([1.0, 0.25, 1.0, 1.0]))

    U = 5 .+ 1.2 .* exp.(-mesh.z_primal ./ 40)
    V = fill(1.0, mesh.n_cells)
    Theta = 262 .+ 0.03 .* mesh.z_primal
    zeta_profile = fill(0.8, mesh.n_cells)

    rows = HistoricalDiagnosticsRow[]
    t0 = 0.0

    for k in 1:20
        U_prev = copy(U)
        step_diffusion!(U, V, Theta, mesh, pm, 5.0; zeta_profile=zeta_profile, K0_u=0.8, K0_v=0.8, K0_th=0.5)

        zeta_face, shear_face = intrinsic_faces(U, Theta, zeta_profile, mesh)
        cap = metric_capacity(pm, zeta_face, shear_face)

        for i in 1:mesh.n_cells
            zeta_i = zeta_profile[i]
            shear_i = abs(U[i] - (i == 1 ? U[i] : U[i - 1])) / (i == 1 ? mesh.cell_volume[1] : (mesh.z_primal[i] - mesh.z_primal[i - 1]))
            g = evaluate_metric_tensor(pm, [zeta_i, shear_i])

            push!(rows, HistoricalDiagnosticsRow(
                t0 + k * 5.0,
                mesh.z_primal[i],
                zeta_i,
                shear_i,
                det(Matrix(g)),
                metric_condition_number(pm, [zeta_i, shear_i]),
                cap[clamp(i, 1, end)],
                (U[i] - U_prev[i]) / 5.0,
            ))
        end
    end

    return rows
end

out_path = normpath(joinpath(@__DIR__, "outputs", "synthetic_historical_scm_diagnostics.csv"))
rows = synthetic_rows()
write_historical_diagnostics_csv(out_path, rows)

println("Synthetic diagnostics rows: ", length(rows))
println("CSV output: ", out_path)
