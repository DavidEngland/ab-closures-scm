# GeometricBoundaryLayer.jl (prototype)

Minimal prototype for manifold-based ABL diagnostics:

- Pullback metric evaluation for an intrinsic stability manifold
- MOST-inspired embedding from intrinsic coordinates to `(zeta, Ri, Rw, Ro_l)`
- Fold proximity and curvature diagnostics for regime-transition detection
- NetCDF ingestion into intrinsic trajectories `(zeta, shear)`
- Pseudo-arc-length continuation for fold-boundary tracing
- DEC-inspired 1D staggered mesh and implicit Crank-Nicolson diffusion stepper

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/scan_fold_map.jl
julia --project=. examples/trace_fold_curve.jl
julia --project=. examples/validate_sheba.jl /absolute/path/to/profile.nc
julia --project=. examples/dec_step_demo.jl
```

## DEC numerics API

```julia
using GeometricBoundaryLayer

mesh = DECMesh1D(64, 200.0; stretching=1.8)
U = fill(6.0, mesh.n_cells)
V = fill(1.0, mesh.n_cells)
Theta = 265 .+ 0.04 .* mesh.z_primal

pm = PullbackMetric(MostEmbedding(), KinematicFluxMetric([1.0, 0.25, 1.0, 1.0]))
step_diffusion!(U, V, Theta, mesh, pm, 5.0)
```

## Real-data ingestion API

```julia
using GeometricBoundaryLayer

samples = load_intrinsic_trajectory("/absolute/path/to/profile.nc")
U = samples_to_intrinsic_matrix(samples)  # rows are [zeta, shear]
```

This loader reuses variable-name fallback conventions from the sibling
`SpectralBL-Analytics` ingestion stack (e.g., `zh/zf/z`, `uw/u/U`, `zL/ZL`).

## Notes

This initial implementation focuses on Stage 1 diagnostics and concrete metric geometry.
It is intentionally modular so continuation and DEC numerics can be added in follow-up modules.
