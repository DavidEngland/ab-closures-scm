# GeometricBoundaryLayer.jl (prototype)

Minimal prototype for manifold-based ABL diagnostics:

- Pullback metric evaluation for an intrinsic stability manifold
- MOST-inspired embedding from intrinsic coordinates to `(zeta, Ri, Rw, Ro_l)`
- Fold proximity and curvature diagnostics for regime-transition detection

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/scan_fold_map.jl
```

## Notes

This initial implementation focuses on Stage 1 diagnostics and concrete metric geometry.
It is intentionally modular so continuation and DEC numerics can be added in follow-up modules.
