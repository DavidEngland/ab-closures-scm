using GeometricBoundaryLayer
using Statistics

n = 64
z_top = 200.0
mesh = DECMesh1D(n, z_top; stretching=1.8)

# Simple stable profile initialization.
U = 6 .+ 2 .* exp.(-mesh.z_primal ./ 50)
V = 1 .+ 0.5 .* exp.(-mesh.z_primal ./ 80)
Theta = 265 .+ 0.04 .* mesh.z_primal

embedding = MostEmbedding()
ambient = KinematicFluxMetric([1.0, 0.25, 1.0, 1.0])
pm = PullbackMetric(embedding, ambient)

dt = 5.0
nsteps = 20
for _ in 1:nsteps
    step_diffusion!(U, V, Theta, mesh, pm, dt; K0_u=0.8, K0_v=0.8, K0_th=0.5, coriolis_f=1e-4, Ug=6.0, Vg=1.0)
end

zeta_face, shear_face = intrinsic_faces(U, Theta, nothing, mesh)
cap = metric_capacity(pm, zeta_face, shear_face)

println("DEC demo complete")
println("U mean: ", mean(U), "  V mean: ", mean(V), "  Theta mean: ", mean(Theta))
println("capacity min/max: ", minimum(cap), " / ", maximum(cap))
