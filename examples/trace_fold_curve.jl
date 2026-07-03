using GeometricBoundaryLayer

embedding = MostEmbedding()
ambient = KinematicFluxMetric([1.0, 0.25, 1.0, 1.0])
pm = PullbackMetric(embedding, ambient)

u0 = [4.5, 0.01]
curve = trace_fold_curve(pm, u0; ds=0.04, nsteps=50, target_det=0.05)

println("Traced fold curve with ", size(curve, 1), " points.")
println("First point: ", curve[1, :])
println("Last point: ", curve[end, :])
