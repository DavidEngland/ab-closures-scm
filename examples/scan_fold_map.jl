using GeometricBoundaryLayer

function main()
    embedding = MostEmbedding()
    ambient = KinematicFluxMetric([1.0, 0.25, 1.0, 1.0])
    pm = PullbackMetric(embedding, ambient)

    zeta_vals = range(0.01, 4.5; length=60)
    shear_vals = range(0.01, 2.0; length=60)

    min_fold = Inf
    argmin_u = [0.0, 0.0]

    for zeta in zeta_vals, shear in shear_vals
        u = [zeta, shear]
        fp = fold_proximity(pm, u)
        if fp < min_fold
            min_fold = fp
            argmin_u = u
        end
    end

    println("Minimum fold proximity found: ", min_fold)
    println("At intrinsic coordinate u = ", argmin_u)
    println("Metric condition number there: ", metric_condition_number(pm, argmin_u))
    println("Curvature proxy there: ", gaussian_curvature_proxy(pm, argmin_u))
end

main()
