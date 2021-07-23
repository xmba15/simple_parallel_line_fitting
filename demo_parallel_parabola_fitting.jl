push!(LOAD_PATH, "$(@__DIR__)/src")
import LinearAlgebra
import Random
import Distributions
import PyPlot
import ParallelFitting


function create_parabola_points(;
    coeffs::Vector{Float64},
    t_range,
    noise_mean_std::Vector{Float64},
    random_seed::Int,
)::Matrix{Float64}
    @assert length(coeffs) == 3
    Random.seed!(random_seed)

    xys = zeros(eltype(t_range), length(t_range), 2)

    xys[:, 1] = t_range
    noise = Random.rand(Distributions.Normal(noise_mean_std...), size(t_range))
    @fastmath xys[:, 2] = (coeffs[1] * t_range .^ 2 + coeffs[2] * t_range) .+ coeffs[3] + noise

    return xys
end


function create_data()::Array{Matrix{Float64}}
    a = 0.5
    b = 1.0
    c0 = 5
    c1 = 30
    xys_list = Array{Matrix{Float64}}(undef, 2)
    xys_list[1] = create_parabola_points(
        coeffs = [a, b, c0],
        t_range = -15:0.8:15,
        noise_mean_std = [-0.4, 0.5],
        random_seed = 2021,
    )
    xys_list[2] = create_parabola_points(
        coeffs = [a, b, c1],
        t_range = -15:0.8:15,
        noise_mean_std = [-1.0, 1.5],
        random_seed = 2021,
    )

    return xys_list
end

function main()
    xys_list = create_data()
    max_iterations = 100
    sample_size = floor(Int, min(size(xys_list[1])[1], size(xys_list[2])[1]) * 0.3)
    inlier_thresh = 0.05

    @time best_model, _ = ParallelFitting.run_ransac(
        xys_list = xys_list,
        estimate = ParallelFitting.Parabola.estimate,
        is_inlier = (xy, coeffs) -> ParallelFitting.Parabola.is_inlier(xy, coeffs, inlier_thresh),
        sample_size = sample_size,
        goal_inliers = sample_size * 2,
        max_iterations = max_iterations,
        stop_at_goal = true,
        random_seed = 2021,
    )

    a, b, c1, c2 = best_model

    colors = ["pink", "blue"]
    for (xys, color) in zip(xys_list, colors)
        PyPlot.scatter(xys[:, 1], xys[:, 2], c = color)
    end

    functions = [x -> a * x^2 + b * x + c1, x -> a * x^2 + b * x + c2]
    labels = ["first_parabola", "second_parabola"]

    for (xys, func, label) in zip(xys_list, functions, labels)
        PyPlot.plot(xys[:, 1], func.(xys[:, 1]), label = label)
    end


    PyPlot.show()
end


main()
