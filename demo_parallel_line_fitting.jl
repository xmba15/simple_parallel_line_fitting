push!(LOAD_PATH, "$(@__DIR__)/src")
import LinearAlgebra
import Random
import Distributions
import PyPlot
import ParallelFitting


function create_line_points(;
    start::Array{Float64,2},
    direction::Array{Float64,2},
    t_range,
    noise_mean_std::Vector{Float64},
    random_seed::Int,
)::Matrix{Float64}
    @assert size(start) == (1, 2)
    @assert size(direction) == (1, 2)
    Random.seed!(random_seed)

    range_arr = hcat(t_range, t_range)
    points = start .+ (direction .* range_arr)
    noise = Random.rand(Distributions.Normal(noise_mean_std...), size(points))

    return points + noise
end


function create_data()::Array{Matrix{Float64}}
    direction = [3.2 6.7]
    direction /= LinearAlgebra.norm(direction) + 1e-10
    xys_list = Array{Matrix{Float64}}(undef, 2)
    xys_list[1] = create_line_points(;
        start = [20.0 10.0],
        direction = direction,
        t_range = 0:1.2:100,
        noise_mean_std = [-0.4, 3],
        random_seed = 2021,
    )
    xys_list[2] = create_line_points(
        start = [4 8.9],
        direction = direction,
        t_range = -50:1.0:50,
        noise_mean_std = [-1.0, 1.5],
        random_seed = 2022,
    )
    return xys_list
end


function main()
    xys_list = create_data()
    max_iterations = 100
    sample_size = floor(Int, min(size(xys_list[1])[1], size(xys_list[2])[1]) * 0.3)
    inlier_thresh = 0.5

    @time best_model, _ = ParallelFitting.run_ransac(
        xys_list = xys_list,
        estimate = ParallelFitting.Line.estimate,
        is_inlier = (xy, coeffs) -> ParallelFitting.Line.is_inlier(xy, coeffs, inlier_thresh),
        sample_size = sample_size,
        goal_inliers = sample_size * 2,
        max_iterations = max_iterations,
        stop_at_goal = true,
        random_seed = 2021,
    )

    n = best_model[begin:2]
    c = best_model[3:end]
    println("n $n")
    println("c $c")

    colors = ["pink", "blue"]
    for (xys, color) in zip(xys_list, colors)
        PyPlot.scatter(xys[:, 1], xys[:, 2], c = color)
    end

    functions = [x -> (-c[1] - n[1] * x) / n[2], x -> (-c[2] - n[1] * x) / n[2]]
    labels = ["first_line", "second_line"]

    for (xys, func, label) in zip(xys_list, functions, labels)
        PyPlot.plot(xys[:, 1], func.(xys[:, 1]), label = label)
    end

    PyPlot.show()
end


main()
