import Random


function run_ransac(;
    xys_list::Array{Matrix{Float64}},
    estimate,
    is_inlier,
    sample_size::Int64,
    goal_inliers::Int64,
    max_iterations::Int64,
    stop_at_goal::Bool,
    random_seed::Int64,
)
    @assert length(xys_list) == 2
    best_model = undef
    best_ic = 0
    Random.seed!(random_seed)

    for i = 1:max_iterations
        sample_xys = Array{Matrix{Float64}}(undef, 2)
        @simd for j = 1:2
            sample_xys[j] = xys_list[j][Random.rand(1:size(xys_list[j])[1], sample_size), :]
        end
        m = estimate(sample_xys)
        coeffs_list = [(m[1], m[2], m[3]), (m[1], m[2], m[4])]
        ics = [0, 0]

        function check_inlier(line_idx)
            cur_check_func = xy -> is_inlier(xy, coeffs_list[line_idx])
            ics[line_idx] += count(x -> x, cur_check_func.(eachrow(xys_list[line_idx])))
        end

        @simd for j = 1:2
            check_inlier(j)
        end

        ic = sum(ics)

        if ic > best_ic
            best_ic = ic
            best_model = m
            if ic > goal_inliers && stop_at_goal
                break
            end
        end
    end

    return best_model, best_ic
end
