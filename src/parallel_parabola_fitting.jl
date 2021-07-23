module Parabola


import LinearAlgebra
import Polynomials


export estimate, is_inlier


function least_square(A::Matrix{Float64}, b::Vector{Float64}, epsilon::Float64 = 1e-9)::Tuple{Vararg{Float64,4}}
    """
    solve Ax=b
    """
    height, width = size(A)

    U, S, V = LinearAlgebra.svd(A)
    y = zeros(min(height, width))
    z = U' * b
    k = 1

    while k <= min(height, width) && S[k] > epsilon
        y[k] = z[k] / S[k]
        k += 1
    end

    return Tuple(V * y)
end


function augment(xys_list::Array{Matrix{Float64}})
    @assert length(xys_list) == 2

    for i = 1:2
        @assert ndims(xys_list[i]) == 2 and size(xys_list[i])[1] == 2
    end

    b = [xys_list[1][:, 2]; xys_list[2][:, 2]]

    A1 = [xys_list[1][:, 1] .^ 2 xys_list[1][:, 1] ones(size(xys_list[1][:, 1])) zeros(size(xys_list[1][:, 1]))]
    A2 = [xys_list[2][:, 1] .^ 2 xys_list[2][:, 1] zeros(size(xys_list[2][:, 1])) ones(size(xys_list[2][:, 1]))]

    A = [A1; A2]  # Ax = b

    return A, b
end


function estimate(xys_list::Array{Matrix{Float64}})::Tuple{Vararg{Float64,4}}
    return least_square(augment(xys_list)...)
end


function is_inlier(xy, coeffs::Tuple{Vararg{Float64,3}}, threshold::Float64)::Bool
    """
    xy: point(x,y)
    coeffs: (a,b,c) where a*x**2+b*x+c=0 is the parabola equation
    """
    x0, y0 = xy
    a, b, c = coeffs
    cubic_eq_coeffs = [4 * a^2, 6 * a * b, 2 * (b^2 + 2 * (c - y0) * a + 1), 2 * b * (c - y0) - 2 * x0]
    roots = Polynomials.roots(Polynomials.Polynomial(reverse(cubic_eq_coeffs)))
    roots = [real(elem) for elem in roots if isreal(elem)]

    if length(roots) == 0
        return false
    end

    function squared_distance_func(x)
        return (
            a^2 * x^4 +
            2 * a * b * x^3 +
            (b^2 + 2 * (c - y0) * a + 1) * x^2 +
            (2 * b * (c - y0) - 2 * x0) * x +
            ((c - y0)^2 + x0^2)
        )
    end
    distances = squared_distance_func.(roots)
    sort!(distances)

    return distances[1] < threshold * threshold
end


end
