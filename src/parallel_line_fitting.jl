module Line


import LinearAlgebra


export estimate, is_inlier


function least_square(A::Matrix{Float64}, dim::Int)::Tuple{Vararg{Float64,4}}
    """
    solve Ax=0 with normalization constraint on x
    """
    height, width = size(A)

    @assert width >= dim + 1 "not enough unknowns"
    @assert height >= dim "not enough equations"

    R = LinearAlgebra.qr(A).R
    V = LinearAlgebra.svd(R[width-dim+1:end, width-dim+1:end]).V
    n = V[:, dim]
    c = -LinearAlgebra.inv(R[1:(width-dim), 1:(width-dim)]) * (R[1:(width-dim), (width-dim+1):width] * n)

    # n0*x+n1*y+c0=0
    # n0*x+n1*y+c1=0
    # n0*n0+n1*n1=1
    return n..., c...
end


function augment(xys_list::Array{Matrix{Float64}})::Matrix{Float64}
    @assert length(xys_list) == 2

    for i = 1:2
        @assert ndims(xys_list[i]) == 2 and size(xys_list[i])[1] == 2
    end

    A1 = [ones(size(xys_list[1][:, 1])) zeros(size(xys_list[1][:, 1])) xys_list[1]]
    A2 = [zeros(size(xys_list[2][:, 1])) ones(size(xys_list[2][:, 1])) xys_list[2]]

    return [A1; A2]  # Ax = 0
end


function estimate(xys_list::Array{Matrix{Float64}})::Tuple{Vararg{Float64,4}}
    return least_square(augment(xys_list), 2)
end


function is_inlier(xy, coeffs::Tuple{Vararg{Float64,3}}, threshold::Float64)::Bool
    """
    xy: point(x,y)
    coeffs: (n0,n1,c) where n0*x+n1*y+c=0 is the line equation
    """
    x0, y0 = xy
    n0, n1, c = coeffs
    return abs(x0 * n0 + y0 * n1 + c) < threshold
end


end
