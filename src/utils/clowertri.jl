# SPDX-License-Identifier: MIT

# Half-vectorization of a matrix.
function vech(A::AbstractMatrix, k=0)
    @assert size(A, 1) == size(A, 2)

    D = size(A, 1)
    out = similar(A, Int((D-k) + ((D-k)^2 - (D-k)) / 2))
    i = 0
    for c = 1:(D-k)
        for r = (c+k):D
            @inbounds out[i += 1] = A[r,c]
        end
    end
    out
end

struct CompressedTriangularMatrix{T,S<:AbstractVector{T}} <: AbstractMatrix{T}
    D::Int64
    k::Int64
    vech::S
end

function CompressedTriangularMatrix(M::AbstractMatrix, k = 0)
    T = eltype(M)
    v = vech(M, k)
    CompressedTriangularMatrix{T, typeof(v)}(size(M, 1), k, v)
end

@Zygote.adjoint CompressedTriangularMatrix(D, k, v) =
    CompressedTriangularMatrix(D, k, v), c -> (0, 0, vech(c, k))

Base.size(M::CompressedTriangularMatrix) = (M.D, M.D)

function Base.getindex(M::CompressedTriangularMatrix, r::Int, c::Int)
    (r > M.k && c <= (M.D - M.k) && c <= (r - M.k)) || return zero(eltype(M))
    s = ((c-1) * c) ÷ 2
    i = (c-1) * (M.D - M.k) + (r - M.k) - s
    M.vech[i]
end

function Base.replace_in_print_matrix(M::CompressedTriangularMatrix, r::Integer, c::Integer,
        s::AbstractString)
    if (r > M.k && c <= (M.D - M.k) && c <= (r - M.k))
        return s
    else
        Base.replace_with_centered_mark(s)
    end
end

