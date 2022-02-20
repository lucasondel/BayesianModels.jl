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

struct CompressedLowerTriangular{T,S<:AbstractVector{T}} <: AbstractMatrix{T}
    D::Int64
    k::Int64
    vech::S
end

function CompressedLowerTriangular(M::AbstractMatrix, k = 0)
    T = eltype(M)
    v = vech(M, k)
    CompressedLowerTriangular{T, typeof(v)}(size(M, 1), k, v)
end

@Zygote.adjoint CompressedLowerTriangular(D, k, v) =
    CompressedLowerTriangular(D, k, v), c -> (0, 0, vech(c, k))

@Zygote.adjoint vech(M, k) =
    vech(M, k), c -> (CompressedLowerTriangular(size(M, 1), k, c), 0)

Base.size(M::CompressedLowerTriangular) = (M.D, M.D)

function Base.getindex(M::CompressedLowerTriangular, r::Int, c::Int)
    (r > M.k && c <= (M.D - M.k) && c <= (r - M.k)) || return zero(eltype(M))
    s = ((c-1) * c) ÷ 2
    i = (c-1) * (M.D - M.k) + (r - M.k) - s
    M.vech[i]
end

function Base.replace_in_print_matrix(M::CompressedLowerTriangular, r::Integer,
                                      c::Integer, s::AbstractString)
    if (r > M.k && c <= (M.D - M.k) && c <= (r - M.k))
        return s
    else
        Base.replace_with_centered_mark(s)
    end
end

struct CompressedSymmetric{T,S<:AbstractVector{T}} <: AbstractMatrix{T}
    D::Int64
    k::Int64
    vech::S
end

function CompressedSymmetric(M::AbstractMatrix, k = 0)
    T = eltype(M)
    v = vech(M, k)
    CompressedSymmetric{T, typeof(v)}(size(M, 1), k, v)
end

@Zygote.adjoint CompressedSymmetric(D, k, v) =
    CompressedSymmetric(D, k, v), c -> (0, 0, (1/2)*vech(Symmetric(c), k))

Base.size(M::CompressedSymmetric) = (M.D, M.D)

function Base.getindex(M::CompressedSymmetric, r::Int, c::Int)
    r, c = max(r, c), min(r, c)
    (r > M.k && c <= (M.D - M.k) && c <= (r - M.k)) || return zero(eltype(M))
    s = ((c-1) * c) ÷ 2
    i = (c-1) * (M.D - M.k) + (r - M.k) - s
    M.vech[i]
end

function Base.replace_in_print_matrix(M::CompressedSymmetric, r::Integer,
                                      c::Integer, s::AbstractString)
    r, c = max(r, c), min(r, c)
    if (r > M.k && c <= (M.D - M.k) && c <= (r - M.k))
        return s
    else
        Base.replace_with_centered_mark(s)
    end
end
