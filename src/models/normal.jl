# SPDX-License-Identifier: MIT

abstract type Normal{D} <: AbstractModel end

Base.show(io::IO, normal::Normal) = print(io, typeof(normal))

basemeasure(::Normal, X::AbstractMatrix{T}) where T =
    -.5*log(2π) * ones(T, size(X,2))

function _quadratic_stats(x::AbstractVector)
    xxᵀ = x*x'
    vcat(diag(xxᵀ), EFD.vec_tril(xxᵀ), 1)
end

function statistics(::Normal, X::AbstractMatrix{T}) where T
    vcat(X, hcat([_quadratic_stats(x) for x in eachcol(X)]...))
end

function loglikelihood(m::Normal, X::AbstractMatrix)
    Tη = vectorize(m)
    TX = statistics(m, X)
    TX' * Tη .+ basemeasure(m, X)
end

"""
    struct Normal{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution.
"""
struct NormalIndependentParams{D} <: Normal{D}
    μ::P where P <: BayesianParameter
    Λ::P where P <: BayesianParameter
end

function vectorize(m::Normal)
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end
