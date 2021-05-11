# SPDX-License-Identifier: MIT

abstract type Normal{D} <: AbstractModel end

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
    struct NormalIndependentParams{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution with independent mean μ and precision matrix Λ.
"""
struct NormalIndependentParams{D} <: Normal{D}
    μ::P where P <: AbstractParameter
    Λ::P where P <: AbstractParameter
end

function vectorize(m::Normal)
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end

abstract type NormalDiag{D} <: Normal{D} end

function statistics(::NormalDiag, X::AbstractMatrix{T}) where T
    vcat(X, hcat([vcat(x.^2, 1) for x in eachcol(X)]...))
end

"""
    struct NormalDiagIndependentParams{D} <: AbstractModel
        μ
        λ
    end

Normal distribution with independent mean μ and diagonal precision
matrix Λ with diagonal elements λ.
"""
struct NormalDiagIndependentParams{D} <: NormalDiag{D}
    μ::P where P <: AbstractParameter
    λ::P where P <: ParameterList
end

function vectorize(m::NormalDiagIndependentParams)
    Tλ = statistics.(m.λ)
    diagΛ = vcat([Tλᵢ[1] for Tλᵢ in Tλ]...)
    lnΛ = sum([Tλᵢ[2] for Tλᵢ in Tλ])
    μ, diag_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ)
    vcat(diagΛ .* μ, -.5 * diagΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end
