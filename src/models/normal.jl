# SPDX-License-Identifier: MIT

abstract type AbstractNormal{D} <: AbstractModel end

function basemeasure(::AbstractNormal, X::AbstractMatrix{T}) where T
    o = similar(X, size(X,2))
    fill!(o, -.5*log(2π))
end

function _quadratic_stats(x::AbstractVector)
    xxᵀ = x*x'
    vcat(diag(xxᵀ), EFD.vec_tril(xxᵀ), 1)
end

statistics(m::AbstractNormal, X::AbstractMatrix) = statistics(typeof(m), X)
statistics(::Type{<:AbstractNormal}, X::AbstractMatrix{T}) where T =
    vcat(X, hcat([_quadratic_stats(x) for x in eachcol(X)]...))

function loglikelihood(m::AbstractNormal, X::AbstractMatrix)
    Tη = vectorize(m)
    TX = statistics(m, X)
    BX = basemeasure(m, X)
    TX' * Tη .+ BX
end

"""
    mutable struct NormalIndependentParams{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution with independent mean μ and precision matrix Λ.
"""
mutable struct NormalIndependentParams{D} <: AbstractNormal{D}
    μ::P where P <: AbstractParameter
    Λ::P where P <: AbstractParameter
end

function vectorize(m::NormalIndependentParams)
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μμᵀ = EFD.matrix(diag_μμᵀ, tril_μμᵀ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end

function Normal(T, D; W₀ = Matrix{T}(I, D, D), μ₀ = zeros(T, D),
                Σ₀ = Matrix{T}(I, D, D), σ = 0, pstrength = 1)

    μ = BayesianParameter(EFD.Normal(μ₀, pstrength * Σ₀),
                          EFD.Normal(μ₀ .+ randn(T, D) .* σ, (1/pstrength) * Σ₀))
    Λ = BayesianParameter(EFD.Wishart(W₀, D - 1 + pstrength),
                          EFD.Wishart(W₀, D - 1 + pstrength))
    NormalIndependentParams{D}(μ, Λ)
end
Normal(D; kwargs...) = Normal(Float64, D; kwargs...)

abstract type AbstractNormalDiag{D} <: AbstractNormal{D} end

statistics(m::AbstractNormalDiag, X::AbstractMatrix) = statistics(typeof(m), X)
function statistics(::Type{<:AbstractNormalDiag}, X::AbstractMatrix{T}) where T
    o = similar(X, 1, size(X,2))
    fill!(o, 1)
    vcat(X, X.^2, o)
end

"""
    mutable struct NormalDiagIndependentParams{D} <: AbstractModel
        μ
        λ
    end

Normal distribution with independent mean μ and diagonal precision
matrix Λ with diagonal elements λ.
"""
mutable struct NormalDiagIndependentParams{D} <: AbstractNormalDiag{D}
    μ::P where P <: AbstractParameter
    λ::P where P <: AbstractParameter
end

function vectorize(m::NormalDiagIndependentParams)
    μ, diag_μμᵀ = statistics(m.μ)
    λ, lnλ = statistics(m.λ)
    logdetΛ = sum(lnλ)
    tr_Λμμᵀ = dot(λ, diag_μμᵀ)
    vcat(λ .* μ, -.5 * λ, -.5 * (tr_Λμμᵀ - logdetΛ))
end

function NormalDiag(T, D; β₀ = ones(T, D), μ₀ = zeros(T, D),
                    diagΣ₀ = ones(T, D), pstrength = 1, σ = 0)

    μ = BayesianParameter(EFD.NormalDiag(μ₀, (1/pstrength) * diagΣ₀),
                          EFD.NormalDiag(μ₀ .+ randn(T, D) .* σ, (1/pstrength) * diagΣ₀))
    λ = BayesianParameter(EFD.Gamma(pstrength * ones(T, D), pstrength .* β₀),
                          EFD.Gamma(pstrength * ones(T, D), pstrength .* β₀))
    NormalDiagIndependentParams{D}(μ, λ)
end
NormalDiag(D; kwargs...) = NormalDiag(Float64, D; kwargs...)
