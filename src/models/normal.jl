# SPDX-License-Identifier: MIT

abstract type AbstractNormal{D} <: AbstractModel end

basemeasure(::AbstractNormal, X::AbstractMatrix{T}) where T =
    -.5*log(2π) * ones(T, size(X,2))

function _quadratic_stats(x::AbstractVector)
    xxᵀ = x*x'
    vcat(diag(xxᵀ), EFD.vec_tril(xxᵀ), 1)
end

function statistics(::AbstractNormal, X::AbstractMatrix{T}) where T
    vcat(X, hcat([_quadratic_stats(x) for x in eachcol(X)]...))
end

function loglikelihood(m::AbstractNormal, X::AbstractMatrix)
    Tη = vectorize(m)
    TX = statistics(m, X)
    BX = basemeasure(m, X)
    TX' * Tη .+ BX
end

"""
    struct NormalIndependentParams{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution with independent mean μ and precision matrix Λ.
"""
struct NormalIndependentParams{D} <: AbstractNormal{D}
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

function statistics(::AbstractNormalDiag, X::AbstractMatrix{T}) where T
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
struct NormalDiagIndependentParams{D} <: AbstractNormalDiag{D}
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

function NormalDiag(T, D; β₀ = ones(T, D), μ₀ = zeros(T, D),
                    diagΣ₀ = ones(T, D), pstrength = 1, σ = 0)

    μ = BayesianParameter(EFD.NormalDiag(μ₀, (1/pstrength) * diagΣ₀),
                          EFD.NormalDiag(μ₀ .+ randn(T, D) .* σ, (1/pstrength) * diagΣ₀))
    λ = Tuple(
          BayesianParameter(EFD.Gamma(pstrength, pstrength * β₀[i]),
                            EFD.Gamma(pstrength, pstrength * β₀[i]))
        for i in 1:D
    )
    NormalDiagIndependentParams{D}(μ, λ)
end
NormalDiag(D; kwargs...) = NormalDiag(Float64, D; kwargs...)
