# SPDX-License-Identifier: MIT

abstract type AbstractNormal{D} <: AbstractModel end

function basemeasure(::AbstractNormal, X::AbstractMatrix)
    o = similar(X, size(X,2))
    fill!(o, -.5*log(2π))
end

function _quadratic_stats(x::AbstractVector, y::AbstractVector)
    xyᵀ = x*y'
    vcat(diag(xyᵀ), EFD.vec_tril(xyᵀ))
end

function statistics(m::AbstractNormal, X::AbstractMatrix)
    XX = hcat([_quadratic_stats(x, x) for x in eachcol(X)]...)
    o = similar(X, 1, size(X,2))
    fill!(o, 1)
    vcat(X, XX, o)
end

function loglikelihood(m::AbstractNormal, X::AbstractMatrix, cache = Dict())
    Tη = vectorize(m, cache)
    @cache cache TX = statistics(m, X)
    BX = basemeasure(m, X)
    TX' * Tη .+ BX
end

function ∇sum_loglikelihood(m::AbstractNormal, cache)
    ∇sum_loglikelihood(m, sum(eachcol(cache[:TX])), cache)
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

function Normal(T, D; W₀ = Matrix{T}(I, D, D), μ₀ = zeros(T, D),
                Σ₀ = Matrix{T}(I, D, D), σ = 0, pstrength = 1)

    μ = BayesianParameter(EFD.Normal(μ₀, T(pstrength) * Σ₀),
                          EFD.Normal(μ₀ .+ randn(T, D) .* T(σ), T(1/pstrength) * Σ₀))
    Λ = BayesianParameter(EFD.Wishart(W₀, D - 1 + T(pstrength)),
                          EFD.Wishart(W₀, D - 1 + T(pstrength)))
    NormalIndependentParams{D}(μ, Λ)
end
Normal(D; kwargs...) = Normal(Float64, D; kwargs...)

function vectorize(m::NormalIndependentParams, cache)
    TΛ = statistics(m.Λ)
    @cache cache diagΛ = TΛ[1]
    @cache cache trilΛ = TΛ[2]
    @cache cache lnΛ = TΛ[3]
    @cache cache Λ = EFD.matrix(diagΛ, trilΛ)

    Tμ = statistics(m.μ)
    @cache cache μ = Tμ[1]
    @cache cache diag_μμᵀ = Tμ[2]
    @cache cache tril_μμᵀ = Tμ[3]
    @cache cache μμᵀ = EFD.matrix(diag_μμᵀ, tril_μμᵀ)

    T = eltype(μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + T(2)*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -T(.5) * diagΛ, -trilΛ, -T(.5) * (tr_Λμμᵀ - lnΛ))
end

function ∇sum_loglikelihood(m::AbstractNormal, Tx::AbstractVector, cache)
    μ = cache[:μ]
    diag_μμᵀ = cache[:diag_μμᵀ]
    tril_μμᵀ = cache[:tril_μμᵀ]

    Λ = cache[:Λ]
    diagΛ = cache[:diagΛ]
    trilΛ = cache[:trilΛ]

    D = length(μ)
    O  = (D^2 - D) ÷ 2
    T = eltype(μ)
    x = view(Tx, 1:D)
    diag_xxᵀ = view(Tx, D+1:2*D)
    tril_xxᵀ = view(Tx, 2*D+1:2*D+O)
    C = Tx[end]

    ∂Tμ₁ = Λ * x
    ∂Tμ₂ = -vcat(T(.5)*diagΛ, trilΛ)*C
    ∂Tμ = vcat(∂Tμ₁, ∂Tμ₂)

    xμᵀ = x * μ'
    diag_xμᵀ = diag(xμᵀ)
    tril_xμᵀ = EFD.vec_tril(xμᵀ)

    ∂diagΛ = -T(.5)*(diag_xxᵀ - diag_xμᵀ + diag_μμᵀ)
    ∂trilΛ = -(tril_xxᵀ - tril_xμᵀ + tril_μμᵀ)
    ∂lnΛ = T(.5)*C
    ∂TΛ = vcat(∂diagΛ, ∂trilΛ, ∂lnΛ)

    Dict(m.μ => ∂Tμ, m.Λ => ∂TΛ)
end

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

function NormalDiag(T, D; β₀ = ones(T, D), μ₀ = zeros(T, D),
                    diagΣ₀ = ones(T, D), pstrength = 1, σ = 0)

    μ = BayesianParameter(EFD.NormalDiag(μ₀, T(1/pstrength) * diagΣ₀),
                          EFD.NormalDiag(μ₀ .+ randn(T, D) .* T(σ), T(1/pstrength) * diagΣ₀))
    λ = BayesianParameter(EFD.Gamma(pstrength * ones(T, D), T(pstrength) .* β₀),
                          EFD.Gamma(pstrength * ones(T, D), T(pstrength) .* β₀))
    NormalDiagIndependentParams{D}(μ, λ)
end
NormalDiag(D; kwargs...) = NormalDiag(Float64, D; kwargs...)

function vectorize(m::NormalDiagIndependentParams, cache)
    Tμ = statistics(m.μ)
    @cache cache μ = Tμ[1]
    @cache cache diag_μμᵀ = Tμ[2]

    Tλ = statistics(m.λ)
    @cache cache λ =  Tλ[1]
    lnλ =  Tλ[2]
    @cache cache logdetΛ = sum(lnλ)

    T = eltype(μ)
    tr_Λμμᵀ = dot(λ, diag_μμᵀ)
    vcat(λ .* μ, -T(.5) * λ, -T(.5) * (tr_Λμμᵀ - logdetΛ))
end

function ∇sum_loglikelihood(m::NormalDiagIndependentParams, Tx::AbstractVector, cache)
    μ = cache[:μ]
    diag_μμᵀ = cache[:diag_μμᵀ]

    λ = cache[:λ]
    logdetΛ = cache[:logdetΛ]

    D = length(μ)
    T = eltype(μ)
    x = view(Tx, 1:D)
    diag_xxᵀ = view(Tx, D+1:2*D)
    C = Tx[end]

    ∂Tμ₁ = λ .* x
    ∂Tμ₂ = -T(.5)*C*λ
    ∂Tμ = vcat(∂Tμ₁, ∂Tμ₂)

    diag_xμᵀ = x .* μ
    ∂diagΛ = -T(.5)*(diag_xxᵀ - diag_xμᵀ + diag_μμᵀ)
    ∂lnλ = similar(λ)
    fill!(∂lnλ, T(.5*C))
    ∂Tλ= vcat(∂diagΛ, ∂lnλ)

    Dict(m.μ => ∂Tμ, m.λ => ∂Tλ)
end
