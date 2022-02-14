# SPDX-License-Identifier: MIT

abstract type AbstractJointNormal <: ExponentialFamilyDistribution end

struct JointNormal{T1<:AbstractMatrix,T2<:AbstractMatrix} <: AbstractJointNormal
    M::T1 # QxD mean
	lnΛ::T2 # QxD per-dimension log precision
end

function η(p::AbstractJointNormal)
    Λ = exp.(p.lnΛ)
    vcat(vec(Λ .* p.M), -(1/2)*vec(Λ))
end

function ξ(p::AbstractJointNormal, η)
	Q, D = size(p.M)
    Λ = -2 * reshape(η[Q*D+1:end], Q, D)
    M = (1 ./ Λ) .* reshape(η[1:Q*D], Q, D)
    vcat(vec(M), vec(log.(Λ)))
end

function unpack(p::AbstractJointNormal, μ)
	Q, D = size(p.M)
	XXᵀ = reshape(μ[Q*D+1:end], Q, D)
    X = reshape(μ[1:Q*D], Q, D)
	(X=X, XXᵀ=XXᵀ)
end

function A(p::AbstractJointNormal, η)
	Q, D = size(p.M)

    Λ = -2 * reshape(η[Q*D+1:end], Q, D)
    M = (1 ./ Λ) .* reshape(η[1:Q*D], Q, D)

    -(1/2) * sum(log.(Λ)) + (1/2)*sum(M .* Λ .* M)
end

function sample(p::AbstractJointNormal)
	Q, D = size(p.M)
    p.M + sqrt(1 ./ exp.(p.Λ)) .* randn(Q, D)
end
