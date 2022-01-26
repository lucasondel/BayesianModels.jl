# SPDX-License-Identifier: MIT

abstract type AbstractNormal <: ExponentialFamilyDistribution end

struct Normal{T1<:AbstractVector,T2<:AbstractMatrix} <: AbstractNormal
    μ::T1
	Λ::T2
end

function η(p::AbstractNormal)
	Λ = Symmetric(p.Λ)
	vcat(Λ*p.μ, -(1/2)*vec(Λ))
end

function ξ(p::AbstractNormal, η)
	D = length(p.μ)
	Λ = -2*reshape(η[D+1:end], D, D)
	μ = Σ*η[1:D]
	vcat(μ, vec(Λ))
end

function unpack(p::AbstractNormal, μ)
	D = length(p.μ)
	xxᵀ = reshape(μ[D+1:end], D, D)
	x = μ[1:D]
	(x=x, xxᵀ=xxᵀ)
end

function A(p::AbstractNormal, η)
	D = length(p.μ)

	Λ = -2*reshape(η[D+1:end], D, D)
	μ = inv(Λ)*η[1:D]

	-(1/2) * logdet(Λ) + (1/2)*μ'*Λ*μ
end

function sample(p::AbstractNormal)
	D = length(p.μ)
    p.μ + cholesky(inv(p.Λ)).L*randn(D)
end

#######################################################################
# Alternative parameterization.

struct NormalCholesky{T1<:AbstractVector,T2<:AbstractMatrix} <: AbstractNormal
    μ::T1
	L::T2

    function NormalCholesky(μ, Λ)
        L = cholesky(Λ).L
        new{typeof(μ), typeof(L)}(μ, L)
    end
end

function Base.getproperty(val::NormalCholesky, name::Symbol)
    if name === :Λ
        return val.L*val.L'
    else
        return getfield(val, name)
    end
end
