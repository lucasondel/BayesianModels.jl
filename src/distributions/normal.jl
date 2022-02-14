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
    μ = inv(Λ)*η[1:D]
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

struct NormalDiag{T1<:AbstractVector,T2<:AbstractVector} <: AbstractNormal
    μ::T1
	lnλ::T2
end

function η(p::NormalDiag)
    λ = exp.(p.lnλ)
	vcat(λ .* p.μ, -(1/2)*λ)
end

function ξ(p::NormalDiag, η)
	D = length(p.μ)
	λ = -2* η[D+1:end]
    μ = (1 ./ λ) .* η[1:D]
    vcat(μ, log.(λ))
end

function unpack(p::NormalDiag, μ)
	D = length(p.μ)
	x² = μ[D+1:end]
	x = μ[1:D]
    (x=x, x²=x²)
end

function A(p::NormalDiag, η)
	D = length(p.μ)

	λ = -2 * η[D+1:end]
	μ = (1 ./ λ) .* η[1:D]

    -(1/2) * sum(log.(λ)) + (1/2) * sum(μ .* λ .* μ)
end

function sample(p::NormalDiag)
	D = length(p.μ)
    p.μ + sqrt.(1 ./ exp.(p.λ)) .* randn(D)
end

struct NormalIso{T1<:AbstractVector,T2<:Real} <: AbstractNormal
    μ::T1
	lnλ::T2
end

function η(p::NormalIso)
    λ = exp.(p.lnλ)
	vcat(λ .* p.μ, -(1/2)*λ)
end

function ξ(p::NormalIso, η)
	D = length(p.μ)
	λ = -2* η[D+1]
    μ = (1 / λ) .* η[1:D]
    vcat(μ, log(λ))
end

function unpack(p::NormalIso, μ)
	D = length(p.μ)
	x² = μ[D+1]
	x = μ[1:D]
    (x=x, x²=x²)
end

function A(p::NormalIso, η)
	D = length(p.μ)

	λ = -2 * η[D+1]
	μ = (1 / λ) .* η[1:D]

    -(D/2) * log(λ) + (1/2) * sum(μ .* λ .* μ)
end

function sample(p::NormalIso)
	D = length(p.μ)
    p.μ + sqrt.(1 / exp.(p.lnλ)) .* randn(D)
end

