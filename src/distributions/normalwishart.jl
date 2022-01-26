# SPDX-License-Identifier: MIT

struct NormalWishart{T1<:AbstractVector,T2<:Real,
					 T3<:AbstractMatrix,T4<:Real} <: ExponentialFamilyDistribution
	μ::T1
	β::T2
	W::T3
	ν::T4
end

function η(p::NormalWishart)
	D = length(p.μ)
	μ, β, W, ν = p.μ, p.β, p.W, p.ν

	M =
	η₁ = β * μ
	η₂ = -β/2
	η₃ = -(1/2)*vec(inv(W) + β * μ*μ')
	η₄ =  (ν - D) / 2
	η = vcat(η₁, η₂, η₃, η₄)
end

function ξ(p::NormalWishart, η)
	D = length(p.μ)
	β = -2*η[D+1]
	μ = (1/β)*η[1:D]
	ν = 2*η[end] + D
    W = inv(Symmetric(reshape(-2*η[D+2:end-1], D, D) - β*(μ*μ')))
	vcat(μ, β, vec(W), ν)
end

function unpack(p::NormalWishart, μ)
	D = length(p.μ)
	Yx = μ[1:D]
	xᵀYx = μ[D+1]
	Y = reshape(μ[D+2:end-1], D, D)
	logdetY = μ[end]
	#(Yx=Yx, xᵀYx=xᵀYx, Y=Y, logdetY=logdetY)
	(Yx, xᵀYx, Y, logdetY)
end

function A(p::NormalWishart, η)
	D = length(p.μ)

	# We don't use `p.W` or `p.ν` to be able to calculate the gradient w.r.t. η.
    β = -2*η[D+1]
	μ = (1/β)*η[1:D]
	ν = 2*η[end] + D
	W⁻¹ = reshape(-2*η[D+2:end-1], D, D) - β*μ*μ'

	(
		- (D/2)*log(β)
		+ (ν/2)*(-logdet(W⁻¹) + D*log(2))
		+ sum(loggamma.((ν + 1 .- (1:D))/2))
	)
end

function sample(p::NormalWishart)
    μ, β, W, ν = p.μ, p.β, p.W, p.ν
    Λ = rand(Distributions.Wishart(ν, W))
    m = rand(Distributions.MultivariateNormal(m, inv(β*Λ)))
	m, Λ
end
