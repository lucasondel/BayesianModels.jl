# SPDX-License-Identifier: MIT

# NOTE: depending on the font the variable 'ν' (Greek letter nu) may appear
# identical to 'v' (the latin letter v).

struct Wishart{T1<:AbstractMatrix, T2<:Real} <: ExponentialFamilyDistribution
    W::T1
	ν::T2
end

function η(p::Wishart)
	D = size(p.W, 1)
	W⁻¹ = inv(Symmetric(p.W))
	vcat(-(1/2)*vec(W⁻¹), (p.ν - D - 1)/2)
end

function ξ(p::Wishart, η)
	D = size(p.W, 1)
	W = inv(-2*reshape(η[1:end-1], D, D))
	ν = 2*η[end] + D + 1
	vcat(vec(W), ν)
end

function unpack(p::Wishart, μ)
	D = size(p.W, 1)
	X = reshape(μ[1:end-1], D, D)
	logdetX = μ[end]
	(X=X, logdetX=logdetX)
end

function A(p::Wishart, η)
	D = size(p.W, 1)

	# We don't use `p.W` or `p.ν` to be able to calculate the gradient w.r.t. η.
	W⁻¹ = -2*reshape(η[1:end-1], D, D)
	ν = 2*η[end] + D + 1

	(ν/2)*(-logdet(W⁻¹) + D*log(2)) + sum(loggamma.((ν + 1 .- (1:D))/2))
end

sample(p::Wishart) = rand(Distributions.Wishart(ν, W))

