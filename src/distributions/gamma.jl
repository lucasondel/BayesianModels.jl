# SPDX-License-Identifier: MIT

struct Gamma{T1, T2} <: ExponentialFamilyDistribution
    α::T1
	β::T2
end

η(p::Gamma) = [-p.β, p.α]
ξ(p::Gamma, η) = vcat(η[2], η[1])
unpack(p::Gamma, μ) = (x=μ[1], lnx=μ[2])

function A(p::Gamma, η)
    D = length(η)
    α, β = η[2], -η[1]
    loggamma(α) - α*log(β)
end

function sample(p::Gamma)
	α, β = ξ(p)
	rand(Distributions.Gamma(α, 1/β))
end
