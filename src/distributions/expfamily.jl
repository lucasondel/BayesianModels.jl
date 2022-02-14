# SPDX-License-Identifier: MIT

"""
	abstract type ExponentialFamilyDistribution end

Abstract base type for distributions member of the exponential family.

| Methods            | Brief description |
|:-------------------|:------------------|
| `η(p)`             | Returns the natural parameters of the distribution |
| `ξ(p[, η])`        | Parametrization of the distribution as a function of the natural parameters |
| `μ(p)`             | Returns the expectation of the sufficient statistics |
| `A(p[, η])`        | Log-normalizer of the distribution as a function of the natural parameters |
| `kldiv(p, q[, μ])` | Returns the Kullback-Leibler divergence ``D[p || q].`` |
| `sample(p)`        | Draw a sample from `p` |
| `unpack(p, μ)      | Unpack the vector of sufficient statistics |

"""
abstract type ExponentialFamilyDistribution end

μ(p::ExponentialFamilyDistribution) = gradient(η -> A(p, η), η(p))[1]

function kldiv(p₁::ExponentialFamilyDistribution,
               p₂::ExponentialFamilyDistribution,
               μ₁::AbstractVector = μ(p₁))
	η₁, η₂ = η(p₁), η(p₂)
    A(p₂, η₂) - A(p₁, η₁) - dot(η₂ - η₁, μ₁)
end

