# SPDX-License-Identifier: MIT

"""
    struct BayesianParameter{DT1,DT2,T<:AbstractVector} <: AbstractParameter
        prior::DT1
        posterior::DT1
        μ::T
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior. `DT1` is the type of distribution of the prior and `DT2` is
the type of the distribution of the posterior. `μ` is the vector of
statistics of the parameter.

# Constructor

    BayesianParameter(prior, posterior)

Create a parameter with a prior and a posterior.
"""
struct BayesianParameter{DT1,DT2,T} <: AbstractParameter
    prior::DT1
    posterior::DT2
    μ::T
end

function BayesianParameter(prior, posterior)
    μ = EFD.gradlognorm(posterior)
    BayesianParameter(prior, posterior, Param(μ))
end

statistics(p::BayesianParameter) = EFD.splitgrad(p.posterior, p.μ)

"""
    isbayesparam(p)

Returns true if `p` is a `BayesianParam`.
"""
isbayesianparam(p) = typeof(p) <: BayesianParameter
