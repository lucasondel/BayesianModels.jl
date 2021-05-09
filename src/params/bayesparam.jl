# Bayesian parameter, i.e. parameter with a prior/posterior.
#
# Lucas Ondel 2021

abstract type AbstractBayesParameter{DT1,DT2} <: AbstractParameter end

function Base.show(io::IO,
                   obj::AbstractBayesParameter{DT1,DT2}) where {DT1,DT2}
    print(io, "$(typeof(obj).name){$(DT1.name),$(DT2.name)}")
end

"""
    struct BayesParameter{T<:AbstractVector,DT1,DT2} <: AbstractParameter
        prior::DT1
        posterior::DT1
        μ::T
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior. `DT1` is the type of distribution of the prior and `DT2` is
the type of the distribution of the posterior. `μ` is the vector of
statistics of the parameter.

# Constructor

    BayesParameter(prior, posterior)

Create a parameter with a prior and a posterior.
"""
struct BayesParameter{DT1,DT2,T<:AbstractVector} <: AbstractBayesParameter{DT1,DT2}
    prior::DT1
    posterior::DT2
    μ::T
end

function BayesParameter(prior, posterior)
    μ = EFD.gradlognorm(posterior)
    BayesParameter(prior, posterior, μ)
end

"""
    isbayesparam(p)

Returns true if `p` is a `BayesianParam`.
"""
isbayesparam(p) = typeof(p) <: AbstractBayesParameter


