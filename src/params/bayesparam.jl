# Bayesian parameter, i.e. parameter with a prior/posterior.
#
# Lucas Ondel 2021

_default_stats_fn(post, μ) = EFD.splitgrad(post, μ)

"""
    struct BayesParam{T,DT1,DT2} <: AbstractParam{T}
        prior::DistType
        posterior::DistType
        ...
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior. `DT1` is the type of distribution of the prior and `DT2` is
the type of the distribution of the posterior.

# Constructor

    BayesParam(prior, posterior, [stats_fn = ...::Function)

Create a parameter with a prior and a posterior. `prior` and
`posterior` should have the same type. `stats_fn` is a function to
get the sufficient statistics of the parameter wrt to the model.
"""
struct BayesParam{T,DT1,DT2} <: AbstractParam{T}
    prior::DT1
    posterior::DT2

    _stats_fn::Function
    _μ::V where V <: AbstractVector{T}

    function BayesParam(prior, posterior;
                        stats_fn = (μ -> _default_stats_fn(posterior, μ)))
        μ = EFD.gradlognorm(posterior)
        T = eltype(μ)
        new{T,typeof(prior),typeof(posterior)}(prior, posterior, stats_fn, μ)
    end
end

statistics(param::BayesParam) = param._stats_fn(param._μ)

"""
    isbayesparam(p)

Returns true if `p` is a `BayesianParam`.
"""
isbayesparam(p) = typeof(p) <: BayesParam


