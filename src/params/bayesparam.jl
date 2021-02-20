# Bayesian parameter, i.e. parameter with a prior/posterior.
#
# Lucas Ondel 2021

_default_stats_fn(post, μ) = EFD.splitgrad(post, μ)

"""
    struct BayesParam{T,DistType} <: AbstractParam{T}
        prior::DistType
        posterior::DistType
        ...
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior. Both the prior and the posterior should be of the same type.

# Constructor

    BayesParam(prior, posterior, [stats_fn = ...::Function,
               grad_map = ::InvertibleMap])

Create a parameter with a prior and a posterior. `prior` and
`posterior` should have the same type. `stats_fn` is a function to
get the sufficient statistics of the parameter wrt to the model.
"""
struct BayesParam{T,DistType} <: AbstractParam{T}
    prior::DistType
    posterior::DistType

    _stats_fn::Function
    _μ::V where V <: AbstractVector{T}
    _grad_map::InvertibleMap

    function BayesParam(
                prior::DistType,
                posterior::DistType;
                stats_fn = (μ -> _default_stats_fn(posterior, μ)),
                grad_map = InvertibleMap(identity, identity)
            ) where DistType
        μ = EFD.gradlognorm(posterior)
        T = eltype(μ)
        new{T,typeof(prior)}(prior, posterior, stats_fn, μ, grad_map)
    end
end

statistics(param::BayesParam) = param._stats_fn(param._μ)

"""
    isbayesparam(p)

Returns true if `p` is a `BayesianParam`.
"""
isbayesparam(p) = typeof(p) <: BayesParam


