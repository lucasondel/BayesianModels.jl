# Bayesian parameter, i.e. parameter with a prior/posterior.
#
# Lucas Ondel 2021

"""
    abstract type AbstractBayesParam{T} end

Generic type for a BayesianParameter.
"""
abstract type AbstractBayesParam{T} <: BMObject end

"""
    struct BayesParam{T}
        prior::T
        posterior::T
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior. Both the prior and the posterior should be of the same type.

# Constructor

    BayesParam(prior, posterior, [stats_fn = function])

Create a parameter with a prior and a posterior. `prior` and
`posterior` should have the same type. `stats_fn` is a function to
get the sufficient statistics of the parameters from a distribution.
"""
struct BayesParam{T} <: AbstractBayesParam{T}
    prior::T
    posterior::T

    _stats_fn::Function

    function BayesParam(prior::T, posterior::T;
                        stats_fn = (p -> EFD.gradlognorm(p, vectorize = false))) where T
        new{T}(prior, posterior, stats_fn)
    end
end


"""
    statistics(param)

Returns the statistics of the parameter's posterior.
"""
statistics(param::BayesParam) = param._stats_fn(param.posterior)

"""
    BayesParamList{N,T}(m1, m2, m3, ...)

Store a list of (bayesian) parameters for an object's attribute. `N`
is the number of the parameters and `T` is teh type of the parameters.
The list is immutable.
"""
const BayesParamList{N,T<:BayesParam} = NTuple{N,T}

"""
    getparams(model)

Returns a list of a all the [`BayesParam`](@ref) in `model` and its
attributes. Note that array of parameters and and array of sub-models
should be stored as [`BayesParamList`](@ref) and [`ModelList`](@ref)
respectively.
"""
function getparams(model::T) where T<:AbstractModel
    params = BayesParam[]
    for name in fieldnames(typeof(model))
        prop = getproperty(model, name)
        if typeof(prop) <: BayesParam
            push!(params, prop)
        elseif typeof(prop) <: BayesParamList
            push!.([params], prop)
        elseif typeof(prop) <: ModelList
            for m in prop
                push!.([params], getparams(m))
            end
        else
            push!.([params], getparams(prop))
        end
    end
    params
end

