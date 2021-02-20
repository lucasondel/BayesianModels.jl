# Models' parameter object
#
# Lucas Ondel 2021

"""
    abstract type AbstractParam{T}

Generic type for all parameters.
"""
abstract type AbstractParam{T} <: BMObject end

"""
    statistics(param)

Returns the statistics of the parameter.
"""
statistics(::AbstractParam)

"""
    ParamList{N,T}(m1, m2, m3, ...)

Store a list of (bayesian) parameters for an object's attribute. `N`
is the number of the parameters and `T` is the type of the parameters.
The list is immutable.
"""
const ParamList{N,T<:AbstractParam} = NTuple{N,T}

"""
    getparams(model)

Returns a list of a all the parameters in `model` and its
attributes. Note that array of parameters and array of sub-models
should be stored as [`BayesParamList`](@ref) and [`ModelList`](@ref)
respectively.
"""
function getparams(model::T) where T<:AbstractModel
    params = AbstractParam[]
    for name in fieldnames(typeof(model))
        prop = getproperty(model, name)
        if typeof(prop) <: AbstractParam
            push!(params, prop)
        elseif typeof(prop) <: ParamList
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

include("bayesparam.jl")
include("constparam.jl")

