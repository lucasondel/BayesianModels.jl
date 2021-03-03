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

# Constructor

    ParamList(m1, m2, m3, ...)
"""
const ParamList{N,T<:AbstractParam} = NTuple{N,T}

ParamList(m...) = tuple(m...)

"""
    getparams(obj)

Returns a list of a all the parameters in `obj` and its
attributes. Note that array of parameters and array of sub-models
should be stored as [`BayesParamList`](@ref) and [`ModelList`](@ref)
respectively.
"""
function getparams(obj)
    params = Set{AbstractParam}()
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        if typeof(prop) <: AbstractParam
            push!(params, prop)
            push!.([params], getparams(prop))
        elseif typeof(prop) <: ParamList
            for item in prop
                push!(params, item)
                push!.([params], getparams(item))
            end
        elseif typeof(prop) <: ModelList
            for m in prop
                push!.([params], getparams(m))
            end
        else
            push!.([params], getparams(prop))
        end
    end
    [params...]
end

include("bayesparam.jl")
include("constparam.jl")

