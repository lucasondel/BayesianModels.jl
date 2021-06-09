# SPDX-License-Identifier: MIT

"""
    abstract type AbstractParameter

Generic type for all parameters. Subtype of `AbstractParameter` should
implement the property `μ` which is the model-dependent
"canonical form" of the parameter.
"""
abstract type AbstractParameter <: BMObject end

"""
    ParameterList{N,T<:AbstractParam}(m1, m2, m3, ...)

Store a list of (parameters for an object's attribute. `N` is the
number of the parameters and `T` is the type of the parameters.
The list is immutable.

# Constructor

    ParameterList(m1, m2, m3, ...)

"""
const ParameterList{N,T<:AbstractParameter} = NTuple{N,T}
ParameterList(m...) = tuple(m...)

function todict(t::ParameterList)
    d = Dict()
    d[:eltype] = eltype(t)
    d[:length] = length(t)
    for (i,el) in enumerate(t)
        d[Symbol("el_$(i)")] = todict(el)
    end
    d
end

function fromdict(T::Type{<:ParameterList}, d::AbstractDict)
    a = []
    for i in 1:d[:length]
        push!(a, fromdict(d[:eltype], d[Symbol("el_$(i)")]))
    end
    tuple(a...)
end

"""
    getparams(obj)

Returns a list of a all the parameters in `obj` and its
attributes. Note that array of parameters and array of sub-models
should be stored as [`BayesParameterList`](@ref) and [`ModelList`](@ref)
respectively.
"""
function getparams(obj)
    params = Set{AbstractParameter}()
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        if typeof(prop) <: AbstractParameter
            push!(params, prop)
            push!.([params], getparams(prop))
        elseif typeof(prop) <: BMObjectList
            for item in prop
                if typeof(item) <: AbstractParameter
                    push!(params, item)
                end
                push!.([params], getparams(item))
            end
        else
            push!.([params], getparams(prop))
        end
    end
    [params...]
end

"""
    statistics(param)

Returns the sufficient statistics of the `param` for the model it
belongs to.
"""
statistics
