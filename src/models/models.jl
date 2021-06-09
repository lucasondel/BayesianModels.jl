# SPDX-License-Identifier: MIT

"""
    abstract type AbstractModel

Abstract type for all (Bayesian) models.
"""
abstract type AbstractModel <: BMObject end

"""
    abstract type AbstractLatentVariableModel

Abstract type for model with latent variable (mixture, pca, ...).
"""
abstract type AbstractLatentVariableModel <: AbstractModel end

"""
    loglikelihood(model::AbstractModel, X)

Return the log-likelihood of `model` of each data sample of `X`.
"""
loglikelihood


"""
    predict(model::AbstractLatentVariableModel, X)

Return the posterior over the latent variable of `model`.
"""
predict

"""
    todict(model)

Return a dictionary of the state of the model.
"""
function todict(m::AbstractModel)
    d = Dict()
    names = fieldnames(typeof(m))
    d[:length] = length(names)
    for (i,name) in enumerate(names)
        prop = getproperty(m, name)
        d[Symbol("prop_$(i)_type")] = typeof(prop)
        d[Symbol("prop_$i")] = todict(prop)
    end
    d
end

"""
    fromdict(T, dict)

Return a model of type `T` from the state stored in `dict`.
"""
function fromdict(T::Type{<:AbstractModel}, d::AbstractDict)
    params = []
    for i in 1:d[:length]
        ptype = d[Symbol("prop_$(i)_type")]
        pstate = d[Symbol("prop_$(i)")]
        push!(params, fromdict(ptype, pstate))
    end
    T(params...)
end

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}

function todict(t::ModelList)
    d = Dict()
    d[:eltype] = eltype(t)
    d[:length] = length(t)
    for (i,el) in enumerate(t)
        d[Symbol("el_$(i)")] = todict(el)
    end
    d
end

function fromdict(T::Type{<:ModelList}, d::AbstractDict)
    a = []
    for i in 1:d[:length]
        push!(a, fromdict(d[:eltype], d[Symbol("el_$(i)")]))
    end
    tuple(a...)
end

