# SPDX-License-Identifier: MIT

"""
    abstract type AbstractModel

AbstractType of (Bayesian) models.
"""
abstract type AbstractModel <: BMObject end


"""
    loglikelihood(model, X)

Returns the log-likelihood of `model` of each data sample of `X`.
"""
loglikelihood

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}


#include("affinetransform.jl")
#include("gsm.jl")
include("normal.jl")
include("mixture.jl")

