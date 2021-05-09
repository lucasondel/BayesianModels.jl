# Generic type model.
#
# Lucas Ondel 2021

#######################################################################
# Base type

"""
    abstract type AbstractModel

AbstractType of (Bayesian) models.
"""
abstract type AbstractModel <: BMObject end

#######################################################################
# Model interface

"""
    loglikelihood(model, X)

Returns the log-likelihood of `model` of each data sample of `X`.
"""
loglikelihood

#######################################################################
# List type

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}


#######################################################################
# Concrete models

include("affinetransform.jl")
include("gsm.jl")
include("normal.jl")
#include("normaldiag.jl")

