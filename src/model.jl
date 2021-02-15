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
    basemeasure(model, x)

Returns the base measure of `model` for the data sample `x`.
"""
basemeasure

"""
    vectorize(model)

Returns a "vectorized" version of the model, usually: `[η , -A(η)]ᵀ`.
"""
vectorize

"""
    statistics(model, x)

Returns the sufficient statistics of `model` for the data sample `x`.
"""
statistics

"""
    loglikelihood(model, x|X)

Returns the log-likelihood of `model` of the data sample `x` or of the
list of data samples `X`.
"""
loglikelihood

"""
    getparam_stats(model, X[, Φ])

Returns a dictionary `(param -> stats)` where `stats` are the
accumulated sufficient statistics of the parameters `param`. `Φ` is a
specific weighting for each frame of `X`.
"""
getparam_stats

getparam_stats(m, X) = getparam_stats(m, X, ones(eltype(X[1]), length(X)))

#######################################################################
# List type

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}

