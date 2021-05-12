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

Returns the log-likelihood of `model` of each data sample of `X`.
"""
loglikelihood


"""
    predict(model::AbstractLatentVariableModel, X)

Returns the posterior over the latent variable of `model`.
"""
predict

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}
