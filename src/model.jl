# Generic type model.
#
# Lucas Ondel 2021

"""
    abstract type AbstractModel

AbstractType of (Bayesian) models.
"""
abstract type AbstractModel end

"""
    ModelList{N,T}(m1, m2, m3, ...)

Store a list of models for an object's attribute. `N` is the number of
models in the list and `T` is the type of the models. The list is
immutable.
"""
const ModelList{N,T<:AbstractModel} = NTuple{N,T}

