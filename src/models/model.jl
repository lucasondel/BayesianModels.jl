# SPDX-License-Identifier: MIT

"""
    abstract type AbstractModel end

Abstract type for all the models.
"""
abstract type AbstractModel end

"""
    loglikelihood(model::abstractmodel, X)

Return the log-likelihood of `model`.
"""
loglikelihood
loglikelihood(model::AbstractModel, X, qθ, Tθ) =
    loglikelihood.(Ref(model), eachcol(X), Ref(unpack(model, qθ, Tθ)))
