# SPDX-License-Identifier: MIT

"""
    abstract type AbstractModel end

Abstract type for all the models.
"""
abstract type AbstractModel end

"""
    kldiv(model, qθ, Tθ)

KL divergence between the variational posterior qθ and the prior
of `model`. `Tθ` is the expected sufficient statistics w.r.t. `qθ`.
"""
kldiv(model::AbstractModel, qθ, Tθ) =
    kldiv(qθ, model.pθ, Tθ)

"""
    loglikelihood(model::AbstractModel, X, uTθ)

Return the log-likelihood of X given `model`. `uTθ` are the unpacked
sufficient statistics of the parameters.
"""
loglikelihood

"""
    elbo(model::AbstractModel, X, qθ, Tθ = getstats(model, qθ))

Evidence Lower-BOund of the `X` given `model` and the variational posterior
`qθ`. `Tθ` are the vectorized sufficient statistics of the parameters.
"""
function elbo(model::AbstractModel, X, qθ, Tθ = getstats(model, qθ); klscale=1,
              llhscale=1)
    uTθ = unpack(model, qθ, Tθ)
    llh = sum(loglikelihood(model, X, uTθ))
    kl = kldiv(model, qθ, Tθ)
    llhscale*llh - klscale*kl
end

"""
    getstats(model, qθ)

Return the expected value of the sufficient statistics of the parameters
`Tθ` w.r.t. `qθ`.
"""
getstats(model::AbstractModel, qθ::ExponentialFamilyDistribution) = μ(qθ)

"""
    unpack(model, qθ, Tθ)

Split the parameters statistics `Tθ` into its main components.
"""
unpack(model::AbstractModel, qθ::ExponentialFamilyDistribution,
       Tθ::AbstractVector) = unpack(qθ, Tθ)

"""
    newposterior(model, qθ, ∇μ; lrate=1, clip=true)

Create a new posterior whose parameter are estimated by a natural gradient
step with learning rate `lrate`.
"""
function newposterior(::AbstractModel, qθ::ExponentialFamilyDistribution, ∇μ;
                      lrate=1, clip=true)
    c∇μ = clip ? ∇μ ./ norm(∇μ) : ∇μ
    ηq = η(qθ)
    ∇̃ξ = ForwardDiff.derivative(t -> ξ(qθ, ηq + t * c∇μ), 0)
    ξᵗ⁺¹ = ξ(qθ, ηq) + lrate*∇̃ξ
    typeof(qθ)(unpack(qθ, ξᵗ⁺¹)...)
end

