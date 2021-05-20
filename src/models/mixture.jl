# SPDX-License-Identifier: MIT

abstract type AbstractMixture{C} <: AbstractLatentVariableModel end

"""
    struct Mixture{C,M} <: AbstractMixture
        π
        components
    end

Mixture model. `C` is the number of  components and `M` is the type of
the components.
"""
struct Mixture{C,M<:AbstractModel} <: AbstractMixture{C}
    π::P where P <: AbstractParameter
    components::ModelList{C,M}
end

basemeasure(m::Mixture, x::AbstractVector) = basemeasure(m.components[1], x)

function Mixture(;components, pstrength = 1)
    C = length(components)
    πprior = EFD.Dirichlet(pstrength .* ones(C) ./ C)
    πposterior = EFD.Dirichlet(pstrength .* ones(C) ./ C)
    π = BayesianParameter(πprior, πposterior)
    Mixture{C,eltype(components)}(π, tuple(components...))
end

function vectorize(m::Mixture{C,M}) where {C,M}
    lnπ = statistics(m.π)
    #vcat(hcat(vectorize.(m.components)...), lnπ')
    hcat([vcat(vectorize(m.components[i]), lnπ[i]) for i in 1:C]...)
end

function predict(m::Mixture, X::AbstractMatrix)
    TH = vectorize(m)
    TX = statistics(m, X)
    r = TH' * TX
    exp.(r .- logsumexp_dim1(r))
end

function statistics(m::Mixture, X::AbstractMatrix)
    one_const = similar(X, eltype(X), (1,size(X,2)))
    fill!(one_const, 1)
    vcat(statistics(m.components[1], X), one_const)
end

function loglikelihood(m::Mixture, X::AbstractMatrix)
    TH = vectorize(m)
    TX = statistics(m, X)
    r = TH' * TX
    lnγ = r .- logsumexp_dim1(r)
    γ = exp.(lnγ)
    exp_llh = γ .* r
    sum(exp_llh, dims = 1)' .- sum(γ .* lnγ, dims = 1)'
end
