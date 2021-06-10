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
mutable struct Mixture{C,M<:AbstractModel} <: AbstractMixture{C}
    π::P where P <: AbstractParameter
    components::ModelList{C,M}
end

basemeasure(m::Mixture, x::AbstractVector) = basemeasure(m.components[1], x)

function Mixture(T ;components, pstrength = 1)
    C = length(components)
    πprior = EFD.Dirichlet(T(pstrength) .* ones(T, C) ./ C)
    πposterior = EFD.Dirichlet(T(pstrength) .* ones(T, C) ./ C)
    π = BayesianParameter(πprior, πposterior)
    Mixture{C,eltype(components)}(π, tuple(components...))
end
Mixture(;kwargs...) = Mixture(Float64; kwargs...)

function vectorize(m::Mixture{C,M}, cache = Dict()) where {C,M}
    @cache cache lnπ = statistics(m.π)
    @cache cache component_caches = [Dict() for i in 1:C]
    vecs = [vectorize(comp, comp_cache)
            for (comp, comp_cache) in zip(m.components, component_caches)]
    vcat(hcat(vecs...), lnπ')
end

function statistics(m::Mixture, X::AbstractMatrix)
    one_const = similar(X, eltype(X), (1,size(X,2)))
    fill!(one_const, 1)
    vcat(statistics(m.components[1], X), one_const)
end

function loglikelihood(m::Mixture, X::AbstractMatrix, cache = Dict())
    TH = vectorize(m, cache)
    @cache cache TX = statistics(m, X)
    r = TH' * TX
    lnγ = r .- logsumexp(r, dims = 1)
    @cache cache γ = exp.(lnγ)
    sum(γ .* r, dims = 1)' .- sum(γ .* lnγ, dims = 1)'
end

function ∇sum_loglikelihood(m::Mixture, cache)
    TX = cache[:TX]
    γ = cache[:γ]
    acc_Tx = TX*γ'
    ∂Tπ = sum(eachcol(γ))
    grads = Dict{Any,Any}(m.π => ∂Tπ)
    for (comp, Tx, c_cache) in zip(m.components, eachcol(acc_Tx), cache[:component_caches])
        merge!(grads, ∇sum_loglikelihood(comp, Tx, c_cache))
    end
    grads
end

function posterior(m::Mixture, X::AbstractMatrix)
    TH = vectorize(m)
    TX = statistics(m, X)
    r = TH' * TX
    exp.(r .- logsumexp(r, dims = 1))
end

function predict(m::Mixture, X::AbstractMatrix)
    TH = vectorize(m)
    TX = statistics(m, X)
    r = TH' * TX
    _, maxinds = findmax(r, dims = 1)
    dropdims(getindex.(maxinds, 1), dims = 1)
end
