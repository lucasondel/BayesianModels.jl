# Mixture model.
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct Mixture{C} <: AbstractModel
        π
        components
    end

Mixture model. `C` is the number of  components.
"""
struct Mixture{C} <: AbstractModel
    π::T where T<:BayesParam
    components::ModelList{C,T} where T<:AbstractModel
end

function Mixture(;components, pstrength = 1)
    C = length(components)
    πprior = Dirichlet(pstrength .* ones(C) ./ C)
    πposterior = Dirichlet(pstrength .* ones(C) ./ C)
    π = BayesParam(πprior, πposterior)
    Mixture{C}(π, tuple(components...))
end

#######################################################################
# Pretty print

function Base.show(io::IO, ::MIME"text/plain", m::Mixture)
    println(io, typeof(m), ":")
    println(io, "  π: $(typeof(m.π))")
    println(io, "  components: $(typeof(m.components))")
end

#######################################################################
# Estimate the latent variables

function (m::Mixture)(X::AbstractVector{<:Real})
    TH = vectorize(m)
    Tx = statistics(m, X)
    r = dot.(TH, [Tx])
    exp.(r .- logsumexp(r))
end

function (m::Mixture)(X::AbstractVector)
    TH = vectorize(m)
    TX = statistics.([m], X)
    retval = []
    for Tx in TX
        r = dot.(TH, [Tx])
        push!(retval, exp.(r .- logsumexp(r)))
    end
    retval
end

#######################################################################
# Model interface

basemeasure(m::Mixture, x::AbstractVector{<:Real}) = basemeasure(m.components[1], x)

function vectorize(m::Mixture)
    lnπ = gradlognorm(m.π.posterior)
    [vcat(vectorize(c), lnπᵢ) for (c, lnπᵢ) in zip(m.components, lnπ)]
end

function statistics(m::Mixture, x::AbstractVector{<:Real})
    vcat(statistics(m.components[1], x), 1)
end

function loglikelihood(m::Mixture, x::AbstractVector{<:Real})
    TH = vectorize(m)
    Tx = statistics(m, x)
    logsumexp(dot.(TH, [Tx]) .+ basemeasure(m, x))
end

function loglikelihood(m::Mixture, X::AbstractVector{<:AbstractVector})
    TH = vectorize(m)
    TX = statistics.([m], X)
    BX = basemeasure.([m], X)
    [logsumexp(dot.(TH, [Tx]) .+ Bx) for (Tx, Bx) in zip(TX, BX)]
end

function getparam_stats(m::Mixture, X, resps)
    q_z = X |> m
    π_s = πstats(m, q_z)
    retval = Dict{BayesParam, Vector}(m.π => π_s)
    for (i, comp) in enumerate(m.components)
        merge!(retval, getparam_stats(comp, X, getindex.(q_z, i) .* resps))
    end
    retval
end
getparam_stats(m::Mixture, X) = getparam_stats(m, X, ones(eltype(X[1]), length(X)))

#######################################################################
# π statistics

πstats(m::Mixture, resps) = sum(resps)

