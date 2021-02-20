# Normal (i.e. Gaussian) model.
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct Normal{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution.
"""
struct Normal{D} <: AbstractModel
    μ::T where T<:AbstractParam
    Λ::T where T<:AbstractParam
end

#######################################################################
# Model interface

basemeasure(::Normal, x::AbstractVector{<:Real}) = -.5*length(x)*log(2π)

function vectorize(m::Normal)
    Λ, lnΛ = statistics(m.Λ)
    μ, μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(vec(Λ), vec(μμᵀ))
    vcat(Λ*μ, -.5 * vec(Λ), -.5 * (tr_Λμμᵀ - lnΛ))
end

statistics(::Normal, x::AbstractVector{<:Real}) = vcat(x, vec(x*x'), 1)

function loglikelihood(m::Normal, x::AbstractVector{<:Real})
    Tη = vectorize(m)
    Tx = statistics(m, x)
    dot(Tη, Tx) + basemeasure(m, x)
end

function loglikelihood(m::Normal, X::AbstractVector{<:AbstractVector})
    Tη = vectorize(m)
    TX = statistics.([m], X)
    dot.([Tη], TX) .+ basemeasure.([m], X)
end

