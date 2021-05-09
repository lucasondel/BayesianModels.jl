# Normal model.
#
# Lucas Ondel 2021

#######################################################################
# Model definition

abstract type Normal{D} <: AbstractModel end

Base.show(io::IO, normal::Normal) = print(io, typeof(normal))

"""
    struct Normal{D} <: AbstractModel
        μ
        Λ
    end

Normal distribution.
"""
struct NormalIndParams{P1<:AbstractParameter,P2<:AbstractParameter,D} <: Normal{D}
    μ::P1
    Λ::P2

    function NormalIndParams(μ, Λ)

    end
end

_basemeasure(::Normal, x::AbstractVector{<:Real}) = -.5*length(x)*log(2π)

function _vectorize(m::Normal)
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end

function _statistics(::Normal, x::AbstractVector{<:Real})
    xxᵀ = x*x'
    vcat(x, diag(xxᵀ), EFD.vec_tril(xxᵀ), 1)
end

#######################################################################
# Model interface

function loglikelihood(m::Normal, X::AbstractVector{<:AbstractVector})
    Tη = _vectorize(m)
    TX = _statistics.([m], X)
    dot.([Tη], TX) .+ _basemeasure.([m], X)
end

