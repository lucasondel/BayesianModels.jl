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
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end

function statistics(::Normal, x::AbstractVector{<:Real})
    xxᵀ = x*x'
    vcat(x, diag(xxᵀ), EFD.vec_tril(xxᵀ), 1)
end

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

