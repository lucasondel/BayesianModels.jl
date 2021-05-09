# Lucas Ondel, 2021

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

basemeasure(::Normal, X::AbstractMatrix{T}) where T =
    -.5*log(2π) * ones(T, size(X,2))

function vectorize(m::Normal)
    diagΛ, trilΛ, lnΛ = statistics(m.Λ)
    Λ = EFD.matrix(diagΛ, trilΛ)
    μ, diag_μμᵀ, tril_μμᵀ = statistics(m.μ)
    tr_Λμμᵀ = dot(diagΛ, diag_μμᵀ) + 2*dot(trilΛ, tril_μμᵀ)
    vcat(Λ*μ, -.5 * diagΛ, -trilΛ, -.5 * (tr_Λμμᵀ - lnΛ))
end

function statistics(::Normal, X::AbstractMatrix{T}) where T
    S1 = X
    S2 = hcat([(xxᵀ = x*x'; vcat(diag(xxᵀ), EFD.vec_tril(xxᵀ), 1))
               for x in eachcol(X)]...)
    vcat(S1, S2)
end

function loglikelihood(m::Normal, X::AbstractMatrix)
    Tη = vectorize(m)
    TX = statistics(m, X)
    TX' * Tη .+ basemeasure(m, X)
end

