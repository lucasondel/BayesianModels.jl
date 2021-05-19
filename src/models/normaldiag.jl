# SPDX-License-Identifier: MIT

#######################################################################
# Model definition

"""
    struct NormalDiag{D} <: AbstractModel
        μ
        λ
    end

Normal distribution with diagonal covariance matrix.
"""
struct NormalDiag{D} <: AbstractModel
    μ::T where T<:BayesParam
    λ::BayesParamList{D,T} where T<:BayesParam
end

function NormalDiag(;datadim, pstrength = 1, init_noise_std = 0, m₀ = nothing,
                    λ₀ = nothing)
    D = datadim
    m₀ = isnothing(m₀) ? zeros(D) : m₀
    λ₀ = isnothing(λ₀) ? ones(D) : λ₀

    μprior = ExpFamilyDistributions.NormalDiag(m₀, λ₀/pstrength)
    μposterior = ExpFamilyDistributions.NormalDiag(m₀ + randn(D)*init_noise_std, λ₀/pstrength)
    μ = BayesParam(μprior, μposterior)

    α₀ = pstrength
    β₀ = pstrength ./ λ₀
    λ = tuple([BayesParam(Gamma{Float64}(α₀, β₀ᵢ), Gamma{Float64}(α₀, β₀ᵢ)) for β₀ᵢ in β₀]...)

    NormalDiag{D}(μ, λ)
end

#######################################################################
# Model interface

basemeasure(::NormalDiag, x::AbstractVector{<:Real}) = -.5*length(x)*log(2π)

function vectorize(m::NormalDiag)
    Tλ = gradlognorm.(getproperty.(m.λ, :posterior), vectorize = false)
    λ = [Tλᵢ[1] for Tλᵢ in Tλ]
    lnλ = [Tλᵢ[2] for Tλᵢ in Tλ]
    μ, μ² = gradlognorm(m.μ.posterior, vectorize = false)
    vcat(λ .* μ, -.5 .* λ, -.5 * (λ' * μ² - sum(lnλ)))
end

statistics(m::NormalDiag, x::AbstractVector{<:Real}) = vcat(x, x.^2, 1)

function loglikelihood(m::NormalDiag, x::AbstractVector{<:Real})
    D = @Zygote.ignore length(x)
    Tη = vectorize(m)
    Tx = @Zygote.ignore statistics(m, x)
    Tη'*Tx + basemeasure(m, x)
end

function loglikelihood(m::NormalDiag, X::AbstractVector{<:AbstractVector})
    D = @Zygote.ignore length(X[1])
    Tη = vectorize(m)
    TX = @Zygote.ignore statistics.([m], X)
    dot.([Tη], TX) .+ basemeasure.([m], X)
end

function getparam_stats(m::NormalDiag, X, resps)
    λ_s = λstats(m, X, resps)
    μ_s = μstats(m, X, resps)
    T = eltype(μ_s)
    retval = Dict{BayesParam, Vector{T}}(m.μ => μ_s)
    for (λᵢ,λᵢ_s) in zip(m.λ, λ_s)
        retval[λᵢ] = λᵢ_s
    end
    retval
end
getparam_stats(m::NormalDiag, X) = getparam_stats(m, X, ones(eltype(X[1]), length(X)))

#######################################################################
# λ statistics

function λstats(m::NormalDiag, X, resps)
    μ, μ² = gradlognorm(m.μ.posterior, vectorize = false)
    s1 = sum(t -> ((x,z) = t ; z * (-.5 * x.^2 + μ .* x -.5*μ²)) , zip(X, resps))
    s2 = .5*ones(eltype(μ), length(μ))*sum(resps)
    map(x -> eltype(μ)[x...], zip(s1, s2))
end

#######################################################################
# μ statistics

function μstats(m::NormalDiag, X, z)
    λ = [gradlognorm(λᵢ.posterior, vectorize = false)[1] for λᵢ in m.λ]
    s = sum(x -> vcat(λ .* x, -λ/2), X)
end

