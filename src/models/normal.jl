# Normal (i.e. Gaussian) model.
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct NormalDiag{T,D}
        μ       # mean
        λ       # diagonal of the precision matrix
    end

Normal distribution with diagonal covariance matrix.
"""
struct NormalDiag{T,D}
    μ::T where T<:BayesParam
    λ::Vector{<:BayesParam}
end

function NormalDiag(T::Type{<:AbstractFloat}; datadim, pstrength = 1,
                   init_noise_std = 0, m₀ = nothing, λ₀ = nothing)
    D = datadim
    m₀ = isnothing(m₀) ? zeros(T, D) : m₀
    λ₀ = isnothing(λ₀) ? ones(T, D) : λ₀

    μprior = ExpFamilyDistributions.NormalDiag(m₀, λ₀/pstrength)
    μposterior = ExpFamilyDistributions.NormalDiag(m₀ + randn(T, D)*init_noise_std, λ₀/pstrength)
    μ = BayesParam(μprior, μposterior)

    α₀ = pstrength
    β₀ = pstrength ./ λ₀
    λ = [BayesParam(Gamma{T}(α₀, β₀ᵢ), Gamma{T}(α₀, β₀ᵢ)) for β₀ᵢ in β₀]

    NormalDiag{T,D}(μ, λ)
end
function NormalDiag(; datadim, pstrength = 1e-3, init_noise_std = 0,
                    m₀ = nothing, λ₀ = nothing)
    NormalDiag(Float64, datadim = datadim, pstrength = pstrength,
               init_noise_std = init_noise_std, m₀ = m₀, λ₀ = λ₀)
end

#######################################################################
# Pretty print

function Base.show(io::IO, ::MIME"text/plain", n::NormalDiag)
    println(io, typeof(n), ":")
    println(io, "  μ: $(typeof(n.μ))")
    println(io, "  λ: $(typeof(n.λ))")
end

#######################################################################
# Model interface

function _llh_d(::NormalDiag, x, Tμ, Tλ)
    λ, lnλ = Tλ
    μ, μμᵀ = Tμ
    (-log(2π) + lnλ - λ*(x^2) - λ*μμᵀ)/2 + λ*μ*x
end

function _llh(m::NormalDiag, x, Tμ, Tλ)
    f = (a,b) -> begin
        xᵢ, Tμᵢ, Tλᵢ = b
        a + _llh_d(m, xᵢ, Tμᵢ, Tλᵢ)
    end
    foldl(f, zip(x, Tμ, Tλ), init = 0)
end

function loglikelihood(m::NormalDiag, X)
    _llh.(
        [m],
        X,
        [collect(zip(gradlognorm(m.μ.posterior, vectorize = false)...))],
        [[gradlognorm(λᵢ.posterior, vectorize = false) for λᵢ in m.λ]],
   )
end

function getparam_stats(m::NormalDiag, X)
    λ_s = λstats(m, X)
    μ_s = μstats(m, X)
    retval = Dict{BayesParam, Vector}(m.μ => μ_s)
    for (λᵢ,λᵢ_s) in zip(m.λ, λ_s)
        retval[λᵢ] = λᵢ_s
    end
    retval
end

#######################################################################
# λ statistics

function _λstats_d(::NormalDiag, x::Real, Tμ)
    μ, μμᵀ = Tμ
    Tλ₁ = (-x^2 -μμᵀ)/2 + μ*x
    Tλ₂ = 1/2
    vcat(Tλ₁, Tλ₂)
end

function _λstats(m::NormalDiag, x::AbstractVector, Tμ)
    _λstats_d.([m], x, Tμ)
end

function λstats(m::NormalDiag, X)
    Tμ = collect(zip(gradlognorm(m.μ.posterior, vectorize = false)...))
    sum(_λstats.([m], X, [Tμ]))
end

#######################################################################
# μ statistics

function _μstats_d(::NormalDiag, x::Real, Tλ)
    λ, _ = Tλ
    vcat(λ*x, -λ/2)
end

function _μstats(m::NormalDiag, x::AbstractVector, Tλ)
    _μstats_d.([m], x, Tλ)
end

function μstats(m::NormalDiag, X)
    Tλ = [gradlognorm(λᵢ.posterior, vectorize = false) for λᵢ in m.λ]
    s = sum(_μstats.([m], X, [Tλ]))
    vcat(getindex.(s, 1), getindex.(s, 2))
end

