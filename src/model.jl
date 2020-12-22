# PPCA - Implementation of the Probabilistic Principal Components
# Analysis (PPCA) model.
#
# Lucas Ondel 2020

#######################################################################
# Model definition

abstract type AbstractPPCAModel{T,D,Q} end

"""
    struct PPCAModel{T,D,Q} <: AbstractPPCAModel{T,D,Q}
        w   # Prior over the bases
        h   # Prior over the embeddings
        λ   # Prior over the precision
    end

Standard PPCA model.
"""
struct PPCAModel{T,D,Q} <: AbstractPPCAModel{T,D,Q}
    w::Normal{T,V} where V # V = Q + 1
    h::Normal{T,Q}
    λ::Gamma{T}
end

"""
    struct PPCAModelHP{T,D,Q} <: AbstractPPCAModel{T,D,Q}
        α   # Hyper-prior over the bases
        w   # Prior over the bases
        h   # Prior over the embeddings
        λ   # Prior over the precision
    end

PPCA model with a hyper-prior over the variance of the prior over the
bases.
"""
struct PPCAModelHP{T,D,Q} <: AbstractPPCAModel{T,D,Q}
    α::Gamma{T}
    w::Normal{T,V} where V # V = Q + 1
    h::Normal{T,Q}
    λ::Gamma{T}
end

function PPCAModel(T::Type{<:AbstractFloat}; datadim, latentdim,
                   pstrength = 1e-3, hyperprior = true)
    D, Q = datadim, latentdim

    λ = Gamma{T}(pstrength, pstrength)
    w = Normal(zeros(T, Q+1), Symmetric(Matrix{T}(I, Q+1, Q+1)))
    h = Normal(zeros(T, Q), Symmetric(Matrix{T}(I, Q, Q)))

    if hyperprior
        α = Gamma{T}(pstrength, pstrength)
        return PPCAModelHP{T,D,Q}(α, w, h, λ)
    else
        return PPCAModel{T,D,Q}(w, h, λ)
    end
end
PPCAModel(;datadim, latentdim, pstrength = 1e-3,
          hyperprior = true) = PPCAModel(Float64;
                                         datadim=datadim,
                                         latentdim=latentdim,
                                         pstrength=pstrength,
                                         hyperprior = hyperprior)

#######################################################################
# Pretty print

function Base.show(io::IO, ::MIME"text/plain", model::PPCAModelHP)
    println(io, typeof(model), ":")
    println(io, "  α:")
    println(IOContext(io, :indent => cindent+4), model.α)
    println(io, " "^(cindent+2), "wprior:")
    println(IOContext(io, :indent => cindent+4), model.w)
    println(io, " "^(cindent+2), "hprior:")
    println(IOContext(io, :indent => cindent+4), model.h)
    println(io, " "^(cindent+2), "λprior:")
    println(IOContext(io, :indent => cindent+4), model.λ)
end

function Base.show(io::IO, ::MIME"text/plain", model::PPCAModel)
    cindent = get(io, :indent, 0)
    println(io, typeof(model), ":")
    println(io, " "^(cindent+2), "wprior:")
    println(IOContext(io, :indent => cindent+4), model.w)
    println(io, " "^(cindent+2), "hprior:")
    println(IOContext(io, :indent => cindent+4), model.h)
    println(io, " "^(cindent+2), "λprior:")
    println(IOContext(io, :indent => cindent+4), model.λ)
end

#######################################################################
# Initialization of the posteriors

function _init_wposts(T, D, Q, w_MAP)
    # Random initialization of the mean of q(W)
    w₀s = [vcat(w₀ ./ norm(w₀), 0) for w₀ in eachcol(randn(T, Q, D))]

    Σ = Symmetric(Matrix{T}(I, Q+1, Q+1))
    wposts = [w_MAP ? δNormal(w₀) : Normal(w₀, Σ) for w₀ in w₀s]
end

function _init_gamma(T, α, β, MAP)
    MAP ? δGamma{T}(α/β) : Gamma{T}(α, β)
end

function θposteriors(model::PPCAModel{T,D,Q}; w_MAP = true,
                     λ_MAP = false) where {T,D,Q}
    Dict(
        :w => _init_wposts(T, D, Q, w_MAP),
        :λ => _init_gamma(T, model.λ.α, model.λ.β, λ_MAP)
    )
end

function θposteriors(model::PPCAModelHP{T,D,Q}; w_MAP = false, λ_MAP = false,
                     α_MAP = false) where {T,D,Q}
    Dict(
        :α => [_init_gamma(T, model.α.α, model.α.β, α_MAP) for i in 1:Q+1],
        :w => _init_wposts(T, D, Q, w_MAP),
        :λ => _init_gamma(T, model.λ.α, model.λ.β, λ_MAP)
    )
end

"""
    θposteriors(model) -> Dict(:w => [Normal(...)], :λ => Gamma(...)[, :α => Gamma(...))

Create and initialize the posteriors of the PPCA parameters. The set
of posteriors are stored in a dictionary where the key is the
corresponding parameter. The `α` parameter is only added when the
model is a [`PPCAModelHP`](@ref).
"""
θposteriors

#######################################################################
# Log-likelihood

# Per dimension log-likelihood
function _llh_d(x, Tŵ, Tλ, Th)
    λ, lnλ = Tλ
    h, hhᵀ = Th

    # Extract the bias parameter
    wwᵀ = Tŵ[2][1:end-1, 1:end-1]
    w = Tŵ[1][1:end-1]
    μ = Tŵ[1][end]
    μ² = Tŵ[2][end, end]

    x̄ = dot(w, h) + μ
    lognorm = (-log(2π) + lnλ - λ*(x^2))/2
    K = λ*(x̄*x - dot(w, h)*μ - (dot(vec(hhᵀ), vec(wwᵀ)) + μ²)/2)

    lognorm + K
end

function _llh(x, Tŵs, Tλ, Th)
    f = (a,b) -> begin
        xᵢ, Tŵᵢ = b
        a + _llh_d(xᵢ, Tŵᵢ, Tλ, Th)
    end
    return foldl(f, zip(x, Tŵs), init = 0)
end

function loglikelihood(m::AbstractPPCAModel, X, θposts, hposts)
    _llh.(
        X,
        [[gradlognorm(p, vectorize = false) for p in θposts[:w] ]],
        [gradlognorm(θposts[:λ], vectorize = false)],
        [gradlognorm(p, vectorize = false) for p in hposts]
   ) - cost_reg.(hposts, [m.h])
end

