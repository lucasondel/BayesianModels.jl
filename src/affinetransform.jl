# Probabilistic Affine Transform, elementary component of PPCA,
# PLDA,...
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct AffineTransform{T,D,Q} <: AbstractAffineTransform{T,D,Q}
        α       # Basis' prior scaling parameter
        W       # Array of basis parameter
        hprior  # Prior over the embeddings
    end

Affine transform with a hyper-prior over the variance of the prior over
the bases.
"""
struct AffineTransform{T,D,Q}
    α::Vector{<:BayesParam}
    W::Vector{<:BayesParam}
    hprior::Normal{T,Q}
end

function _init_wposts(T, D, Q, w_MAP)
    # Random initialization of the mean of q(W)
    w₀s = [vcat(w₀ ./ norm(w₀), 0) for w₀ in eachcol(randn(T, Q, D))]

    Σ = Symmetric(Matrix{T}(I, Q+1, Q+1))
    wposts = [w_MAP ? δNormal(w₀) : Normal(w₀, Σ) for w₀ in w₀s]
end

function AffineTransform(T::Type{<:AbstractFloat}; outputdim, inputdim,
                         pstrength = 1e-3, W_MAP = false, α_MAP = false)
    D, Q = outputdim, inputdim

    hprior = Normal(zeros(T, Q), Symmetric(Matrix{T}(I, Q, Q)))

    wprior = Normal(zeros(T, Q+1), Symmetric(Matrix{T}(I, Q+1, Q+1)))
    W = [BayesParam(wprior, post) for post in _init_wposts(T, D, Q, W_MAP)]

    _init_gamma(T, α, β, MAP) = MAP ? δGamma{T}(α/β) : Gamma{T}(α, β)
    αprior = Gamma{T}(pstrength, pstrength)
    αposts = [_init_gamma(T, αprior.α, αprior.β, α_MAP) for i in 1:Q+1]
    α = [BayesParam(αprior, post) for post in αposts]

    AffineTransform{T,D,Q}(α, W, hprior)
end
function AffineTransform(;inputdim, outputdim, pstrength = 1e-3,
                         W_MAP = false, α_MAP = false)
    AffineTransform(Float64; inputdim = inputdim, outputdim = outputdim,
                    pstrength = pstrength, W_MAP = W_MAP, α_MAP = α_MAP)
end

#######################################################################
# Pretty print

function Base.show(io::IO, ::MIME"text/plain", trans::AffineTransform)
    println(io, typeof(trans), ":")
    println(io, "  α: $(typeof(trans.α))")
    println(io, "  W: $(typeof(trans.W))")
    println(io, "  hprior: $(typeof(trans.hprior))")
end

#######################################################################
# Affine transformation from the parameters' posterior

# single dimension transform
function _affine_transform_d(ŵ, h)
    w = ŵ[1:end-1]
    μ = ŵ[end]
    dot(w, h) + μ
end
_affine_transform(ŵs, h) = _affine_transform_d.(ŵs, [h])

function (trans::AffineTransform)(X::AbstractVector)
    bases = mean.([w.posterior for w in trans.W])
    _affine_transform.([bases], X)
end

#######################################################################
# Sufficient statistics for the latent variable h

function _hstats_d(::AffineTransform, x::Real, Tŵ)
    # Extract the bias parameter
    w = Tŵ[1][1:end-1]
    μ = Tŵ[1][end]
    (x-μ)*w
end

function _hstats(m::AffineTransform{T,D,Q}, x::AbstractVector, Tŵ) where {T,D,Q}
    f = (a,b) -> begin
        xᵢ, Tŵᵢ = b
        a + _hstats_d(m, xᵢ, Tŵᵢ)
    end
    foldl(f, zip(x, Tŵ), init = zeros(T, Q))
end

function hstats(m::AffineTransform{T,D,Q}, X) where {T,D,Q}
    Tŵ = [gradlognorm(w.posterior, vectorize = false) for w in m.W]
    S₁ = _hstats.([m], X, [Tŵ])
    S₂ = sum(a -> a[2][1:Q, 1:Q], Tŵ)
    S₁, S₂
end

#######################################################################
# Update of the bases W

function _wstats_d(::AffineTransform, x::Real, Th)
    ĥ = vcat(Th[1], [1])
    x * ĥ
end

function _wstats(m::AffineTransform, x::AbstractVector, Th)
    sum(_wstats_d.([m], x, Th))
end

function wstats(m::AffineTransform{T,D,Q}, X, hposts) where {T,D,Q}
    Th = [gradlognorm(p, vectorize = false) for p in hposts]
    Tw₁ = [_wstats(m, getindex.(X, i), Th) for i in 1:D]

    Tw₂ = Matrix{T}(I, Q+1, Q+1)
    Tw₂[1:end-1, 1:end-1] .= sum(getindex.(Th, 2))
    Tw₂[1:end-1, end] = Tw₂[end, 1:end-1] = sum(getindex.(Th, 1))
    Tw₂[end,end] = length(X)
    [Tw₁, T(-.5) * Tw₂]
end

function update_W!(m::AffineTransform{T,D,Q}, accstats)::Nothing where {T,D,Q}
    s1s, s2 = accstats

    Λ₀ = inv(m.W[1].prior.Σ)
    Λ₀μ₀ = Λ₀ * m.W[1].prior.μ

    Σ = Symmetric(inv(Λ₀ + -2*s2))
    for (param, s1) in zip(m.W, s1s)
        param.posterior.μ = Σ * (Λ₀μ₀ + s1)

        if typeof(param.posterior) <: ExpFamilyDistribution
            param.posterior.Σ = Σ
        end
    end

    # Update the hyper-posterior of the scaling of the bases' prior
    update_α!(m, αstats(m))

    nothing
end

#######################################################################
# Update of the bases α parameter

function _αstats(::AffineTransform, Tw)
    w, wwᵀ = Tw
    s1 = -diag(wwᵀ) /2
    s2 = zero(w) .+ 1/2
    vcat(s1', s2')
end

function αstats(m::AffineTransform{T,D,Q}) where {T,D,Q}
    stats = foldl(
        (x, y) -> x + _αstats(m, gradlognorm(y.posterior, vectorize = false)),
        m.W,
        init = zeros(T, 2, Q+1)
    )
    eachcol(stats)
end

function update_α!(m::AffineTransform, accstats)::Nothing
    for (param, s) in zip(m.α, accstats)
        η₀ = naturalparam(param.prior)
        update!(param.posterior, η₀ + s)
    end

    # Update the prior of the model
    m.W[1].prior.Σ = Symmetric(diagm(1 ./ mean.([v.posterior for v in m.α])))

    nothing
end

