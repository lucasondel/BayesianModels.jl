# PPCA - Accumulation of the sufficient statistics.
#
# Lucas Ondel 2020

#######################################################################
# Accumulator for the latent variable h

function _hstats_d(::AbstractPPCAModel, x::Real, Tŵ, Tλ)
    λ, _ = Tλ

    # Extract the bias parameter
    w = Tŵ[1][1:end-1]
    μ = Tŵ[1][end]

    λ*(x-μ)*w
end

function _hstats(m::AbstractPPCAModel{T,D,Q}, x::AbstractVector, Tŵ, Tλ) where {T,D,Q}
    f = (a,b) -> begin
        xᵢ, Tŵᵢ = b
        a + _hstats_d(m, xᵢ, Tŵᵢ, Tλ)
    end
    foldl(f, zip(x, Tŵ), init = zeros(T, Q))
end

function hstats(m::AbstractPPCAModel{T,D,Q}, X, θposts) where {T,D,Q}
    Tŵ = [gradlognorm(p, vectorize = false) for p in θposts[:w]]
    Tλ = gradlognorm(θposts[:λ], vectorize = false)
    S₁ = _hstats.([m], X, [Tŵ], [Tλ])
    S₂ = Tλ[1]*sum(a -> a[2][1:Q, 1:Q], Tŵ)
    S₁, S₂
end

function hstats(m::AbstractPPCAModel{T,D,Q}, X, z, θposts) where {T,D,Q}
    Tŵ = [gradlognorm(p, vectorize = false) for p in θposts[:w]]
    Tλ = gradlognorm(θposts[:λ], vectorize = false)
    M = _hstats.([m], X, [Tŵ], [Tλ])
    S₁ = Dict{eltype(z), Tuple{eltype(X), Int64}}()
    for (zᵢ, mᵢ) in zip(z, M)
        v, c = get(S₁, zᵢ, (zero(mᵢ), 0))
        S₁[zᵢ] = (v + mᵢ, c + 1)
    end
    S₂ = Tλ[1]*sum(a -> a[2][1:Q, 1:Q], Tŵ)
    S₁, S₂
end

function hposteriors(m::AbstractPPCAModel, X, θposts; MAP = false)
    S₁, S₂ = hstats(m, X, θposts)
    Λ₀ = inv(m.h.Σ)
    Λ₀μ₀ = Λ₀ * m.h.μ
    Σ = Symmetric(inv(Λ₀ + S₂))
    if MAP
        [δNormal(Σ * (Λ₀μ₀ + mᵢ)) for mᵢ in S₁]
    else
        [Normal(Σ * (Λ₀μ₀ + mᵢ), Σ) for mᵢ in S₁]
    end
end

function hposteriors(m::AbstractPPCAModel{T,D,Q}, X, z, θposts; MAP = false) where {T,D,Q}
    S₁, S₂ = hstats(m, X, z, θposts)
    Λ₀ = inv(m.h.Σ)
    Λ₀μ₀ = Λ₀ * m.h.μ
    pT = MAP ? δNormal : Normal
    posts = Dict{eltype(z), pT{T,Q}}()
    for k in keys(S₁)
        v, c = S₁[k]
        Σ = Symmetric(inv(Λ₀ + c*S₂))
        μ = Σ * (Λ₀μ₀ + v)
        posts[k] = MAP ? pT(μ) : pT(μ, Σ)
    end
    posts
end

#######################################################################
# Accumulator for the bases posteriors

function _wstats_d(::AbstractPPCAModel, x::Real, Tλ, Th)
    λ, _ = Tλ
    ĥ = vcat(Th[1], [1])
    λ * x * ĥ
end

function _wstats(m::AbstractPPCAModel, x::AbstractVector, Tλ, Th)
    sum(_wstats_d.([m], x, [Tλ], Th))
end

function wstats(m::AbstractPPCAModel{T,D,Q}, X, θpost, hposts) where {T,D,Q}
    Tλ = gradlognorm(θpost[:λ], vectorize = false)
    Th = [gradlognorm(p, vectorize = false) for p in hposts]
    Tw₁ = [_wstats(m, getindex.(X, i), Tλ, Th) for i in 1:D]

    Tw₂ = Matrix{T}(I, Q+1, Q+1)
    Tw₂[1:end-1, 1:end-1] .= sum(getindex.(Th, 2))
    Tw₂[1:end-1, end] = Tw₂[end, 1:end-1] = sum(getindex.(Th, 1))
    Tw₂[end,end] = length(X)
    [Tw₁, T(-.5) * Tλ[1] * Tw₂]
end

function wposteriors!(m::AbstractPPCAModel{T,D,Q}, θposts,
                      accstats)::Nothing where {T,D,Q}
    s1s, s2 = accstats

    Λ₀ = inv(m.w.Σ)
    Λ₀μ₀ = Λ₀ * m.w.μ

    Σ = Symmetric(inv(Λ₀ + -2*s2))
    for (wpost, s1) in zip(θposts[:w], s1s)
        wpost.μ = Σ * (Λ₀μ₀ + s1)

        if typeof(wpost) <: ExpFamilyDistribution
            wpost.Σ = Σ
        end
    end

    nothing
end

#######################################################################
# Accumulator for the precision parameter

function _λstats_d(::AbstractPPCAModel, x::Real, x², Tŵ, Th)
    h, hhᵀ = Th

    # Extract the bias parameter
    w = Tŵ[1][1:end-1]
    wwᵀ = Tŵ[2][1:end-1, 1:end-1]
    μ = Tŵ[1][end]
    μ² = Tŵ[2][end, end]

    x̄ = dot(w, h) + μ
    Tλ₁ = -.5*x² + x*x̄ - dot(w, h)*μ - .5*(dot(vec(hhᵀ), vec(wwᵀ)) + μ²)
    Tλ₂ = 1/2
    vcat(Tλ₁, Tλ₂)
end

function _λstats(m::AbstractPPCAModel, x::AbstractVector, x², Tŵ, Th)
    sum(_λstats_d.([m], x, x², Tŵ, [Th]))
end

function λstats(m::AbstractPPCAModel, X, X², θposts, hposts)
    Tŵ = [gradlognorm(p, vectorize = false) for p in θposts[:w]]
    Th = [gradlognorm(p, vectorize = false) for p in hposts]
    sum(_λstats.([m], X, X², [Tŵ], Th))
end

function λstats(m::AbstractPPCAModel, X, θposts, hposts)
    Tŵ = [gradlognorm(p, vectorize = false) for p in θposts[:w]]
    Th = [gradlognorm(p, vectorize = false) for p in hposts]
    sum(_λstats.([m], X, map(x -> x.^2, X), [Tŵ], Th))
end

function λposterior!(m::AbstractPPCAModel, θposts, accstats)::Nothing
    η₀ = naturalparam(m.λ)
    update!(θposts[:λ], η₀ + accstats)

    nothing
end

