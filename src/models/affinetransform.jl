# Probabilistic Affine Transform.
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct AffineTransform{D,Q}
        W
        b
    end

Affine transform.
"""
struct AffineTransform{D,Q} <: BMObject
    W::ParamList
    b::AbstractParam
end

function (trans::AffineTransform)(x::AbstractVector{<:Number})
    W = hcat([statistics(w) for w in trans.W]...)
    b = statistics(trans.b)
    W' * x + b
end

function (trans::AffineTransform)(X::AbstractVector{<:AbstractVector})
    W = hcat([statistics(w) for w in trans.W]...)
    b = statistics(trans.b)
    [W'*x + b for x in X]
end

#######################################################################
# Hierarchical prior for the bases

function DefaultHNormalDiagParameter(μ::AbstractVector{T}) where T
    EFD.Parameter(μ, identity, identity)
end

struct HNormalDiag{D} <: BMObject #<: EFD.AbstractNormal{D}
    αs::ParamList
    param::EFD.Parameter{T} where T
end

function HNormalDiag(αs, μ::AbstractVector{T}) where T
    D = length(μ)
    p = DefaultHNormalDiagParameter(μ)
    HNormalDiag{D}(αs, p)
end

#######################################################################
# EFD.Distribution interface

function _hnormal_split(Tαs)
    ([getindex.(Tαs, 1)...], [getindex.(Tαs, 2)...])
end

function EFD.basemeasure(n::HNormalDiag{D}, x::AbstractVector{T}) where {T,D}
    αs, lnαs = _hnormal_split(statistics.(n.αs))
    sum( T(.5)*(lnαs .- αs .* x.^2 .- log(T(2π))) )
end

function EFD.lognorm(n::HNormalDiag{D},
                     η::AbstractVector{T} = EFD.naturalform(n.param)) where {T,D}
    μ = η
    αs, _ = _hnormal_split(statistics.(n.αs))
    -T(.5)*dot(αs, μ.^2)
end

EFD.stats(::HNormalDiag, x) = x

EFD.splitgrad(n::HNormalDiag, m) = m

function EFD.stdparam(n::HNormalDiag{D},
                      η::AbstractVector{T} = EFD.naturalform(n.param)) where{T,D}
    (μ = η,)
end

function EFD.kldiv(q::EFD.NormalDiag{D}, p::HNormalDiag{D};
                   μ = EFD.gradlognorm(q)) where D
    q_η = EFD.naturalform(q.param)
    αs, lnαs = _hnormal_split(statistics.(p.αs))
    m = EFD.naturalform(p.param)
    T = eltype(m)
    p_η = vcat(αs .* m, T(-.5) .* αs)

    A_p_η = -sum( T(.5) .* (lnαs .- αs .* m.^2 ) )
    A_p_η - EFD.lognorm(q, q_η) - dot(p_η .- q_η, μ)
end

