# PPCA - Implementation of the Probabilistic Principal Components
# Analysis (PPCA) model.
#
# Lucas Ondel 2020

#######################################################################
# Model definition

abstract type AbstractPPCAModel{T,D,Q} end

"""
    struct PPCAModel{T,D,Q} <: AbstractPPCAModel{T,D,Q}
        αprior      # hyper-prior over the bases
        wprior      # Prior over the bases
        hprior      # Prior over the embeddings
        λprior      # Prior over the precision
    end

Standard PPCA model.
"""
struct PPCAModel{T,D,Q} <: AbstractPPCAModel{T,D,Q}
    wprior::Normal{T,V} where V # V = Q + 1
    hprior::Normal{T,Q}
    λprior::Gamma{T}
end

"""
    struct PPCAModelHP{T,D,Q} <: AbstractPPCAModel{T,D,Q}
        αprior      # hyper-prior over the bases
        wprior      # Prior over the bases
        hprior      # Prior over the embeddings
        λprior      # Prior over the precision
    end

PPCA model with a hyper-prior over the variance of the prior over the
bases.
"""
struct PPCAModelHP{T,D,Q} <: AbstractPPCAModel{T,D,Q}
    αprior::Gamma{T}
    wprior::Normal{T,V} where V # V = Q + 1
    hprior::Normal{T,Q}
    λprior::Gamma{T}
end

function PPCAModel(T::Type{<:AbstractFloat}; datadim, latentdim,
                   pstrength = 1e-3, hyperprior = true)
    D, Q = datadim, latentdim

    if hyperprior
        return PPCAModelHP{T,D,Q}(
            Gamma{T}(pstrength, pstrength),
            Normal(zeros(T, Q+1), Symmetric(Matrix{T}(I, Q+1, Q+1))),
            Normal(zeros(T, Q), Symmetric(Matrix{T}(I, Q, Q))),
            Gamma{T}(pstrength, pstrength)
        )
    else
        return PPCAModel{T,D,Q}(
            Normal(zeros(T, Q+1), Symmetric(Matrix{T}(I, Q+1, Q+1))),
            Normal(zeros(T, Q), Symmetric(Matrix{T}(I, Q, Q))),
            Gamma{T}(pstrength, pstrength)
        )
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
    cindent = get(io, :indent, 0)
    println(io, typeof(model), ":")
    println(io, " "^(cindent+2), "αprior:")
    println(IOContext(io, :indent => cindent+4), model.αprior)
    println(io, " "^(cindent+2), "wprior:")
    println(IOContext(io, :indent => cindent+4), model.wprior)
    println(io, " "^(cindent+2), "hprior:")
    println(IOContext(io, :indent => cindent+4), model.hprior)
    println(io, " "^(cindent+2), "λprior:")
    println(IOContext(io, :indent => cindent+4), model.λprior)
end

function Base.show(io::IO, ::MIME"text/plain", model::PPCAModel)
    cindent = get(io, :indent, 0)
    println(io, typeof(model), ":")
    println(io, " "^(cindent+2), "wprior:")
    println(IOContext(io, :indent => cindent+4), model.wprior)
    println(io, " "^(cindent+2), "hprior:")
    println(IOContext(io, :indent => cindent+4), model.hprior)
    println(io, " "^(cindent+2), "λprior:")
    println(IOContext(io, :indent => cindent+4), model.λprior)
end

#######################################################################
# Initialization of the posteriors

function _init_wposts(T, D, Q, w_MAP)
    # Random initialization of the mean of q(W)
    w₀s = [vcat(w₀ ./ norm(w₀), 0) for w₀ in eachcol(randn(T, Q, D))]

    Σ = Symmetric(Matrix{T}(I, Q+1, Q+1))
    wposts = [w_MAP ? δNormal(w₀) : Normal(w₀, Σ) for w₀ in w₀s]
end


function θposteriors(model::PPCAModel{T,D,Q}; w_MAP = true) where {T,D,Q}
    Dict(
        :w => _init_wposts(T, D, Q, w_MAP),
        :λ => Gamma{T}(model.λprior.α, model.λprior.β)
    )
end

function θposteriors(model::PPCAModelHP{T,D,Q}; w_MAP = true) where {T,D,Q}
    Dict(
        :α => [Gamma{T}(model.αprior.α, model.αprior.β) for i in 1:Q+1],
        :w => _init_wposts(T, D, Q, w_MAP),
        :λ => Gamma{T}(model.λprior.α, model.λprior.β),
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

