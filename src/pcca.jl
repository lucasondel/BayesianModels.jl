# Probabilistic Canonical Correlation Analysis model.

struct PCCAModel{D, K, Q}
    hprior::Normal
    Wprior::Array{Normal}
    Wposterior::Array{Normal}
    Bprior::Array{Normal}
    Bposterior::Array{Normal}
    λprior::Gamma
    λposterior::Gamma
end

function PCCAModel{D,K,Q}(;pstrength::Real = 1.) where {D,K,Q}
    T = Float64
    τ = 1 / pstrength

    U = max(Q+1, D)
    initW = qr(randn(T, U, U)).Q
    Wprior = [Normal(zeros(T, Q+1), τ * Matrix{T}(I, Q+1, Q+1)) for d in 1:D]
    Wposterior = [Normal(initW[d, 1:Q+1], τ * Matrix{T}(I, Q+1, Q+1)) for d in 1:D]

    Bprior = [Normal(zeros(T, Q+1), τ * Matrix{T}(I, Q+1, Q+1)) for k in 1:K-1]
    Bposterior = [Normal(zeros(T, Q+1), τ * Matrix{T}(I, Q+1, Q+1)) for k in 1:K-1]

    λprior = Gamma([T(pstrength)], [T(pstrength)])
    λposterior = Gamma([T(pstrength)], [T(pstrength)])

    hprior = Normal(zeros(T, Q), Matrix{T}(I, Q, Q))

    PCCAModel{D,K,Q}(hprior, Wprior, Wposterior, Bprior, Bposterior, λprior,
                     λposterior)
end

struct PPCAModel{D,Q}
    hprior::Normal
    Wprior::Array{Normal}
    Wposterior::Array{Normal}
    λprior::Gamma
    λposterior::Gamma
end

function PPCAModel{D,Q}(;pstrength::Real = 1.) where {D,Q}
    T = Float64
    τ = 1 / pstrength

    U = max(Q+1, D)
    initW = qr(randn(T, U, U)).Q
    Wprior = [Normal(zeros(T, Q+1), τ * Matrix{T}(I, Q+1, Q+1)) for d in 1:D]
    Wposterior = [Normal(initW[d, 1:Q+1], τ * Matrix{T}(I, Q+1, Q+1)) for d in 1:D]

    λprior = Gamma([T(pstrength)], [T(pstrength)])
    λposterior = Gamma([T(pstrength)], [T(pstrength)])

    hprior = Normal(zeros(T, Q), Matrix{T}(I, Q, Q))
    PPCAModel{D,Q}(hprior, Wprior, Wposterior, λprior, λposterior)
end

