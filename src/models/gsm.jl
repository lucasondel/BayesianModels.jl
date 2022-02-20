# SPDX-License-Identifier: MIT

struct GSM{T1,T2} <: AbstractModel
    f::Function
    pW::T1
    models::T2
end

function kldiv(gsm::GSM, qθ, Tθ)
    qW, qH = qθ
    μW, μH = Tθ
    kldiv(qW, gsm.pW, μW) + sum(kldiv.(gsm.models, qH, μH))
end

function reparamtrick(x̄, diagx̄x̄ᵀ, vechx̄x̄ᵀ)
    T = eltype(x̄)
    mmᵀ = x̄ * x̄'
    x̄x̄ᵀ = diagm(diagx̄x̄ᵀ) + CompressedSymmetric(size(x̄, 1), 1, vechx̄x̄ᵀ)
    x̄ + cholesky(Symmetric(x̄x̄ᵀ - mmᵀ)).L * randn(T, size(x̄)...)
end

function reparamtrick(x̄::AbstractVector, x̄²::AbstractVector)
    T = eltype(x̄)
    m² = x̄ .^ 2
    x̄ + sqrt.(max.(0, x̄² .- m²)) .* randn(T, size(x̄)...)
end

reparamtrick(μx) = reparamtrick(μx...)

function getstats(gsm::GSM, qθ)
    qW, qH = qθ
    μ(qW), μ.(qH)
end

function unpack(gsm::GSM, qθ, Tθ)
    qW, qH = qθ
    μW, μH = Tθ
    T = eltype(μW)

    #W = reparamtrick(unpack(qW, μW), noisescale=0)
    W = unpack(qW, μW)[1] # Just use the mean, do not learn the variance
    H = hcat(reparamtrick.(unpack.(qH, μH))...)

    Ĥ = vcat(H, ones(T, 1, size(H, 2)))
    gsm.f.(eachcol(W' * Ĥ))
end

loglikelihood(gsm::GSM, Xs, uTθ) = sum(loglikelihood.(gsm.models, Xs, uTθ))

function newposterior(gsm::GSM, qθ::Tuple, ∇μ; lrate=1, clip=true)
    q₀W, q₀H = qθ
    ∇μW, ∇μH = ∇μ

    qW = newposterior(gsm, q₀W, ∇μW; lrate, clip)
    qH = newposterior.(gsm.models, q₀H, ∇μH; lrate, clip)

    qW, qH
end

