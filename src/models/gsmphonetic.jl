# SPDX-License-Identifier: MIT

struct GSMPhonetic{T1,T2,T3} <: AbstractModel
    f::Function
    pL::T1
    pS::T2
    model::T3
end

function kldiv(gsm::GSMPhonetic, qθ, Tθ)
    qL, qS, qλ, qσ, qπ = qθ
    μL, μS, μλ, μσ, μπ = Tθ
    retval = (
        sum(kldiv.(qL, (gsm.pL,), μL))
        + sum(kldiv.(qS, (gsm.pS,), μS))
        + sum(kldiv.(qλ, (gsm.model.pθ.pλ,), μλ))
        + sum(kldiv.(qσ, (gsm.model.pθ.pσ,), μσ))
        + sum(kldiv.(qπ, (gsm.model.pθ.pπ,), μπ))
    )
end

function getstats(gsm::GSMPhonetic, qθ)
    qL, qS, qλ, qσ, qπ = qθ
    μλ = [μ(v) for v in qλ]
    μσ = [μ(v) for v in qσ]
    μπ = [μ(v) for v in qπ]
    μ.(qL), μ.(qS), μλ, μσ, μπ
end

function unpack(gsm::GSMPhonetic, qθ, Tθ)
    qL, qS, qλ, qσ, qπ = qθ
    μL, μS, μλ, μσ, μπ = Tθ

    L = getindex.(unpack.(qL, μL), 1)
    S = getindex.(unpack.(qS, μS), 1)
    λ̂ = [vcat(reparamtrick(unpack(q, μ)), 1) for (q, μ) in zip(qλ, μλ)]
    σ̂ = [vcat(reparamtrick(unpack(q, μ)), 1) for (q, μ) in zip(qσ, μσ)]
    π̂ = [vcat(reparamtrick(unpack(q, μ)), 1) for (q, μ) in zip(qπ, μπ)]

    L, S, λ̂, σ̂, π̂
end

function _llh(gsm, labels, x::AbstractVector, uTθ)
    L, S, λ̂, σ̂, π̂ = uTθ
    (l, s, p) = labels
    P = sum(L .* λ̂[l]) + sum(S .* σ̂[s])
    ψ = P' * π̂[p]
    η = gsm.f(ψ)
    loglikelihood(gsm.model, x, η)
end

function loglikelihood(gsm::GSMPhonetic, groups, uTθ)
    sum(groups) do labels_data
        (l, s, p), data = labels_data
        L, S, λ̂, σ̂, π̂ = uTθ
        P = sum(L .* λ̂[l]) + sum(S .* σ̂[s])
        ψ = P' * π̂[p]
        η = gsm.f(ψ)
        sum(loglikelihood(gsm.model, data, η))
    end
end

function newposterior(gsm::GSMPhonetic, qθ::Tuple, ∇μ; lrate=1, clip=true)
    q₀L, q₀S, q₀λ, q₀σ, q₀π = qθ
    ∇μL, ∇μS, ∇μλ, ∇μσ, ∇μπ = ∇μ

    qL = newposterior.((gsm,), q₀L, ∇μL; lrate=lrate, clip=clip)
    qS = newposterior.((gsm,), q₀S, ∇μS; lrate=lrate, clip=clip)
    qλ = newposterior.((gsm,), q₀λ, ∇μλ; lrate=lrate, clip=clip)
    qσ = newposterior.((gsm,), q₀σ, ∇μσ; lrate=lrate, clip=clip)
    qπ = newposterior.((gsm,), q₀π, ∇μπ; lrate=lrate, clip=clip)

    qL, qS, qλ, qσ, qπ
end

