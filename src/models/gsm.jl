# SPDX-License-Identifier: MIT

struct GSM{T1,T2,T3} <: AbstractModel
    f::Function
    pW::T1
    pb::T2
    models::T3
end

function kldiv(gsm::GSM, qθ, Tθ)
    qW, qb, qH = qθ
    μW, μb, μH = Tθ
    kldiv(qW, gsm.pW, μW) + kldiv(qb, gsm.pb, μb) + sum(kldiv.(gsm.models, qH, μH))
end

function reparamtrick(μx)
    x̄, x̄² = μx
    T = eltype(x̄)
    mm = x̄ .^ 2
    x̄ #+ sqrt.(x̄² .- mm) .* randn(T, size(x̄)...)
end

function getstats(gsm::GSM, qθ)
    qW, qb, qH = qθ
    μ(qW), μ(qb), μ.(qH)
end

function unpack(gsm::GSM, qθ, Tθ)
    qW, qb, qH = qθ
    μW, μb, μH = Tθ
    T = eltype(μW)

    W = reparamtrick(unpack(qW, μW))
    b = reparamtrick(unpack(qb, μb))
    H = hcat(reparamtrick.(unpack.(qH, μH))...)
    #Ĥ = vcat(H, ones(T, 1, size(H, 2)))

    gsm.f.(eachcol(W' * H .+ b))
end

loglikelihood(gsm::GSM, Xs, uTθ) = sum(loglikelihood.(gsm.models, Xs, uTθ))

function newposterior(gsm::GSM, qθ::Tuple, ∇μ; lrate=1)
    q₀W, q₀b, q₀H = qθ
    ∇μW, ∇μb, ∇μH = ∇μ

    qW = newposterior(gsm, q₀W, ∇μW; lrate)
    qb = newposterior(gsm, q₀b, ∇μb; lrate)
    qH = newposterior.(gsm.models, q₀H, ∇μH; lrate)

    qW, qb, qH
end

