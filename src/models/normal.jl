# SPDX-License-Identifier: MIT

struct NormalModel{T} <: AbstractModel
    pθ::T
end

function loglikelihood(::NormalModel, X::AbstractMatrix, uTθ)
    Λμ, μᵀΛμ, Λ, logdetΛ = uTθ
    D = size(X, 1)
    XᵀΛX = sum(X .* (Λ * X), dims=1)
    μᵀΛX = Λμ' * X
    -(1/2)*(XᵀΛX .+ (μᵀΛμ - logdetΛ + D*log(2π))) .+ μᵀΛX
end

#######################################################################
# GSM prior/posterior

#getstats(model::NormalModel, pθ::NormalWishart) = μ(pθ)
#unpack(model::NormalModel, pθ::NormalWishart, Tθ) = unpack(pθ, Tθ)
#
#function newposterior(::NormalModel, qθ::NormalWishart, ∇μ::AbstractVector;
#                      lrate=1)
#    T = typeof(qθ)
#    ηq = η(qθ)
#    ∇̃ξ = ForwardDiff.derivative(t -> ξ(qθ, ηq + t*∇μ), 0)
#    ξᵗ⁺¹ = ξ(qθ, ηq) + lrate*∇̃ξ
#    NormalWishart(unpack(qθ, ξᵗ⁺¹)...)
#end
