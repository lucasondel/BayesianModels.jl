# SPDX-License-Identifier: MIT

struct NormalModel <: AbstractModel end
const normal = NormalModel()

function loglikelihood(::NormalModel, xₙ::AbstractVector, Tθ)
    Λμ, μᵀΛμ, Λ, logdetΛ = Tθ
    D = length(xₙ)
    -(1/2)*(xₙ'*Λ*xₙ + μᵀΛμ - logdetΛ + D*log(2π)) + dot(xₙ, Λμ)
end

#######################################################################
# Normal-Wishart prior/posterior

getstats(::NormalModel, pθ::NormalWishart) = μ(pθ)
getstats(::NormalModel, pθs::NTuple{N, <:NormalWishart}) where N =
    vcat(μ.(pθs)...)
unpack(::NormalModel, pθ::NormalWishart, Tθ) = unpack(pθ, Tθ)
unpack(m::NormalModel, pθs::NTuple{N, <:NormalWishart}, Tθs) where N =
    unpack.(pθs, eachcol(reshape(Tθs, :, N)))


function newposterior(::NormalModel, qθ::NormalWishart, ∇μ::AbstractVector; lrate=1)
    T = typeof(qθ)
    ηq = η(qθ)
    ∇̃ξ = ForwardDiff.derivative(t -> ξ(qθ, ηq + t*∇μ), 0)
    ξᵗ⁺¹ = ξ(qθ, ηq) + lrate*∇̃ξ
    NormalWishart(unpack(qθ, ξᵗ⁺¹)...)
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
