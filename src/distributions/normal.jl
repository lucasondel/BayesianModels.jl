# SPDX-License-Identifier: MIT

abstract type AbstractNormal <: ExponentialFamilyDistribution end

struct Normal{T1<:AbstractVector,T2<:AbstractVector,T3<:AbstractVector} <: AbstractNormal
    μ::T1
    lnλ::T2 # Logarithm of the diagonal of the L matrix.
    vechL::T3 # Half-vectorization of the L matrix
end

function Normal(μ::AbstractVector, Σ::AbstractMatrix)
    L = cholesky(inv(Σ)).L
    lnλ = log.(diag(L))
    vechL = vech(L, 1)
    Normal(μ, lnλ, vechL)
end

# Utility function to get the standard parameters from the expectation
# parameters
function stdparams(n::Normal, packed_μ)
    unpacked_μ = unpack(n,  packed_μ)
    x̄, diagx̄x̄ᵀ, vechx̄x̄ᵀ = unpacked_μ
    x̄x̄ᵀ = diagm(diagx̄x̄ᵀ) + CompressedSymmetric(size(x̄, 1), 1, vechx̄x̄ᵀ)
    x̄, Symmetric(x̄x̄ᵀ - x̄ * x̄')
end

Normal(μ::AbstractVector) =
    Normal(μ, copyto!(similar(μ, size(μ, 1), size(μ, 1)), I))

Normal(D::Int) = Normal(zeros(D), Matrix(I, D, D))

function η(p::AbstractNormal)
    D = length(p.μ)
    L = diagm(exp.(p.lnλ)) .+ CompressedLowerTriangular(D, 1, p.vechL)
    Λ = L * L'
    vcat(Λ*p.μ, -(1/2)*diag(Λ), -vech(Λ, 1))
end

function ξ(p::AbstractNormal, η)
	D = length(p.μ)
    diagΛ = -2 * diagm(η[D+1:2*D])
    shΛ = - CompressedSymmetric(D, 1, η[2*D+1:end])
    Λ = diagΛ + shΛ
    L = cholesky(Λ).L
    L⁻¹ = inv(L)
    μ = L⁻¹' * L⁻¹ * η[1:D]
    vcat(μ, log.(diag(L)), vech(L, 1))
end

function unpack(p::AbstractNormal, μ)
	D = length(p.μ)
	x = μ[1:D]
    diagxxᵀ = μ[D+1:2*D]
    vechxxᵀ = μ[2*D+1:end]
	(x=x, diagxxᵀ=diagxxᵀ, vechxxᵀ=vechxxᵀ)
end

function A(p::AbstractNormal, η)
	D = length(p.μ)

    diagΛ = -2 * diagm(η[D+1:2*D])
    shΛ = -CompressedSymmetric(D, 1, η[2*D+1:end])
    Λ = diagΛ + shΛ
    L = cholesky(Symmetric(Λ)).L
    L⁻¹ = inv(L)
    μ = L⁻¹' * L⁻¹ * η[1:D]

    #-(1/2)*logdet(Symmetric(Λ)) + (1/2)* μ' * η[1:D]
    -sum(log.(diag(L))) + (1/2)* μ' * η[1:D]
end

function sample(p::AbstractNormal)
	D = length(p.μ)
    L = diagm(exp.(p.lnλ)) .+ CompressedLowerTriangular(D, 1, p.vechL)
    Λ = Symmetric(L * L')
    p.μ + cholesky(inv(Λ)).L*randn(D)
end

struct NormalDiag{T1<:AbstractVector,T2<:AbstractVector} <: AbstractNormal
    μ::T1
	lnλ::T2
end

NormalDiag(μ::AbstractVector) =
    NormalDiag(μ, fill!(similar(μ, size(μ, 1), zero(eltype(μ)))))

NormalDiag(D::Int) = NormalDiag(zeros(D), zeros(D))

function η(p::NormalDiag)
    λ = exp.(p.lnλ)
	vcat(λ .* p.μ, -(1/2)*λ)
end

function ξ(p::NormalDiag, η)
	D = length(p.μ)
	λ = -2* η[D+1:end]
    μ = (1 ./ λ) .* η[1:D]
    vcat(μ, log.(λ))
end

function unpack(p::NormalDiag, μ)
	D = length(p.μ)
	x² = μ[D+1:end]
	x = μ[1:D]
    (x=x, x²=x²)
end

function A(p::NormalDiag, η)
	D = length(p.μ)

	λ = -2 * η[D+1:end]
	μ = (1 ./ λ) .* η[1:D]

    -(1/2) * sum(log.(λ)) + (1/2) * sum(μ .* λ .* μ)
end

function sample(p::NormalDiag)
	D = length(p.μ)
    p.μ + sqrt.(1 ./ exp.(p.λ)) .* randn(D)
end

struct NormalIso{T1<:AbstractVector,T2<:Real} <: AbstractNormal
    μ::T1
	lnλ::T2
end

function η(p::NormalIso)
    λ = exp.(p.lnλ)
	vcat(λ .* p.μ, -(1/2)*λ)
end

function ξ(p::NormalIso, η)
	D = length(p.μ)
	λ = -2* η[D+1]
    μ = (1 / λ) .* η[1:D]
    vcat(μ, log(λ))
end

function unpack(p::NormalIso, μ)
	D = length(p.μ)
	x² = μ[D+1]
	x = μ[1:D]
    (x=x, x²=x²)
end

function A(p::NormalIso, η)
	D = length(p.μ)

	λ = -2 * η[D+1]
	μ = (1 / λ) .* η[1:D]

    -(D/2) * log(λ) + (1/2) * sum(μ .* λ .* μ)
end

function sample(p::NormalIso)
	D = length(p.μ)
    p.μ + sqrt.(1 / exp.(p.lnλ)) .* randn(D)
end

