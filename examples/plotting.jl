# SPDX-License-Identifier: MIT

# Helper functions for plotting 2D-Normal pdfs.
# Lucas Ondel, 2021

function normalShape(μ, Σ, σ = 1)
    θ = LinRange(0, 2π, 1000)
    λ, U = eigen(Σ)
    B = U * diagm(σ * sqrt.(λ))
    circle = hcat(sin.(θ), cos.(θ))'
    contour = B * circle .+ μ
    contour[1, :], contour[2, :]
end

function plotnormal!(μ::AbstractVector, Σ::AbstractMatrix; σ = 1, kwargs...)
    plot!(normalShape(μ, Σ, σ), seriestype = [:shape];  kwargs...)
end
