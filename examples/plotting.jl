using LinearAlgebra
using Plots

export plotnormal2d

"""
    plotnormal2d(p, μ::AbstractVector, Σ::AbstractMatrix; ncontours=2, args...)
Plot the contours of a 2d Normal density. `ncontours` is the number of
contour line to plots.
"""
function plotnormal2d!(p, μ::AbstractVector, Σ::AbstractMatrix; ncontours=2, color="blue", label="", args...)
    μ = μ[1:2]
    Σ = Σ[1:2,1:2]
    λ, U = eigen(Σ)
    for i in 1:ncontours
        B = U * diagm(i * sqrt.(λ))
        θ = range(0, stop=2 * pi, length=1000)
        circle = hcat(sin.(θ), cos.(θ))'
        contour = B * circle .+ μ
        plot!(p, contour[1, :], contour[2, :]; color=color, label=i > 1 ? "" : label, args...)
    end
    p
end

"""
    plotmodel!(p, model, θposts[, color = :green])

Plot multivariate Gaussian resulting from the marginalization of the latent prior (standard Normal)
while taking Maximum A Posteriori of the other parameters.
"""
function plotmodel!(p, model::AbstractPPCAModel{T,D,Q}, θposts;
                    color = :green) where {T,D,Q}
    λ = mean(θposts[:λ])
    W = hcat([θposts[:w][d].μ for d in 1:D]...)
    m = W[end, :]
    S = (1/λ) * Matrix{T}(I, 2, 2)
    for d in 1:Q
        w = W[1:end-1, :][d, :]
        w = w / sqrt(w[1]^2 + w[2]^2)
        o = m
        #plot!(p, [o[1], w[1] + m[1]], [o[2], w[2] + m[2]], arrow = (0.4, 0.4), color = :black)
        plot!(p, [o[1] - 100*w[1], 100*w[1] + m[1]], [o[2] - 100*w[2], 100*w[2] + m[2]], color = :black)
        S += w * w'
    end
    plotnormal2d!(p, m, S, color = color)
end

"""
    plotnormal!(p, normals[, color = :red, alpha = 0.4])

Plot a set of 2D Normal distribution
"""
function plotnormals!(p, normals; color = :red, alpha = 0.4)
    for n in normals
        plotnormal2d!(p, n.μ, n.Σ, alpha = alpha, color = color)
    end
    p
end

