# Implementation of Pólya-Gamma augmentation for the logistic sigmoid
# function.

function pgmean(
    c::Union{Vector{T}, Matrix{T}};
    b::Real = 1
) where T <: AbstractFloat
    retval = (b ./ (2 * c)) .* tanh.(c ./ 2)

    # When c = 0 the mean is not defined but can be extended by
    # continuity observing that lim_{x => 0} (e^(x) - 1) / x = 0             !
    # which lead to the mean = b / 4
    idxs = isnan.(retval)
    retval[idxs] .= b / 4
    return retval
end

# Log-normalizer of PolyaGamma distributions
function pglognorm(
    c::Union{Vector{T}, Matrix{T}};
    b::Real = 1
) where T <: AbstractFloat
    return -b .* (log1pexp.(c) .- log(2) .- c ./ 2)
end

"""
    pgaugmentation(ψ²[, b = 1])

Return the expected value of `ω` and the KL divergence posterior/prior.
`b` is the parameter of the Pólya-Gamma prior.
"""
function pgaugmentation(
    Ψ²::Union{Vector{T}, Matrix{T}};
    b::Real = 1
) where T <: AbstractFloat
    c = sqrt.(Ψ²)
    E_ω = pgmean(c)
    kl = -.5 .* (c.^2) .* E_ω .- pglognorm(c)
    E_ω, kl
end

