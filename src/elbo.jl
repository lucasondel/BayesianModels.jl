# PPCA - Computation of the objective function, the Evidence Lower
# Bound (ELBO).
#
# Lucas Ondel 2020

# Per dimension log-likelihood
function _llh_d(::AbstractPPCAModel, x::Real, Tŵ, Tλ, Th)
    λ, lnλ = Tλ
    h, hhᵀ = Th

    # Extract the bias parameter
    wwᵀ = Tŵ[2][1:end-1, 1:end-1]
    w = Tŵ[1][1:end-1]
    μ = Tŵ[1][end]
    μ² = Tŵ[2][end, end]

    x̄ = dot(w, h) + μ
    lognorm = (-log(2π) + lnλ - λ*(x^2))/2
    K = λ*(x̄*x - dot(w, h)*μ - (dot(vec(hhᵀ), vec(wwᵀ)) + μ²)/2)

    lognorm + K
end

function _llh(m, x::AbstractVector, Tŵs::AbstractVector, Tλ, Th)
    f = (a,b) -> begin
        xᵢ, Tŵᵢ = b
        a + _llh_d(m, xᵢ, Tŵᵢ, Tλ, Th)
    end
    return foldl(f, zip(x, Tŵs), init = 0)
end

function loglikelihood(m::AbstractPPCAModel, X, θposts, hposts)
    _llh.(
        [m],
        X,
        [[gradlognorm(p, vectorize = false) for p in θposts[:w] ]],
        [gradlognorm(θposts[:λ], vectorize = false)],
        [gradlognorm(p, vectorize = false) for p in hposts]
   ) - kldiv.(hposts, [m.hprior])
end


function elbo(m::PPCAModel, dataloader, θposts; detailed = false)
    L = @distributed (+) for X in dataloader
        hposts = hposteriors(m, X, θposts)
        sum(loglikelihood(m, X, θposts, hposts))
    end

    if typeof(θposts[:w][1]) <: ExpFamilyDistribution
        KL += sum(kldiv.(θposts[:w], [m.wprior]))
    else
        KL -= sum(
            p -> ExpFamilyDistributions.loglikelihood(m.wprior, p.μ),
            θposts[:w]
        )
    end

    KL += kldiv(θposts[:λ], m.λprior)
    detailed ? (L - KL, L, KL) : L - KL
end

function elbo(m::PPCAModelHP, dataloader, θposts; detailed = false)
    L = @distributed (+) for X in dataloader
        hposts = hposteriors(m, X, θposts)
        sum(loglikelihood(m, X, θposts, hposts))
    end
    KL = 0

    KL += sum(kldiv.(θposts[:α], [m.αprior]))

    if typeof(θposts[:w][1]) <: ExpFamilyDistribution
        KL += sum(kldiv.(θposts[:w], [m.wprior]))
    else
        KL -= sum(
            p -> ExpFamilyDistributions.loglikelihood(m.wprior, p.μ),
            θposts[:w]
        )
    end

    KL += kldiv(θposts[:λ], m.λprior)
    detailed ? (L - KL, L, KL) : L - KL
end

"""
    elbo(model, dataloader, θposts, [, detailed = false])

Compute the Evidence Lower-BOund (ELBO) of the model. If `detailed` is
set to `true` returns a tuple `elbo, loglikelihood, KL`.
"""
elbo

