# PPCA - Computation of the objective function, the Evidence Lower
# Bound (ELBO).
#
# Lucas Ondel 2020

# Regularization cost of the posterior. It will depends on the type of
# of the variational posterior:
#   * std posterior -> classical VB inference -> KL(q || p)
#   * δ-distribution -> Maximum A Posteriori -> - ln p( q.μ )
cost_reg(q::ExpFamilyDistribution, p::ExpFamilyDistribution) = kldiv(q, p)
function cost_reg(q::δDistribution, p::ExpFamilyDistribution)
    -ExpFamilyDistributions.loglikelihood(p, q.μ)
end
cost_reg(qs::AbstractVector, p::ExpFamilyDistribution) = sum(cost_reg.(qs, [p]))

function elbo(m, dataloader::DataLoader, θposts; detailed = false)
    L = @distributed (+) for X in dataloader
        hposts = hposteriors(m, X, θposts)
        sum(loglikelihood(m, X, θposts, hposts))
    end
    C = sum(pair -> cost_reg(pair.second, getproperty(m, pair.first)), θposts)
    detailed ? (L - C, L, C) : L - C
end

function elbo(m, X::AbstractVector, θposts; detailed = false)
    hposts = hposteriors(m, X, θposts)
    L = sum(loglikelihood(m, X, θposts, hposts))
    C = sum(pair -> cost_reg(pair.second, getproperty(m, pair.first)), θposts)
    detailed ? (L - C, L, C) : L - C
end

"""
    elbo(model, dataloader, θposts[, detailed = false])
    elbo(model, X, θposts[, detailed = false])

Compute the Evidence Lower-BOund (ELBO) of the model. If `detailed` is
set to `true` returns a tuple `elbo, loglikelihood, KL`.
"""
elbo

