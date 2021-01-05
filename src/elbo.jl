# Computation of the Evidence Lower Bound (ELBO).
#
# Lucas Ondel 2021

# Regularization cost of the posterior. It will depends on the type of
# of the variational posterior:
#   * std posterior  -> classical VB inference -> KL(q || p)
#   * δ-distribution -> Maximum A Posteriori   -> -ln p( q.μ )
cost_reg(q::ExpFamilyDistribution, p::ExpFamilyDistribution) = kldiv(q, p)
cost_reg(q::δDistribution, p::ExpFamilyDistribution) = -ExpFamilyDistributions.loglikelihood(p, q.μ)
function cost_reg(model)
    params = getparams(model)
    cost = 0
    for param in params
        cost += cost_reg(param.posterior, param.prior)
    end
    cost
end

function elbo(m, dataloader::DataLoader; detailed = false)
    L = @distributed (+) for X in dataloader
        sum(loglikelihood(m, X))
    end
    C = cost_reg(m)
    detailed ? (L - C, L, C) : L - C
end

function elbo(m, X::AbstractVector; detailed = false)
    L = sum(loglikelihood(m, X))
    C = cost_reg(m)
    detailed ? (L - C, L, C) : L - C
end

"""
    elbo(model, dataloader[, detailed = false])
    elbo(model, X[, detailed = false])

Compute the Evidence Lower-BOund (ELBO) of the model. If `detailed` is
set to `true` returns a tuple `elbo, loglikelihood, KL`.
"""
elbo

