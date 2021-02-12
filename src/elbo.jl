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

function elbo(m, dataloader::DataLoader, args...; detailed = false)
    L = @distributed (+) for X in dataloader
        sum(loglikelihood(m, X, args...))
    end
    C = cost_reg(m)
    detailed ? (L - C, L, C) : L - C
end

function elbo(m, args...; detailed = false)
    L = sum(loglikelihood(m, args...))
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

#######################################################################
# Natural gradient of the elbo.

function ∇elbo(model, args...; stats_scale = 1)
    stats = getparam_stats(model, args...)

    retval = Dict()
    for (param, s) in stats
        η₀ = naturalparam(param.prior)
        η = naturalparam(param.posterior)
        retval[param] = η₀ + stats_scale*s - η
    end
    retval
end

"""
    gradstep(param_grad; lrate)

"""
function gradstep(param_grad; lrate::Real)
    for (param, grad) in param_grad
        η⁰ = naturalparam(param.posterior)
        #θ⁰ = param.gradspace.f(η⁰)
        η¹ = η⁰ + lrate * grad
        # θ¹ = θ⁰ + lrage * grad
        #η¹ = param.gradspace.f_inv(θ¹)
        update!(param.posterior, η¹)
    end
end

