# Computation of the Evidence Lower Bound (ELBO).
#
# Lucas Ondel 2021

"""
    elbo(model, X[, detailed = false])

Compute the Evidence Lower-BOund (ELBO) of the model. If `detailed` is
set to `true` returns a tuple `elbo, loglikelihood, KL`.
"""
function elbo(model, args...; detailed = false, stats_scale=1)
    llh = sum(loglikelihood(model, args...))*stats_scale

    params = Zygote.@ignore filter(isbayesparam, getparams(model))
    KL = sum(param -> EFD.kldiv(param.posterior, param.prior, μ = param._μ),
             params)
    detailed ? (llh - KL, llh, KL) : llh - KL
end

"""
    ∇elbo(model, args...[, stats_scale = 1])

Compute the natural gradient of the elbo w.r.t. to the posteriors'
parameters.
"""
function ∇elbo(model, args...; stats_scale = 1)
    bayesparams = filter(isbayesparam, getparams(model))
    P = Params([param._μ for param in bayesparams])
    μgrads = gradient(() -> elbo(model, args..., stats_scale = stats_scale), P)

    grads = Dict()
    for param in bayesparams
        ∂𝓛_∂μ = μgrads[param._μ]
        η = EFD.naturalparam(param.posterior)
        J = FD.jacobian(param._grad_map, η)
        grads[param] = J * ∂𝓛_∂μ
    end
    grads
end

"""
    gradstep(param_grad; lrate)

Update the parameters' posterior by doing one natural gradient steps.
"""
function gradstep(param_grad; lrate::Real)
    for (param, ∇𝓛) in param_grad
        η⁰ = EFD.naturalparam(param.posterior)
        ξ⁰ = param._grad_map(η⁰)
        ξ¹ = ξ⁰ + lrate*∇𝓛
        η¹ = (param._grad_map^-1)(ξ¹)
        EFD.update!(param.posterior, η¹)
        param._μ[:] = EFD.gradlognorm(param.posterior)
    end
end

