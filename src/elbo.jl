# SPDX-License-Identifier: MIT

"""
    elbo(model, X[, detailed = false, stats_scale = 1, params = params])

Compute the Evidence Lower-BOund (ELBO) of the model. If `detailed` is
set to `true` returns a tuple `elbo, loglikelihood, KL`. If
`params` is provided the function will return the natural gradient
for those parameters.
"""
elbo

function elbo(model, args...; cache = Dict(), detailed = false, stats_scale = 1)
    llh = loglikelihood(model, args..., cache)
    T = eltype(llh)
    sllh = sum(llh)*T(stats_scale)

    params = filter(isbayesianparam, getparams(model))
    KL = sum([EFD.kldiv(param.posterior, param.prior, μ = param.μ)
              for param in params])
    detailed ? (sllh - KL, sllh, KL) : sllh - KL
end

function _diagonal(param)
    d = similar(EFD.realform(param))
    fill!(d, 1)
    Diagonal(d)
end

function ∇elbo(model, cache, params)
    grads_Tμ = ∇sum_loglikelihood(model, cache)
    grads = Dict()
    for param in params
        ηq = EFD.naturalform(param.posterior.param)
        ηp = EFD.naturalform(param.prior.param)
        ∂KL_∂Tμ = (ηq - ηp)
        ∂𝓛_∂Tμ = grads_Tμ[param] - ∂KL_∂Tμ
        #J = EFD.jacobian(param.posterior.param)
        J = _diagonal(param.posterior.param)
        grads[param] = J * ∂𝓛_∂Tμ
    end
    grads
end

"""
    gradstep(param_grad; lrate)

Update the parameters' posterior by doing one natural gradient steps.
"""
function gradstep(param_grad; lrate::Real)
    for (param, ∇𝓛) in param_grad
        ξ = param.posterior.param.ξ
        ξ[:] = ξ + lrate*∇𝓛
        param.μ.value = EFD.gradlognorm(param.posterior)
    end
end
