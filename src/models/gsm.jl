# Generalized Subspace Model (GSM)
#
# Lucas Ondel 2021

struct GSM{ModelType} <: AbstractModel
    transform::AffineTransform
    f::Function
    vec2params::Function
end

function newmodel(gsm::GSM{ModelType}, e::BayesParam) where ModelType
    η = gsm.f(statistics(e) |> gsm.transform)
    params = gsm.vec2params(η)
    ModelType(params...)
end

function newmodel(gsm::GSM{ModelType}, embeddings) where ModelType
    E = [statistics(e) for e in embeddings]
    H = gsm.f.(E |> gsm.transform)
    [ModelType(gsm.vec2params(η)...) for η in H]
end

#######################################################################
# Model interface

function loglikelihood(gsm::GSM, embeddings, Xs)
    models = newmodel(gsm, embeddings)
    [sum(loglikelihood(models[k], Xs[k])) for k in 1:length(models)]
end

