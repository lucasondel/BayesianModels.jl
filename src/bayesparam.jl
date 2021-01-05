# Bayesian parameter, i.e. parameter with a prior/posterior.
#
# Lucas Ondel 2021

abstract type AbstractBayesParam{T1,T2} end

"""
    struct BayesParam
        prior
        posterior
    end

Bayesian parameter, i.e. a parameter with a prior and a (variational)
posterior.
"""
struct BayesParam{T1,T2} <: AbstractBayesParam{T1,T2}
    prior::T1
    posterior::T2
end

function Base.show(io::IO, ::MIME"text/plain", p::BayesParam)
    println(io, "$(typeof(p)):")
    println(io, "  prior: $(p.prior)")
    println(io, "  posterior: $(p.posterior)")
end

"""
    getparams(obj)

Returns a list of a all the [`BayesParam`](@ref) in `obj` and its
attributes.
"""
function getparams(obj)
    params = BayesParam[]
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        if typeof(prop) <: BayesParam
            push!(params, prop)
        elseif typeof(prop) <: AbstractArray{<:BayesParam}
            push!.([params], prop)
        else
            push!.([params], getparams(prop))
        end
    end
    params
end

