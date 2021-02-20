# Constant parameter.
#
# Lucas Ondel 2021

"""
    struct ConstParam{T} <: AbstractParam{T}
        θ::V <: AbstractVector{T}
        ...
    end

Constant parameter.

# Constructor

    ConstantParam(θ, stats_fn = ...::Function)

Create a constant parameter. The function `stats_fn` takes θ as input
and returns the statistics of the parameters for the model.
"""
struct ConstParam{T} <: AbstractParam{T}
    θ::V where V <: AbstractVector{T}
    _stats_fn::Function

    function ConstParam(θ::AbstractVector{T}; stats_fn) where T
        new{T}(θ, stats_fn)
    end
end

statistics(param::ConstParam) = param._stats_fn(param.θ)

