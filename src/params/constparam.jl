# SPDX-License-Identifier: MIT

# Constant parameter.
#
# Lucas Ondel 2021

"""
    struct ConstParam{T<:AbstractVector} <: AbstractParameter
        μ::T
    end

Constant parameter.

# Constructor

    ConstantParam(μ)

Create a constant parameter.
"""
struct ConstParameter{T<:AbstractVector} <: AbstractParameter
    μ::T
    _stats::Function
end

statistics(param::ConstParameter) = param._stats(param.μ)

