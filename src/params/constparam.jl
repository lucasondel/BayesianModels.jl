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
struct ConstParam{T<:AbstractVector} <: AbstractParameter
    μ::T
end

