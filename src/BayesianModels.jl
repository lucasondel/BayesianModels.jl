module BayesianModels

#######################################################################
# Setup/Utilities

import ExpFamilyDistributions
using LinearAlgebra
using StatsFuns: logsumexp
using Zygote

const EFD = ExpFamilyDistributions

# Make sure that these function are differentiable by Zygote
using Zygote: @adjoint
@adjoint EFD.inv_vec_tril(M) = EFD.inv_vec_tril(M), Δ -> (EFD.vec_tril(Δ),)
@adjoint EFD.vec_tril(v) = EFD.vec_tril(v), Δ -> (EFD.inv_vec_tril(Δ),)

include("invmap.jl")

#######################################################################
# BayesianModels generic object

include("bmobj.jl")

#######################################################################
# Model parameter

export AbstractParameter
export ParameterList
export BayesianParameter
export ConstParameter

export getparams
export isbayesianparam

include("params/params.jl")

#######################################################################
# Model

export loglikelihood

include("models/models.jl")

#######################################################################
# Objective function

export elbo
export ∇elbo
export gradstep

include("elbo.jl")

end # module

