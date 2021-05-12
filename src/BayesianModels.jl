# SPDX-License-Identifier: MIT

module BayesianModels

#######################################################################
# Dependencies

import ExpFamilyDistributions
const EFD = ExpFamilyDistributions
using LinearAlgebra
using StatsFuns: logsumexp
using Zygote

# Make sure that these function are differentiable by Zygote
using Zygote: @adjoint
@adjoint EFD.inv_vec_tril(M) = EFD.inv_vec_tril(M), Δ -> (EFD.vec_tril(Δ),)
@adjoint EFD.vec_tril(v) = EFD.vec_tril(v), Δ -> (EFD.inv_vec_tril(Δ),)

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
include("params/bayesparam.jl")
include("params/constparam.jl")

#######################################################################
# Model

export loglikelihood
export predict

include("models/models.jl")

export Mixture
export Normal
export NormalDiag

include("models/mixture.jl")
include("models/normal.jl")

#######################################################################
# Optimization API

export elbo
export ∇elbo
export gradstep

include("elbo.jl")

end # module

