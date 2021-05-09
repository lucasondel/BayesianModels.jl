module BayesianModels

#######################################################################
# Setup/Utilities

# Dependencies
using Distributed
import ExpFamilyDistributions
import ForwardDiff
using LinearAlgebra
using StatsFuns: logsumexp
using Zygote

const EFD = ExpFamilyDistributions
const FD = ForwardDiff

# Make sure that these function are differentiable by Zygote
using Zygote: @adjoint
@adjoint EFD.inv_vec_tril(M) = EFD.inv_vec_tril(M), Δ -> (EFD.vec_tril(Δ),)
@adjoint EFD.vec_tril(v) = EFD.vec_tril(v), Δ -> (EFD.inv_vec_tril(Δ),)

include("invmap.jl")

#######################################################################
# BayesianModels generic object

include("bmobj.jl")

#######################################################################
# Model

export AbstractModel
export ModelList

#######################################################################
# Model parameter

export AbstractParameter
export ParameterList
export BayesParameter
export ConstParameter

export getparams
export isbayesparam

include("params/params.jl")

#######################################################################
# Model

export AbstractModel
export loglikelihood

export ModelList

export Normal

export HNormalDiag
export AffineTransform

export GSM
export newmodel

include("models/models.jl")

#######################################################################
# Objective function

export elbo
export ∇elbo
export gradstep

include("elbo.jl")

end # module

