module BayesianModels

#######################################################################
# Setup/Utilities

# Dependencies
using BasicDataLoaders
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

export InvertibleMap
include("invmap.jl")

#######################################################################
# BayesianModels generic object

include("bmobj.jl")

#######################################################################
# Model

export AbstractModel
export ModelList

export basemeasure
export vectorize
export statistics
export loglikelihood

include("model.jl")

#######################################################################
# Bayesian parameter

export AbstractParam
export ParamList
export BayesParam
export ConstParam

export getparams
export isbayesparam

include("params/params.jl")

#include("bayesparam.jl")


#######################################################################
# Objective function

export elbo
export ∇elbo
export gradstep

include("elbo.jl")

#######################################################################
# Models

export Normal
include("models/normal.jl")

#export AffineTransform
#include("models/affinetransform.jl")

#export NormalDiag
#include("models/normaldiag.jl")

#export Mixture
#include("models/mixture.jl")

end # module

