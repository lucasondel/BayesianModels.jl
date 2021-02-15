module BayesianModels

using BasicDataLoaders
using Distributed
import ExpFamilyDistributions
using LinearAlgebra
using StatsFuns: logsumexp

const EFD = ExpFamilyDistributions

#######################################################################
# BayesianModels generic object

include("bmobj.jl")

#######################################################################
# Model

export AbstractModel
export basemeasure
export vectorize
export statistics
export loglikelihood
export getparam_stats
export ModelList

include("model.jl")

#######################################################################
# Bayesian parameter

export BayesParam
export BayesParamList
export getparams

include("bayesparam.jl")

#######################################################################
# Objective function

export elbo
export cost_reg
export getparam_stats
export ∇elbo
export gradstep

include("elbo.jl")

#######################################################################
# Models

export Normal
include("models/normal.jl")

#export NormalDiag
#include("models/normaldiag.jl")

#export Mixture
#include("models/mixture.jl")

end # module

