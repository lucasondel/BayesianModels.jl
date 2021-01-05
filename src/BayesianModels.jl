module BayesianModels

using BasicDataLoaders
using Distributed
using ExpFamilyDistributions
using LinearAlgebra

#######################################################################
# Bayesian parameter

export BayesParam
export getparams

include("bayesparam.jl")

#######################################################################
# Objective function

export elbo
export cost_reg

include("elbo.jl")

#######################################################################
# Models

export loglikelihood
export fit!

export AffineTransform
include("models/affinetransform.jl")

export PPCA
include("models/ppca.jl")

end # module

