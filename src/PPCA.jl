module PPCA

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
# AffineTransform

export AffineTransform

include("affinetransform.jl")

#######################################################################
# Models

export loglikelihood
export fit!

export PPCAModel

include("model.jl")

end # module

