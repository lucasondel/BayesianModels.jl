module BayesianModels

using BasicDataLoaders
using Distributed
using ExpFamilyDistributions
using LinearAlgebra
using StatsFuns: logsumexp

#######################################################################
# Model

export AbstractModel
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

export loglikelihood
export fit!

export AffineTransform
include("models/affinetransform.jl")

export PPCA
include("models/ppca.jl")

export PLDA
export hstats
export ustats
export update_u!
export wstats_within_class
export update_W_within_class!
export wstats_across_class
export update_W_across_class!
include("models/plda.jl")

export NormalDiag
export vectorize
export statistics
include("models/normal.jl")

export Mixture
include("models/mixture.jl")

end # module

