module PPCA

using BasicDataLoaders
using Distributed
using ExpFamilyDistributions
using LinearAlgebra

#######################################################################
# Objective function

export elbo
export cost_reg

include("elbo.jl")

#######################################################################
# Model

export AbstractPPCAModel
export PPCAModel
export PPCAModelHP
export θposteriors
export hposteriors

include("model.jl")

#######################################################################
# Accumulating and update functions

export hposteriors
export wposteriors!
export λposterior!
export αposteriors!
export wstats
export λstats
export αstats

include("accumulators.jl")

#######################################################################
# Inference

export fit!

include("inference.jl")

end # module

