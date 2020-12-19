module PPCA

using Distributed
using ExpFamilyDistributions
using LinearAlgebra

#######################################################################
# Model

export AbstractPPCAModel
export PPCAModel
export PPCAModelHP
export θposteriors
export hposteriors

include("model.jl")

#######################################################################
# Objective function

export elbo

include("elbo.jl")

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

