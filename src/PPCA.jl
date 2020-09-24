module PPCA

using ExpFamilyDistributions
using LinearAlgebra
using StatsFuns: log1pexp

#######################################################################
# Pólya-Gamma augmentation

export pgaugmentation

include("pgaugment.jl")

#######################################################################
# Model

export PCCAModel
export PPCAModel

include("pcca.jl")

#######################################################################
# Inference

export inferh
export stats_W
export stats_λ
export stats_B
export loglikelihood

include("inference.jl")

end # module

