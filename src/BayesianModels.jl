# SPDX-License-Identifier: MIT

module BayesianModels

import Distributions
import ForwardDiff
using LinearAlgebra
using SpecialFunctions
using Zygote

include("distributions/expfamily.jl")

export μ, η, ξ, A, kldiv, sample, unpack

include("distributions/normal.jl")
include("distributions/gamma.jl")
include("distributions/wishart.jl")
include("distributions/normalwishart.jl")

export Normal, Gamma, Wishart, NormalWishart

include("models/model.jl")

export loglikelihood
export elbo
export getstats

include("models/normal.jl")

export NormalModel

end # module
