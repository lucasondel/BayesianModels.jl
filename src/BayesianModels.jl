# SPDX-License-Identifier: MIT

module BayesianModels

import Distributions
import ForwardDiff
using LinearAlgebra
using SpecialFunctions
using Zygote

# Custom type for positive definite matrix.
include("utils/clowertri.jl")

export vech
export CompressedTriangularMatrix

include("distributions/expfamily.jl")

export η
export μ
export ξ
export A
export kldiv
export sample
export unpack

include("distributions/normal.jl")
include("distributions/gamma.jl")
include("distributions/wishart.jl")
include("distributions/normalwishart.jl")
include("distributions/jointnormal.jl")

export JointNormal
export JointNormalDiag
export Normal
export NormalDiag
export NormalIso
export Gamma
export Wishart
export NormalWishart

include("models/model.jl")

export loglikelihood
export elbo
export getstats

include("models/normal.jl")
include("models/gsm.jl")

export NormalModel
export GSM

end # module
