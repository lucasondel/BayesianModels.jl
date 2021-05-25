# SPDX-License-Identifier: MIT

module BayesianModels

using AutoGrad
using CUDA
import ExpFamilyDistributions
const EFD = ExpFamilyDistributions
using LinearAlgebra
import StatsFuns

include("bmobj.jl")

export getparams, isbayesianparam
include("params/params.jl")
include("params/bayesparam.jl")
include("params/constparam.jl")

export gpu!, cpu!, init_gpu
include("utils.jl")

export loglikelihood, predict
include("models/models.jl")

export Mixture, Normal, NormalDiag
include("models/mixture.jl")
include("models/normal.jl")


export elbo, ∇elbo, gradstep
include("elbo.jl")

end # module
