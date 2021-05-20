# SPDX-License-Identifier: MIT

module BayesianModels

using AutoGrad
import ExpFamilyDistributions
const EFD = ExpFamilyDistributions
using LinearAlgebra
import StatsFuns

# Primitive to differentiate packed pos. def. matrix
@primitive EFD.matrix(diagM, trilM),dM diag(dM) EFD.vec_tril(dM) + EFD.vec_tril(dM')
@primitive EFD.inv_vec_tril(v),dM EFD.vec_tril(dM)
@primitive EFD.vec_tril(M),dv EFD.inv_vec_tril(dv)

logsumexp_dim1(x) = StatsFuns.logsumexp(x, dims = 1)
@primitive logsumexp_dim1(x),dy,y (dy .* exp.(x .- y))

include("bmobj.jl")

export getparams, isbayesianparam
include("params/params.jl")
include("params/bayesparam.jl")
include("params/constparam.jl")

export loglikelihood, predict
include("models/models.jl")

export Mixture, Normal, NormalDiag
include("models/mixture.jl")
include("models/normal.jl")


export elbo, ∇elbo, gradstep
include("elbo.jl")

end # module
