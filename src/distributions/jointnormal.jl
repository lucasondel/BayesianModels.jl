# SPDX-License-Identifier: MIT

# Joint Normal distribution with fixed identity covariance matrix.

struct JointNormalFixedCov{T1<:AbstractMatrix} <: AbstractNormal
    M::T1 # QxD mean
end

η(p::JointNormalFixedCov) = vec(p.M)
ξ(p::JointNormalFixedCov, η) = η

function unpack(p::JointNormalFixedCov, μ)
	Q, D = size(p.M)
    X = reshape(μ, Q, D)
	(X=X,)
end

A(p::JointNormalFixedCov, η) = (1/2) * dot(η, η)

function sample(p::JointNormalFixedCov)
	Q, D = size(p.M)
    p.M + randn(Q, D)
end
