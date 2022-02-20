# SPDX-License-Identifier: MIT

# Joint Normal distribution with fixed identity covariance matrix.

struct JointNormalFixedCov{T1<:AbstractMatrix} <: AbstractNormal
    M::T1 # QxD mean
end

η(p::JointNormalFixedCov) = vec(p.M)

function ξ(p::JointNormalFixedCov, η)
	Q, D = size(p.M)
    M = reshape(η, Q, D)
    vec(M)
end

function unpack(p::JointNormalFixedCov, μ)
	Q, D = size(p.M)
    X = reshape(μ, Q, D)
	(X=X,)
end

function A(p::JointNormalFixedCov, η)
	Q, D = size(p.M)
    M = reshape(η, Q, D)
    (1/2)*sum(M .* M)
end

function sample(p::JointNormalFixedCov)
	Q, D = size(p.M)
    p.M + randn(Q, D)
end
