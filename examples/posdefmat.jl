### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 0e3503f0-882f-11ec-1a62-13f539a38ae9
begin
	using Revise
	using Pkg
	Pkg.activate("../")

	using LinearAlgebra
	using BayesianModels
	using ForwardDiff
	using FiniteDiff
	using Zygote
end

# ╔═╡ be862e45-482e-438f-8aaa-49da31d32533
CompressedSymmetric(randn(3, 3), 2)

# ╔═╡ 90707c10-e754-4249-92d6-3538c5cef64f
f(M) = sum(vech(M, 1))

# ╔═╡ 32074133-0ece-4227-afe4-1fab700162a9
ForwardDiff.gradient(f, ones(3, 3))

# ╔═╡ 9a5b8cc5-a9eb-4db0-82c3-dfef223656a4
Zygote.gradient(f, ones(3, 3))

# ╔═╡ fc1e397a-94f1-44b6-a805-e2ec7e13f027
n = Normal(ones(2), ones(2), ones(1))

# ╔═╡ 6d0f5b95-28fa-4c8d-83f6-ed8645a42431
η(n)

# ╔═╡ bc9e78d9-1eca-4ba4-8a15-9c42fe16de42
ξ(n, η(n))

# ╔═╡ 0c63f4bb-d440-4c7c-870d-1c0c04a7fcdd
inv([1 1; 1 2])

# ╔═╡ e3438e5b-53f7-460c-af56-bede7958b099
ForwardDiff.jacobian(η -> ξ(n, η), η(n))

# ╔═╡ 16d82230-339b-4faa-a38b-7552443b89b5
Zygote.jacobian(η -> ξ(n, η), η(n))

# ╔═╡ 5ac40637-aa73-4166-a44e-80ba1585af1a
FiniteDiff.finite_difference_jacobian(η -> ξ(n, η), η(n))

# ╔═╡ 853fcf39-9a93-461d-96b9-acdceaa743c6
A(n, η(n))

# ╔═╡ 9fef2bd3-0ff7-42d8-a1c0-896fc2b68ebb
x, dxx, vxx = unpack(n, μ(n))

# ╔═╡ 6267fe50-5d77-4df6-8b8e-a83ec018bd05
Σ = diagm(dxx) + CompressedSymmetric(2, 1, vxx) - x*x'

# ╔═╡ 0ada5640-4534-4407-ac56-9b9c0a2e8650
Λ = inv(diagm(dxx) + CompressedSymmetric(2, 1, vxx) - x*x')

# ╔═╡ ba914642-2d7d-4483-ac4d-3bce4306ffca
L = cholesky(Λ).L

# ╔═╡ dac87f2d-82b8-449e-93e6-0a05bb78014b
gradient(BayesianModels.vech(ones(3, 3), 1)) do x
	L = CompressedTriangularMatrix(3, 1, x)
	sum(L*L')
end

# ╔═╡ ce06a55e-c1cd-412e-8d5d-62cbfb8b9db7
gradient(LowerTriangular(ones(3, 3))) do x
	sum(x*x')
end

# ╔═╡ e1001f96-d515-4a0d-bd91-399551d2122b
L * L'

# ╔═╡ b9ed0e5d-5ea0-49e4-8a5f-27c8d7752325
C = CompressedSymmetricMatrix(A)

# ╔═╡ 70f075ec-2ccd-435b-b2b0-b478bff2d223
inv(C)

# ╔═╡ 50488d99-e019-48a8-a340-d01a51c55ef6
BayesianModels.vech(A)

# ╔═╡ e4338dc0-cb33-4f38-9181-475bdb3c523b
M = BayesianModels.PosDefMatrix(A)

# ╔═╡ acac9b27-aca5-410a-80f9-7a10eac7612b
M

# ╔═╡ Cell order:
# ╠═0e3503f0-882f-11ec-1a62-13f539a38ae9
# ╠═be862e45-482e-438f-8aaa-49da31d32533
# ╠═90707c10-e754-4249-92d6-3538c5cef64f
# ╠═32074133-0ece-4227-afe4-1fab700162a9
# ╠═9a5b8cc5-a9eb-4db0-82c3-dfef223656a4
# ╠═fc1e397a-94f1-44b6-a805-e2ec7e13f027
# ╠═6d0f5b95-28fa-4c8d-83f6-ed8645a42431
# ╠═bc9e78d9-1eca-4ba4-8a15-9c42fe16de42
# ╠═0c63f4bb-d440-4c7c-870d-1c0c04a7fcdd
# ╠═e3438e5b-53f7-460c-af56-bede7958b099
# ╠═16d82230-339b-4faa-a38b-7552443b89b5
# ╠═5ac40637-aa73-4166-a44e-80ba1585af1a
# ╠═853fcf39-9a93-461d-96b9-acdceaa743c6
# ╠═9fef2bd3-0ff7-42d8-a1c0-896fc2b68ebb
# ╠═6267fe50-5d77-4df6-8b8e-a83ec018bd05
# ╠═0ada5640-4534-4407-ac56-9b9c0a2e8650
# ╠═ba914642-2d7d-4483-ac4d-3bce4306ffca
# ╠═dac87f2d-82b8-449e-93e6-0a05bb78014b
# ╠═ce06a55e-c1cd-412e-8d5d-62cbfb8b9db7
# ╠═e1001f96-d515-4a0d-bd91-399551d2122b
# ╠═b9ed0e5d-5ea0-49e4-8a5f-27c8d7752325
# ╠═70f075ec-2ccd-435b-b2b0-b478bff2d223
# ╠═50488d99-e019-48a8-a340-d01a51c55ef6
# ╠═e4338dc0-cb33-4f38-9181-475bdb3c523b
# ╠═acac9b27-aca5-410a-80f9-7a10eac7612b
