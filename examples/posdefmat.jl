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
	using Zygote
end

# ╔═╡ 925ee465-d77c-49c6-aae3-6ac70c83418a
3 ÷ 2

# ╔═╡ 6f7c5706-d916-4536-8bac-8652e3130e13
A = [2 0.75 -0.2; 0.75 3 1.2; -0.2 1.2 2.4]

# ╔═╡ fc1e397a-94f1-44b6-a805-e2ec7e13f027
B = BayesianModels.CompressedTriangularMatrix(A)

# ╔═╡ ec788974-263b-48d1-9494-bcc4fd53b1d4
L = BayesianModels.CompressedTriangularMatrix(3, 1, BayesianModels.vech(A, 1))

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
C = BayesianModels.CompressedSymmetricMatrix(A)

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
# ╠═925ee465-d77c-49c6-aae3-6ac70c83418a
# ╠═6f7c5706-d916-4536-8bac-8652e3130e13
# ╠═fc1e397a-94f1-44b6-a805-e2ec7e13f027
# ╠═ec788974-263b-48d1-9494-bcc4fd53b1d4
# ╠═dac87f2d-82b8-449e-93e6-0a05bb78014b
# ╠═ce06a55e-c1cd-412e-8d5d-62cbfb8b9db7
# ╠═e1001f96-d515-4a0d-bd91-399551d2122b
# ╠═b9ed0e5d-5ea0-49e4-8a5f-27c8d7752325
# ╠═70f075ec-2ccd-435b-b2b0-b478bff2d223
# ╠═50488d99-e019-48a8-a340-d01a51c55ef6
# ╠═e4338dc0-cb33-4f38-9181-475bdb3c523b
# ╠═acac9b27-aca5-410a-80f9-7a10eac7612b
