### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 3395c983-7d14-4163-bebb-096c851cbde8
begin
	using Revise
	using Pkg
	Pkg.activate("../")
	
	using BayesianModels
	using DataFrames
	using DelimitedFiles
	using Downloads
	using LinearAlgebra
	using Plots
	using Zygote

	include("./plotting.jl")
end

# ╔═╡ e5883656-7aa1-11ec-2648-4f9569fddcf8
md"""
# Normal distribution

Learn a multivariate normal distribution 
"""

# ╔═╡ 9b39f491-8836-4368-877b-6cf1eea0ff19
md"""
## Data
"""

# ╔═╡ 60f71c97-f4c5-4b4d-9665-4838d676793b
datapath = Downloads.download("https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/speech/database/pb/pb.tgz", "data.tar.gz")	

# ╔═╡ 52b2c2a9-01c2-4d39-b9d6-4bcd7a11dec1
run(`tar xf $datapath`)

# ╔═╡ 04ed61ae-a1ed-4114-a160-100e86f0769a
Text(read("PetersonBarney/HEADER", String))

# ╔═╡ ff11bde6-a2c8-47f5-a3bf-a4b6f9758747
# We remove the "*" marker.
open("PetersonBarney/verified_pb.data.filt", "w") do f
	write(f, replace(read("PetersonBarney/verified_pb.data", String), '*' => ""))
end

# ╔═╡ 4b071c89-67b6-4964-a4b1-c784f6d99f9c
begin
	rawX = readdlm("PetersonBarney/verified_pb.data.filt", '\t')
	
	# Normalize each dimension to have 0 mean and 1 standard deviation. 
	local X = rawX[:, 5:end]
	local μ̄ = sum(X, dims=1) / size(X,1)
	local σ̄ = sqrt.( sum((X .- μ̄).^2, dims=1) / size(X,1) ) 
	rawX[:, 5:end] = (X .- μ̄) ./ σ̄

	data = DataFrame(
		rawX,
		[:type, :speaker, :phone_id, :phone_ascii, :f0, :f1, :f2, :f3]
	)
end

# ╔═╡ 570dc7f7-10dc-46ed-9958-2acc0ed76ad1
phones = Set(data[:, :phone_ascii])

# ╔═╡ c74b04e1-b19d-4398-b376-dda5fa88ca86
begin
	local p = plot()
	for phone in phones
		X = subset(data, :phone_ascii => p -> p .== phone)[:, [:f1, :f2]]
		scatter!(
			X[:, 1], X[:, 2], 
			label=phone
		)
	end
	p
end

# ╔═╡ 15c82689-be5d-411b-9594-1ba031fb2a0d
Xs = [Array{Float64}(subset(data, :phone_ascii => p -> p .== phone)[:, [:f0, :f1, :f2, :f3]])'
	  for phone in phones]

# ╔═╡ 6c8c5ecc-9dc2-446a-8726-699694677c42
md"""
## Basic Model

We start by learning a basic model: a set of independant Normal distribution each with a Normal-Wishart prior. This basic model will then be used to initialize the GSM.
"""

# ╔═╡ 2d01cdec-0da2-48f7-9ae9-7cf2298e9d11
# Dimension of the observed space.
D = size(Xs[1], 1)

# ╔═╡ 3814457c-fb1b-4f49-9717-c9d9fcdcc070
prior = NormalWishart(zeros(4), 1., Matrix{Float64}(I, 4, 4), 4.)

# ╔═╡ 5fa65b07-fe1c-4119-93fa-85738d74af6c
models = [NormalModel(prior) for phone in phones]

# ╔═╡ c43a677b-58dd-4266-a93a-c914b7cb3ebc
begin
	qθs = [prior for phone in phones]
	local N = sum(size.(Xs, 2))
	L = []
	
	for t in 1:1000
		μs = getstats.(models, qθs)
		(l, (∇μs,)) = withgradient(μs -> sum(elbo.(models, Xs, qθs, μs))/N, μs)
		qθs = BayesianModels.newposterior.(models, qθs, ∇μs; lrate=1)
	
		push!(L, l)
	end
	L
end

# ╔═╡ e8148b69-8fba-4d15-8c41-8e445d2cb5a0
plot(L, label=false)

# ╔═╡ 67ee5a37-0baa-4986-8558-77b720321771
begin 
	local p = plot(legend=false)
	for (phone, X, qθ) in zip(phones, Xs, qθs)
		μ = qθ.μ[2:3]
		Σ = inv(qθ.ν*qθ.W)[2:3,2:3]
		plotnormal!(μ, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
		scatter!(X[2,:], X[3,:], label=phone)
	end
	p
end

# ╔═╡ d12fcd36-51b4-463e-b01c-807239522d6b
md"""
## GSM

Now, we build the GSM. We start by defining the parameter transformation from the "real space" ($\boldsymbol{\psi}$) to the "standard space" ($\boldsymbol{\Lambda}$, $\boldsymbol{\mu}$). The transformation is defined by:

```math
\begin{align}
	\boldsymbol{\psi} &= \begin{bmatrix}
		\boldsymbol{\psi}^{(\boldsymbol{\Lambda\mu})} \\
		\boldsymbol{\psi}^{(\mathbf{L})}_1 \\
		\boldsymbol{\psi}^{(\mathbf{L})}_2
	\end{bmatrix} \\
	\boldsymbol{\Lambda\mu} &= \boldsymbol{\psi}^{(\boldsymbol{\Lambda\mu})} \\
	\text{diag}(\mathbf{L}) &= \exp \{ \boldsymbol{\psi}^{(\mathbf{L})}_1 \} \\
	\text{vech}^{\dagger}(\mathbf{L}) &= \boldsymbol{\psi}^{(\mathbf{L})}_2 \\
\end{align}
```
where $\text{vech}^{\dagger}$ is the half-vectorization of a matrix **not including the diagonal**, 
"""

# ╔═╡ 44f0dfd2-cde0-4cc5-b8e7-03d36932e335
function f(ψ::AbstractVector)
	L = length(ψ)
	D = Int((-(3/2) + sqrt((9/4) + 2L)) )
	
	Λμ = ψ[1:D]
	A = diagm(exp.(1e-3 .+ ψ[D+1:2*D]))
	L = A #+ CompressedTriangularMatrix(D, 1, ψ[2*D+1:end])
	L⁻¹ = inv(L)
	
	Λ = L * L'
	μᵀΛμ = ((L⁻¹' * L⁻¹) * Λμ)' * Λμ
	logdetΛ = 2*sum(ψ[D+1:2*D])
	
	Λμ, μᵀΛμ, Λ, logdetΛ
end

# ╔═╡ fb6c11f7-e68c-4edf-965e-d36dd19cad2f
function f⁻¹((Λμ, μᵀΛμ, Λ, logdetΛ))
	C = cholesky(Symmetric(Λ))
	vcat(Λμ, log.(diag(C.L)), vech(C.L, 1))
end

# ╔═╡ 1717eead-5f0a-45cb-bc73-85414e9f06e8
Λμ, μᵀΛμ, Λ, logdetΛ = f(f⁻¹(unpack(qθs[1], μ(qθs[1]))))

# ╔═╡ d159e0b9-7118-4155-ba20-a8bc73465fc8
Λ

# ╔═╡ 07f8c201-682a-4e82-a210-0b8c8050d7bc
logdet(Λ)

# ╔═╡ 9163576a-5aff-49a2-8c52-f240c0c338e3
M= inv(Λ)

# ╔═╡ 93de5fe9-6fe8-49d2-b9be-462f90148dd9
(M* Λμ)' * Λμ

# ╔═╡ a0cc6a05-6d8a-4b3f-9886-aa6139382a0c
md"""
 $Q$ is dimension of the subspace 
"""

# ╔═╡ c2f69d52-7533-4bea-9457-f8ba4f324ee8
Q = 2

# ╔═╡ 3d4c8ff8-1621-4710-a49e-7ccda892cd7c
md"""
 $P$ is the dimension of model's parameter space 
"""

# ╔═╡ 4a09cd83-a32c-486d-afa0-af460a20d784
P = 2D + (D^2 - D) ÷ 2

# ╔═╡ 0ae825a4-f21e-4ed9-a08c-2d781d2c9576
pW = JointNormal(zeros(Q, P), zeros(Q, P))

# ╔═╡ 595b1c89-f134-4a73-8193-f5f5b6e5d83d
pb = NormalDiag(zeros(P), zeros(P))

# ╔═╡ 0fd9bb0e-a8b0-4b55-9af8-8b3356fc53a5
ph = NormalDiag(zeros(Q), zeros(Q))

# ╔═╡ 37067855-9706-404e-ac02-8e15dc9a84d5
gsm = GSM(f, pW, pb, [NormalModel(ph) for p in collect(phones)[1:3]])

# ╔═╡ ca225bbf-b945-4366-a9cd-2b6436a78b26
md"""
We initialize the mean of variational posterior of the subspace $q(\mathbf{W})$ to the $Q+1$ eigenvectors of the, we extract the 
"""

# ╔═╡ e11390db-5c52-48b8-bc18-1b5991c74816
begin 
	Ψ = hcat(vcat(f⁻¹.(unpack.(qθs, μ.(qθs))))...)
	Ψ̄ = sum(Ψ, dims=2) ./ size(Ψ, 2)
	local Y = (Ψ .- Ψ̄)
	local V = eigen(Symmetric(Y * Y')).vectors[(end-Q+1):end, :]
	#local V = eigen(Symmetric(Y * Y')).vectors[:, :]
	W₀ = Matrix(vcat(V, reshape(Ψ̄, 1,  :)))
	eigen(Symmetric(Y * Y')).values
end

# ╔═╡ 391e8086-206f-4a5c-ab67-05d608f252dc
W₀

# ╔═╡ 236cc2fb-c314-46b8-a096-56a4a599f2a4
Ψ

# ╔═╡ c5dfed78-b5ea-4430-aa2f-680c2c442d22
f(Ψ[:, 1])

# ╔═╡ 899b1b8e-1a96-4d36-89ba-77cd2f9c612d
begin 
	local p = plot(legend=false)
	local k = 2
	for (phone, X, ψ) in zip(phones, Xs, eachcol(Ψ))
		ψ = W₀[k:end-1,:]' * (W₀[k:end-1, :] * ψ) + W₀[end,:]
		Λμ, μᵀΛμ, Λ, logdetΛ = f(ψ)
		Σ = inv(Λ)
		μ = Σ*Λμ
		μ = μ[2:3]
		Σ = Σ[2:3,2:3]
		plotnormal!(μ, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ d8fe12ad-ab08-404f-8c98-92271e8d2e25
begin
	local k = length(gsm.models)
	local Ys = Xs[1:k]
	
	local qH = [NormalDiag(zeros(Q), zeros(Q)) for p in 1:k]
	local qW = JointNormal(zeros(Q, P), zeros(Q, P))
	local qb = NormalDiag(zeros(P), zeros(P))
	local μ = getstats(gsm, (qW, qb, qH))
	elbo(gsm, Ys, (qW, qb, qH), μ) / sum(size.(Xs, 2))
end

# ╔═╡ 4274cf0e-0cc1-41f4-a0bf-ad46785fea7d
begin
	local k = length(gsm.models)
	#local Ys = Xs[1:k]
	Ys = [
		0.5*randn(4, 500) .+ [0, 5, 0, 0], 
		0.5*randn(4, 500) .+ [0, 0, 5, 0], 
		0.5*randn(4, 500) .+ [0, -5, 0, 0]
	]
	
	qH = [NormalDiag(zeros(Q), zeros(Q) .- 1) for p in 1:k]
	#qH = [NormalIso(zeros(Q), Float64(0)) for p in 1:k]
	qW = JointNormal(zeros(Q, P), zeros(Q, P) .- 1)
	qb = NormalDiag(zeros(P), zeros(P) .- 1)
	
	β = 10
	l = 1
	local qθ = (qW, qb, qH)
	N = sum(size.(Ys, 2))
	Lg = []
	
	for t in 1:10000
		μ = getstats(gsm, qθ)
		(l, (∇μ,)) = withgradient(μ -> elbo(gsm, Ys, qθ, μ)/N, μ)

		# lr = 0.1
		# if t < 50000
		# 	∇μ = (
		# 		∇μ[1] ./ norm(∇μ[1]),
		# 		∇μ[2] ./ norm(∇μ[2]),
		# 		∇μ[3] ./ norm(∇μ[3])
		# 	)
		# 	lr = 10
		# end
		for (i, g) in enumerate(∇μ)
			println("norm $i : $(norm(g))")
		end
		∇μ = (norm(g) > 100 ? g ./ norm(g) : g for g in ∇μ)

		qθ = BayesianModels.newposterior(gsm, qθ, ∇μ; lrate=1)
	
		push!(Lg, l)
	end
	qW, qb, qH = qθ
	Lg
end

# ╔═╡ 732afd32-13b0-4d7e-8d4a-ca60e14e0be0
qW

# ╔═╡ 937cf6f8-8fe6-4ea0-85d2-af94f6214784
μ(qW)

# ╔═╡ 47d83857-1c47-4aa7-b04a-22aecd821849
x̄, x̄² = unpack(qW, μ(qW))

# ╔═╡ a51f99b6-6219-4bb4-b2d0-1cd7b0e103eb
log.(x̄² .- (x̄ .^ 2))

# ╔═╡ 73bd4d43-4912-40f4-a25b-b9aed4876607


# ╔═╡ df3c96b6-02b5-4f51-877a-564062c749ba
μ.(qH)

# ╔═╡ b46d8867-ce48-47d6-9be0-df3ed9e77110
qH

# ╔═╡ 6944ed01-1932-407d-86b0-b4269af6d25c
qb

# ╔═╡ 35984926-37a0-4de4-a0b7-4eb93ed6bade
begin 
	local p = plot(legend=false)
	local k = 1
	local d1, d2 = 2, 3
	for (phone, X, qh) in zip(phones, Ys, qH)
		ψ = (qW.M + exp.(-.5*qW.lnΛ) .* randn(size(qW.M)...))' * (qh.μ .+ exp.(-0.5 * qh.lnλ) .* randn(2))
		ψ = ψ + qb.μ + exp.( -.5 * qb.lnλ) .* randn(size(qb.μ))
		Λμ, μᵀΛμ, Λ, logdetΛ = f(ψ)
		Σ = inv(Λ)
		μ = Σ*Λμ
		μ = μ[d1:d2]
		Σ = Σ[d1:d2,d1:d2]
		plotnormal!(μ, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
		scatter!(X[d1,:], X[d2,:])
	end
	
	p
end

# ╔═╡ 926070be-3e40-456f-95b5-53399d7c2644
Xs[1]

# ╔═╡ 70dad4a3-e470-4c57-93c6-840eb29e5317
begin 
	local p = plot(legend=false)
	local k = 1
	local d1, d2 = 1,2
	for (phone, X, qh) in zip(phones, Xs, qH)
		Σ = inv(diagm(ones(length(qh.μ)) .* exp.(qh.lnλ)))[d1:d2,d1:d2]
		μ = qh.μ[d1:d2]
		plotnormal!(μ, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ ebba18b7-1a20-45aa-b843-2baa1b60f468
plot(Lg[8000:10:end], label=false)

# ╔═╡ e62cd0a8-56d3-468a-8a92-5ac3f227547e
qW.M

# ╔═╡ df2f789e-27a7-4b32-86e0-072e0b08ea0b
(1 ./ exp.(qW.lnΛ))

# ╔═╡ d4709c19-3581-4d8b-9497-45ba43cdf938
qh = qH[1]

# ╔═╡ 0dfd2c1f-6cd5-4397-bddb-dd7e15838fd0
μx = unpack(qh, μ(qh))

# ╔═╡ d1251073-2ca7-4b54-901b-96e1933fe46c
x, x² = μx

# ╔═╡ dd2dd84c-1884-426b-96dc-2e30bdd6cf92
mm = x .^ 2

# ╔═╡ 8f0e8dfa-5be2-45ce-9dd1-a02bf989a883
σ² = x² .- mm

# ╔═╡ de63b69a-004b-4052-baf8-2fbf9d9b33fe
Z = x .+ sqrt.(x² .- mm) .* randn(eltype(x), 4, 100)

# ╔═╡ 3eb2e6ee-85d4-44fa-af7b-3eeda3f94bcd
begin
	m = qh.μ[1:2]
	Σ = diagm(exp.(-qh.lnλ))[1:2,1:2]
	p = plot(legend=false)
	plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	scatter!(Z[1, :], Z[2,:])
	p
end

# ╔═╡ 65cc389d-a5e6-4f12-b6e7-6c32fd192018
qh.μ

# ╔═╡ Cell order:
# ╟─e5883656-7aa1-11ec-2648-4f9569fddcf8
# ╠═3395c983-7d14-4163-bebb-096c851cbde8
# ╠═9b39f491-8836-4368-877b-6cf1eea0ff19
# ╠═60f71c97-f4c5-4b4d-9665-4838d676793b
# ╠═52b2c2a9-01c2-4d39-b9d6-4bcd7a11dec1
# ╠═04ed61ae-a1ed-4114-a160-100e86f0769a
# ╠═ff11bde6-a2c8-47f5-a3bf-a4b6f9758747
# ╠═4b071c89-67b6-4964-a4b1-c784f6d99f9c
# ╠═570dc7f7-10dc-46ed-9958-2acc0ed76ad1
# ╠═c74b04e1-b19d-4398-b376-dda5fa88ca86
# ╠═15c82689-be5d-411b-9594-1ba031fb2a0d
# ╟─6c8c5ecc-9dc2-446a-8726-699694677c42
# ╠═2d01cdec-0da2-48f7-9ae9-7cf2298e9d11
# ╠═3814457c-fb1b-4f49-9717-c9d9fcdcc070
# ╠═5fa65b07-fe1c-4119-93fa-85738d74af6c
# ╠═c43a677b-58dd-4266-a93a-c914b7cb3ebc
# ╠═e8148b69-8fba-4d15-8c41-8e445d2cb5a0
# ╠═67ee5a37-0baa-4986-8558-77b720321771
# ╟─d12fcd36-51b4-463e-b01c-807239522d6b
# ╠═44f0dfd2-cde0-4cc5-b8e7-03d36932e335
# ╠═fb6c11f7-e68c-4edf-965e-d36dd19cad2f
# ╠═1717eead-5f0a-45cb-bc73-85414e9f06e8
# ╠═d159e0b9-7118-4155-ba20-a8bc73465fc8
# ╠═07f8c201-682a-4e82-a210-0b8c8050d7bc
# ╠═9163576a-5aff-49a2-8c52-f240c0c338e3
# ╠═93de5fe9-6fe8-49d2-b9be-462f90148dd9
# ╟─a0cc6a05-6d8a-4b3f-9886-aa6139382a0c
# ╠═c2f69d52-7533-4bea-9457-f8ba4f324ee8
# ╟─3d4c8ff8-1621-4710-a49e-7ccda892cd7c
# ╠═4a09cd83-a32c-486d-afa0-af460a20d784
# ╠═0ae825a4-f21e-4ed9-a08c-2d781d2c9576
# ╠═595b1c89-f134-4a73-8193-f5f5b6e5d83d
# ╠═0fd9bb0e-a8b0-4b55-9af8-8b3356fc53a5
# ╠═37067855-9706-404e-ac02-8e15dc9a84d5
# ╟─ca225bbf-b945-4366-a9cd-2b6436a78b26
# ╠═e11390db-5c52-48b8-bc18-1b5991c74816
# ╠═391e8086-206f-4a5c-ab67-05d608f252dc
# ╠═236cc2fb-c314-46b8-a096-56a4a599f2a4
# ╠═c5dfed78-b5ea-4430-aa2f-680c2c442d22
# ╠═899b1b8e-1a96-4d36-89ba-77cd2f9c612d
# ╠═d8fe12ad-ab08-404f-8c98-92271e8d2e25
# ╠═4274cf0e-0cc1-41f4-a0bf-ad46785fea7d
# ╠═732afd32-13b0-4d7e-8d4a-ca60e14e0be0
# ╠═937cf6f8-8fe6-4ea0-85d2-af94f6214784
# ╠═47d83857-1c47-4aa7-b04a-22aecd821849
# ╠═a51f99b6-6219-4bb4-b2d0-1cd7b0e103eb
# ╠═73bd4d43-4912-40f4-a25b-b9aed4876607
# ╠═df3c96b6-02b5-4f51-877a-564062c749ba
# ╠═b46d8867-ce48-47d6-9be0-df3ed9e77110
# ╠═6944ed01-1932-407d-86b0-b4269af6d25c
# ╠═35984926-37a0-4de4-a0b7-4eb93ed6bade
# ╠═926070be-3e40-456f-95b5-53399d7c2644
# ╠═70dad4a3-e470-4c57-93c6-840eb29e5317
# ╠═ebba18b7-1a20-45aa-b843-2baa1b60f468
# ╠═e62cd0a8-56d3-468a-8a92-5ac3f227547e
# ╠═df2f789e-27a7-4b32-86e0-072e0b08ea0b
# ╠═d4709c19-3581-4d8b-9497-45ba43cdf938
# ╠═0dfd2c1f-6cd5-4397-bddb-dd7e15838fd0
# ╠═d1251073-2ca7-4b54-901b-96e1933fe46c
# ╠═dd2dd84c-1884-426b-96dc-2e30bdd6cf92
# ╠═8f0e8dfa-5be2-45ce-9dd1-a02bf989a883
# ╠═de63b69a-004b-4052-baf8-2fbf9d9b33fe
# ╠═3eb2e6ee-85d4-44fa-af7b-3eeda3f94bcd
# ╠═65cc389d-a5e6-4f12-b6e7-6c32fd192018
