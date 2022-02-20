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
	using ForwardDiff
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

# ╔═╡ e2aa337d-4cdc-43d2-a8f6-5d0a607a3de0
arpa2ipa = Dict(
	"IY" => "i",
	"IH" => "I",
	"EH" => "ε", 
	"AE" => "æ",
	"AH" => "Λ",
	"AA" => "ɑ",
	"AO" => "ɔ",
	"UH" => "ʊ",
	"UW" => "u",
	"ER" => "ɝ"
)

# ╔═╡ 570dc7f7-10dc-46ed-9958-2acc0ed76ad1
phones = Set(data[:, :phone_ascii])

# ╔═╡ c74b04e1-b19d-4398-b376-dda5fa88ca86
begin
	local p = plot()
	for phone in phones
		X = subset(data,  :phone_ascii => p -> p .== phone)
		X = subset(X, :type => t -> t .== 1)
		X = X[:, [:f1, :f2]]
		scatter!(X[:, 1], X[:, 2], label=phone)
	end
	p
end

# ╔═╡ 15c82689-be5d-411b-9594-1ba031fb2a0d
Xs = [
	Array{Float64}(
		subset(
			data, 
			:type => t -> t .== 1, 
			:phone_ascii => p -> p .== phone
		)[:, [:f0, :f1, :f2, :f3]])'
	for phone in phones
]

# ╔═╡ 6c8c5ecc-9dc2-446a-8726-699694677c42
md"""
## Basic Model

We start by learning a basic model: a set of independent Normal distributions each with a Normal-Wishart prior. 
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
	
	for t in 1:2000
		μs = getstats.(models, qθs)
		(l, (∇μs,)) = withgradient(μs -> sum(elbo.(models, Xs, qθs, μs))/N, μs)
		qθs = newposterior.(models, qθs, ∇μs; lrate=0.1)
	
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
	L = A + CompressedLowerTriangular(D, 1, ψ[2*D+1:end])
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

# ╔═╡ 93109c2d-a784-408a-9466-c1d43dc8cd48
function f⁻¹(η::AbstractVector)
	L = length(η)
	D = Int((-(3/2) + sqrt((9/4) + 2L)))

	Λμ = η[1:D]
	diagΛ = η[D+1:2*D]
	vechΛ = η[2*D+1:end]
	Λ = diagm(diagΛ) + CompressedSymmetric(D, 1, vechΛ)
	
	C = cholesky(Λ)
	vcat(Λμ, log.(diag(C.L)), vech(C.L, 1))
end

# ╔═╡ 3d4c8ff8-1621-4710-a49e-7ccda892cd7c
md"""
 $P$ is the dimension of model's parameter space 
"""

# ╔═╡ 4a09cd83-a32c-486d-afa0-af460a20d784
P = 2D + (D^2 - D) ÷ 2

# ╔═╡ a0cc6a05-6d8a-4b3f-9886-aa6139382a0c
md"""
 $Q$ is dimension of the subspace 
"""

# ╔═╡ c2f69d52-7533-4bea-9457-f8ba4f324ee8
Q = 2

# ╔═╡ 0ae825a4-f21e-4ed9-a08c-2d781d2c9576
pW = JointNormalFixedCov(zeros(Q+1, P))

# ╔═╡ 0fd9bb0e-a8b0-4b55-9af8-8b3356fc53a5
ph = Normal(Q)

# ╔═╡ 37067855-9706-404e-ac02-8e15dc9a84d5
gsm = GSM(f, pW, [NormalModel(ph) for p in collect(phones)[1:end]])

# ╔═╡ 4274cf0e-0cc1-41f4-a0bf-ad46785fea7d
begin	
	# Initialization of the variational posterior
	local qθ = (
		JointNormalFixedCov(randn(Q+1, P)), 
		[Normal(Q) for model in gsm.models]
	)

	# Total number of observed data points 
	local N = sum(size.(Xs, 2))
	
	Lg = []
	for t in 1:100000
		μ = getstats(gsm, qθ)
		(l, (∇μ,)) = withgradient(μ -> elbo(gsm, Xs, qθ, μ)/N, μ)		
		qθ = newposterior(gsm, qθ, ∇μ; lrate=1e-2)
		
		push!(Lg, l)
	end
	qW, qH = qθ
	Lg
end

# ╔═╡ 35984926-37a0-4de4-a0b7-4eb93ed6bade
begin 
	local p = plot(legend=false, xrange=(-3, 3), yrange=(-3,3))
	local k = 1
	local d1, d2 = 2, 3
	for (phone, X, qh) in zip(phones, Xs, qH)
		h = sample(qh)
		W = qW.M 
		ψ = W' * vcat(h, 1)
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

# ╔═╡ cfc2db2f-ec6f-4e7c-bda5-aee0a94ef15e
qW

# ╔═╡ 926070be-3e40-456f-95b5-53399d7c2644
μ(qH[1])

# ╔═╡ 70dad4a3-e470-4c57-93c6-840eb29e5317
begin 
	local p = plot(legend=false)
	local k = 1
	local d1, d2 = 1, 2
	for (phone, X, qh) in zip(phones, Xs, qH)
		x, diagxxᵀ, vechxxᵀ =  unpack(qh, μ(qh))
		m = x[d1:d2]
		xxᵀ = diagm(diagxxᵀ) + CompressedSymmetric(size(x, 1), 1, vechxxᵀ)
		Σ = (xxᵀ - m*m')[d1:d2,d1:d2]
		annotate!(m[1], m[2], phone)
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ ebba18b7-1a20-45aa-b843-2baa1b60f468
plot(Lg[10000:10:end], label=false)

# ╔═╡ c6d3f311-7593-40cd-b01c-7b7f6b067a72
function gsm_η(gsm::GSM, qW, h)
	Λμ, μᵀΛμ, Λ, logdetΛ = f(qW.M' * vcat(h, 1))
	vcat(Λμ, diag(Λ), vech(Λ, 1))
end

# ╔═╡ c0a18a14-475f-4d99-8594-f8054e404f14
gsm_η(gsm, qW, zeros(2))

# ╔═╡ 3cf213b2-b9e6-4765-a444-76142e700d93
function gsm_μ(gsm::GSM, η)
	L = length(η)
	D = Int((-(3/2) + sqrt((9/4) + 2L)))

	Λμ = η[1:D]
	diagΛ = η[D+1:2*D]
	vechΛ = η[2*D+1:end]
	Λ = diagm(diagΛ) + CompressedSymmetric(D, 1, vechΛ)

	Σ = inv(Symmetric(Λ))
	μ = Σ * Λμ

	vcat(μ, vec(Σ + μ * μ'))
end

# ╔═╡ a05fb9e3-5ce9-4ccc-9418-178d3d4aa18f
function gsm_ξ(gsm::GSM, qW, η)
	W = qW.M
	W⁺ = inv(W*W') * W
	h = W⁺ * f⁻¹(η)
	h[1:end-1]
end

# ╔═╡ 45d5823c-2044-4815-a6e4-78e02c568f20
function gsm_A(gsm::GSM, η)
	L = length(η)
	D = Int((-(3/2) + sqrt((9/4) + 2L)))

	Λμ = η[1:D]
	diagΛ = η[D+1:2*D]
	vechΛ = η[2*D+1:end]
	Λ = diagm(diagΛ) + CompressedSymmetric(D, 1, vechΛ)
	Σ = inv(Λ)
	
	-(1/2) * logdet(Λ) + (1/2) * Λμ' * (Σ * Λμ) 
end

# ╔═╡ 534dc966-f719-4049-aabc-bed102de1901
gsm_A(gsm, gsm_η(gsm, qW, zeros(2)))

# ╔═╡ b7c4d472-a001-41d8-b2ae-97ee9b095c47
ForwardDiff.hessian(η -> gsm_A(gsm, η), gsm_η(gsm, qW, zeros(2)))

# ╔═╡ 6eaac0ae-2153-4dec-9abc-8114f95b8a67
ForwardDiff.jacobian(h -> gsm_ξ(gsm, qW, h), gsm_η(gsm, qW, zeros(2)))

# ╔═╡ 9952e00f-0b61-4c12-a8f6-4c0384fdd34b
gsm_μ(gsm, gsm_η(gsm, qW, zeros(2)))

# ╔═╡ cee9c7c9-c356-482e-b90e-4c9ec869eb0a
gsm_ξ(gsm, qW, gsm_η(gsm, qW, zeros(2)))

# ╔═╡ 8488be6d-f08e-4d26-a41f-fa3350627d12
function trajectory(gsm, qW, A, B)
	traj = [A]
	while norm(traj[end] - B) > 1e-1		
		η = gsm_η(gsm, qW, traj[end])
		H = ForwardDiff.hessian(η -> gsm_A(gsm, η), η)
		J = ForwardDiff.jacobian(h -> gsm_ξ(gsm, qW, h), η)
		(∇h,) = Zygote.gradient(h -> -(h - B)' * (h - B), traj[end])	
		
		#∇h = ∇h ./ norm(∇h) 
		G = J * H * J'
		v = inv(G) * ∇h
		v = v ./ norm(v)
		 
		push!(traj, traj[end] + 1e-3 * v)
	end
	hcat(traj...)
end

# ╔═╡ 839249a8-dae0-4b29-8717-278b5f495176
begin
	p = plot(xlims=(-3, 3), ylims=(-3, 3), legend=false)
	s, e = -4, 4
	for i = s:0.5:e
		println("i = $(i)")
		t = trajectory(gsm, qW, Float64[s, i], Float64[e, i])
		plot!(t[1, :], t[2, :], linecolor=:black)
	end

	for i = s:0.5:e
		t = trajectory(gsm, qW, [i, s], [i, e])
		plot!(t[1, :], t[2, :], linecolor=:black)
	end
	
	p
end

# ╔═╡ 6b77d67e-1690-4f60-812e-9687030e3aa2
begin 
	local k = 1
	local d1, d2 = 1, 2
	for (phone, X, qh) in zip(phones, Xs, qH)
		x, diagxxᵀ, vechxxᵀ =  unpack(qh, μ(qh))
		m = x[d1:d2]
		xxᵀ = diagm(diagxxᵀ) + CompressedSymmetric(size(x, 1), 1, vechxxᵀ)
		Σ = (xxᵀ - m*m')[d1:d2,d1:d2]
		
		annotate!(m[1], m[2], arpa2ipa[phone])
		plotnormal!(m, Σ, σ=2, colorfillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ Cell order:
# ╟─e5883656-7aa1-11ec-2648-4f9569fddcf8
# ╠═3395c983-7d14-4163-bebb-096c851cbde8
# ╠═9b39f491-8836-4368-877b-6cf1eea0ff19
# ╠═60f71c97-f4c5-4b4d-9665-4838d676793b
# ╠═52b2c2a9-01c2-4d39-b9d6-4bcd7a11dec1
# ╠═04ed61ae-a1ed-4114-a160-100e86f0769a
# ╠═ff11bde6-a2c8-47f5-a3bf-a4b6f9758747
# ╠═4b071c89-67b6-4964-a4b1-c784f6d99f9c
# ╠═e2aa337d-4cdc-43d2-a8f6-5d0a607a3de0
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
# ╠═93109c2d-a784-408a-9466-c1d43dc8cd48
# ╟─3d4c8ff8-1621-4710-a49e-7ccda892cd7c
# ╠═4a09cd83-a32c-486d-afa0-af460a20d784
# ╟─a0cc6a05-6d8a-4b3f-9886-aa6139382a0c
# ╠═c2f69d52-7533-4bea-9457-f8ba4f324ee8
# ╠═0ae825a4-f21e-4ed9-a08c-2d781d2c9576
# ╠═0fd9bb0e-a8b0-4b55-9af8-8b3356fc53a5
# ╠═37067855-9706-404e-ac02-8e15dc9a84d5
# ╠═4274cf0e-0cc1-41f4-a0bf-ad46785fea7d
# ╠═35984926-37a0-4de4-a0b7-4eb93ed6bade
# ╠═cfc2db2f-ec6f-4e7c-bda5-aee0a94ef15e
# ╠═926070be-3e40-456f-95b5-53399d7c2644
# ╠═70dad4a3-e470-4c57-93c6-840eb29e5317
# ╠═ebba18b7-1a20-45aa-b843-2baa1b60f468
# ╠═c6d3f311-7593-40cd-b01c-7b7f6b067a72
# ╠═c0a18a14-475f-4d99-8594-f8054e404f14
# ╠═3cf213b2-b9e6-4765-a444-76142e700d93
# ╠═a05fb9e3-5ce9-4ccc-9418-178d3d4aa18f
# ╠═45d5823c-2044-4815-a6e4-78e02c568f20
# ╠═534dc966-f719-4049-aabc-bed102de1901
# ╠═b7c4d472-a001-41d8-b2ae-97ee9b095c47
# ╠═6eaac0ae-2153-4dec-9abc-8114f95b8a67
# ╠═9952e00f-0b61-4c12-a8f6-4c0384fdd34b
# ╠═cee9c7c9-c356-482e-b90e-4c9ec869eb0a
# ╠═8488be6d-f08e-4d26-a41f-fa3350627d12
# ╠═839249a8-dae0-4b29-8717-278b5f495176
# ╠═6b77d67e-1690-4f60-812e-9687030e3aa2
