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
## Model
"""

# ╔═╡ 3814457c-fb1b-4f49-9717-c9d9fcdcc070
prior = NormalWishart(zeros(4), 1, Matrix{Float64}(I, 4, 4), 4)

# ╔═╡ 606456c9-8bd3-41b8-9f38-c9d10b5b7421
model = NormalModel()

# ╔═╡ 5fa65b07-fe1c-4119-93fa-85738d74af6c
models = Tuple(NormalModel() for phone in phones)

# ╔═╡ 5e8550e0-c661-4e37-b974-400e5245b98b
function elbo(model::NormalModel, X, posterior::NormalWishart,
			  Tθ=getstats(model, posterior)) 
	llh = sum(loglikelihood(model, X, posterior, Tθ))
	kl = kldiv(posterior, prior, Tθ)
	N = size(X, 2)
	(1/N)*(llh - kl)
end

# ╔═╡ 6113d3dd-67dc-4125-b241-57fac583f70c
function elbo(model, Xs, posteriors::NTuple{N,<:NormalWishart}, 
			  Tθs=getstats(model, posteriors)) where N 
	Tθs = reshape(Tθs, :, 10)
	L = sum(elbo.(Ref(model), Xs, posteriors, eachcol(Tθs)))
	L / length(Xs)
end

# ╔═╡ fc0d57f5-2572-41c8-87cc-8a3b2823cfc5
elbo(model, Xs, Tuple(prior for phone in phones))

# ╔═╡ c43a677b-58dd-4266-a93a-c914b7cb3ebc
begin
	local C = length(phones)
	qθs = Tuple(prior for phone in phones)
	L = []
	for t in 1:1000
		μ = getstats(model, qθs)
		(l, (∇μ,)) = withgradient(μ -> elbo(model, Xs, qθs, μ), μ)
		qθs = Tuple(
			BayesianModels.newposterior.((model,), qθs, 
										 eachcol(reshape(∇μ, :, C)), lrate=1)
		)
		push!(L, l)
		#push!(L, elbo(model, Xs, qθs))
	end
	L
end

# ╔═╡ e8148b69-8fba-4d15-8c41-8e445d2cb5a0
plot(L, label=false)

# ╔═╡ 49555a8f-69c4-4e37-bd03-6e09197b27cf


# ╔═╡ 67ee5a37-0baa-4986-8558-77b720321771
begin 
	local p = plot(legend=false)
	for (phone, X, qθ) in zip(phones, Xs, qθs)
		μ = qθ.μ[2:3]
		Σ = inv(qθ.ν*qθ.W)[2:3,2:3]
		#scatter!(X[2,:], X[3,:],  label=phone, alpha=0.4)
		plotnormal!(μ, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ e81105ac-e628-46cf-bf1c-8533711f8558
Xs[1]

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
# ╠═3814457c-fb1b-4f49-9717-c9d9fcdcc070
# ╠═606456c9-8bd3-41b8-9f38-c9d10b5b7421
# ╠═5fa65b07-fe1c-4119-93fa-85738d74af6c
# ╠═5e8550e0-c661-4e37-b974-400e5245b98b
# ╠═6113d3dd-67dc-4125-b241-57fac583f70c
# ╠═fc0d57f5-2572-41c8-87cc-8a3b2823cfc5
# ╠═c43a677b-58dd-4266-a93a-c914b7cb3ebc
# ╠═e8148b69-8fba-4d15-8c41-8e445d2cb5a0
# ╠═49555a8f-69c4-4e37-bd03-6e09197b27cf
# ╠═67ee5a37-0baa-4986-8558-77b720321771
# ╠═e81105ac-e628-46cf-bf1c-8533711f8558
