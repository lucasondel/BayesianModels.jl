### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 3cd07b90-9e21-4a74-8d99-a6375beb7d1e
begin
	using Revise
	using Pkg
	Pkg.activate("../")
	
	using BayesianModels
	using DelimitedFiles
	using LinearAlgebra
	using Plots
	using Zygote

	include("./plotting.jl")
end

# ╔═╡ 6958bde2-c50f-4afd-8d08-7dfbfd97169a
md"""
# Learning and Modeling the Languages Phonetics  

*[Lucas Ondel](https://lucasondel.github.io/), 8 March 2022*
"""

# ╔═╡ b83311ab-ca4f-432e-8627-5207517a81db
md"""## Lingua Libre Formants data

We use the [Lingua Libre Formants](https://github.com/lucasondel/LinguaLibreFormants) data for our analysis.
"""

# ╔═╡ c01de0cd-43f8-49cc-a8c9-ffbbc182d9e9
begin
	# Database URL
	local dataurl = "https://raw.githubusercontent.com/lucasondel/LinguaLibreFormants/main/data/formants.tsv"
	# File name stored on disk
	local localfile = "formants.tsv"

	# Download the data if it is not done already
	if ! isfile(localfile)
		run(`wget $dataurl`)
	end

	# Read the raw data
	local rawdata = readdlm(localfile)

	# Prepare the data
	speakers = Set()
	languages = Set()
	phones = Set()
	labels = []
	data = []
	for (lang, spk, phone, f1, f2, f3, duraiton) in eachrow(rawdata)
		lang, spk, phone = Symbol(lang), Symbol(spk), Symbol(phone)
		push!(data, vcat(f1, f2, f3))
		push!(labels, (lang, spk, phone))
		push!(speakers, spk)
		push!(languages, lang)
		push!(phones, phone)
	end
	data = hcat(data...)
	labels = vcat(labels...)

	# Transform the languages, speakers and phone sets to ordered collections
	languages = collect(languages)
	speakers = collect(speakers)
	phones = collect(phones)

	# Build the element <-> id mappings
	lang2id = Dict(lang => i for (i, lang) in enumerate(languages))
	id2lang = Dict(i => lang for (i, lang) in enumerate(languages))
	spk2id = Dict(spk => i for (i, spk) in enumerate(speakers))
	id2spk = Dict(i => spk for (i, spk) in enumerate(speakers))
	phone2id = Dict(phone => i for (i, phone) in enumerate(phones))
	id2phone = Dict(i => phone for (i, phone) in enumerate(phones))

	# Transform the labels to use the id rather than symbols
	labels = [(lang2id[lang], spk2id[spk], phone2id[phone])
			  for (lang, spk, phone) in labels]
	
	# Group the data for each triplet (lang, speaker, phone)
	groups = Dict()
	for (label, sample) in zip(labels, eachcol(data))
		list = get(groups, label, [])
		push!(list, sample)
		groups[label] = list
	end
	groups = tuple((k => hcat(groups[k]...) for k in keys(groups))...)


	# Some useful constants
	D = size(data, 1)
	N = size(data, 2)
	L = length(languages)
	S = length(speakers)
	P = length(phones)

	md"""
	| \# languages | \# speakers  | \# phones |
	|:------------:|:------------:|:---------:|
	| $L           | $S           | $P        |

	We note $D$ the dimension of the observed space. In our case, $D$ is $D.
	"""
end

# ╔═╡ e9a8feec-eb20-4c55-8669-6c6c085b4d85
labels

# ╔═╡ 4fd12bad-b244-428e-8faa-f6000d0d9e3a
collect(languages)

# ╔═╡ dc2d02f7-2044-483c-99b1-b864c6940223
data

# ╔═╡ 29b92c17-36dd-49c5-afdf-21c2ed759fba
groups

# ╔═╡ 32804074-8126-4328-9b2b-9785690c76e5
lang2id

# ╔═╡ 8b535f17-7a55-42da-ba7a-c161bec9fd28
md"""
## Model
	
### Embeddings   
1.  $\boldsymbol{\lambda} \sim p(\boldsymbol{\lambda}) \triangleq \mathcal{N}( \mathbf{0}, \mathbf{I})$, $\boldsymbol{\lambda}$ → language embedding of dimension $Q_\lambda$
2.  $\boldsymbol{\sigma} \sim p(\boldsymbol{\sigma}) \triangleq \mathcal{N}( \mathbf{0}, \mathbf{I})$, $\boldsymbol{\sigma}$ → speaker embedding of dimension $Q_\sigma$
3.  $\boldsymbol{\pi} \sim p(\boldsymbol{\pi}) \triangleq \mathcal{N}( \mathbf{0}, \mathbf{I})$ $\boldsymbol{\pi}$ → phone embedding of dimension $Q_\pi$
"""

# ╔═╡ 5175712e-7f27-459e-8e6a-9bed325d6b4e
md"""
### Subspace construction 
1.  $\mathbf{L}_i \sim p(\mathbf{L}) \triangleq \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\{\mathbf{L}_0, \dots, \mathbf{L}_{Q_\lambda} \}$ → Language hyper-subspace bases of dimension $(Q_\pi + 1) \times K$
2.  $\mathbf{S}_j \sim p(\mathbf{S}) \triangleq \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\{\mathbf{S}_0, \dots, \mathbf{S}_{Q_\sigma} \}$ → Speaker hyper-subspace bases $(Q_\pi + 1) \times K$
3.  $\mathbf{P} = \sum_{i=0}^{Q_\lambda} \lambda_i \mathbf{L}_i + \sum_{j=0}^{Q_\sigma} \sigma_j \mathbf{S}_j$ where $\lambda_0 = \sigma_0 = 1$, → Phonetic subspace bases of dimension $(Q_\pi + 1) \times K$
"""

# ╔═╡ 763b5f95-db02-4ede-b632-848a646232d9
md"""
### Phone model 
1.  $\boldsymbol{\psi} = \mathbf{P}^\top \begin{bmatrix} \boldsymbol{\pi} \\ 1 \end{bmatrix}$, $\boldsymbol{\psi} \in \mathbb{R}^K$
2.  $\boldsymbol{\eta} = f(\boldsymbol{\psi})$ 
3.  $\mathbf{x}_n \sim p(\mathbf{x} | \boldsymbol{\eta}) \triangleq \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$, where $\boldsymbol{\eta}$ are the natural parameters of the multivariate Normal distribution
"""

# ╔═╡ 841a3f0e-8975-49d2-959e-afe832ed41e4
md"""
### Parameter function mapping

The function $f$ transforms the parameters $\boldsymbol{\psi}$ from the real space into the natural parameter space. It is defined as follows:

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
	\boldsymbol{\Lambda} &= \mathbf{L} \mathbf{L}^\top
	\end{align}
```
where $\text{vech}^{\dagger}$ is the half-vectorization of a matrix **not including the diagonal**
"""

# ╔═╡ b0761304-49df-4626-a677-488295dc786d
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

# ╔═╡ 4154a267-326d-495c-aa60-156560aa5f72
Qλ = 2

# ╔═╡ 4c69229d-a272-49eb-9dc1-484c53bad338
pλ = Normal(Qλ)

# ╔═╡ eebb75b8-8ece-4e26-972b-ed962ce1dd42
Qσ = 2

# ╔═╡ 3a303018-4560-4553-96f6-1600b0e1d6c6
pσ = Normal(Qσ)

# ╔═╡ b629f310-dbf7-4b6e-9c3d-199a31a1bfb5
Qπ = 2

# ╔═╡ 51d10c40-5dce-47e2-aeba-958beadda7b8
pπ = Normal(Qπ)

# ╔═╡ c352df19-b653-43f3-b490-046447cea85a
phonemodel = NormalModel((pλ=pλ, pσ=pσ, pπ=pπ))

# ╔═╡ c47dcaf4-049a-45e1-853f-d589283b2038
# The dimension of the parameter space is determined 
# by the dimension of the observed space
K = Int(2D + (D^2 - D) / 2)

# ╔═╡ add6cb90-47bd-40e8-bbbb-34938714a79b
pL = JointNormalFixedCov(zeros(Qπ + 1, K))

# ╔═╡ 5415acd0-5e5f-40db-97eb-91b659a1ac5b
pS = JointNormalFixedCov(zeros(Qπ + 1, K))

# ╔═╡ b69c63d7-3f45-45d2-8278-05bc2fb5f3f1
md"""
### Hyper-parameters 

| name          | type            | value|  description                  | 
|:--------------|:----------------|:----:|:------------------------------|
| ``Q_\lambda`` | hyper-parameter | $Qλ  | language embedding dimension  |
| ``Q_\sigma``  | hyper-parameter | $Qσ  | speaker embedding dimension   |
| ``Q_\pi``     | hyper-parameter | $Qπ  | phone embedding dimension     | 
| ``D``         | constant        | $D   | observed space dimension      |
| ``K``         | constant        | $K   | phone model natural parameter dimension|
| ``L`` 		| constant 		  | $L   | number of languages in the data |
| ``S``         | constant        | $S   | number of speakers in  the data |
| ``P``         | constant        | $P   | number of phones in the data  |
"""

# ╔═╡ 946af84f-a5f4-4780-a8c5-13d4acbb9a80
md"""
### Model parameters summary

```math
\begin{align}

\boldsymbol{\theta} &= \Big\{ \{\mathbf{L}_i\}, \{\mathbf{S}_j \}, \{\boldsymbol{\lambda}_l \}, \{\boldsymbol{\sigma}_s \}, \{ \boldsymbol{\pi}_p \} \Big\} \\
\{\mathbf{L}_i \}, \; &\forall i \in \{1, \dots, Q_\lambda \} \\
\{\mathbf{S}_j \}, \;  &\forall j \in \{1, \dots, Q_\sigma \} \\
\{\mathbf{\boldsymbol{\lambda}}_l \}, \; &\forall l \in \{1, \dots, L\} \\ 
\{\mathbf{\boldsymbol{\sigma}}_s \}, \; &\forall s \in \{1, \dots, S\} \\
\{\mathbf{\boldsymbol{\pi}_s} \}, \; &\forall p \in \{1, \dots, P\}
\end{align}
```
"""

# ╔═╡ f2f5e843-3505-421e-9c9c-71a564387bb2
# Complete model: hierarchical generalized subspace model.
hgsm = GSMPhonetic(f, pL, pS, phonemodel)

# ╔═╡ 1dd9bfa7-b7c9-4075-9c6c-2f33871d5e0a
md"""
### Variational posterior

```math
\begin{align}
	q(\boldsymbol{\theta}) &= \prod_{m}q(\mathbf{L}_m) \prod_{n} q(\mathbf{S}_n) \prod_{l} q(\boldsymbol{\lambda}_l) \prod_{s} q(\boldsymbol{\sigma}_s) \prod_{p} q(\boldsymbol{\pi}_p) \\
	q(\mathbf{L}_m) &= \delta(\mathbf{L}_m - \boldsymbol{\mu}^{L}_m) \\ 
	q(\mathbf{S}_n) &=  \delta(\mathbf{S}_n - \boldsymbol{\mu}^{S}_n) \\ 
	q(\boldsymbol{\lambda}_l) &= \mathcal{N}(\boldsymbol{\mu}^{\lambda}_l, \boldsymbol{\Sigma}^{\lambda}_l) \\
	q(\boldsymbol{\sigma}_s) &= \mathcal{N}(\boldsymbol{\mu}^{\sigma}_s, \boldsymbol{\Sigma}^{\sigma}_s) \\
		q(\boldsymbol{\pi}_p) &= \mathcal{N}(\boldsymbol{\mu}^{\pi}_p, \boldsymbol{\Sigma}^{\pi}_p) 
\end{align}
```
"""

# ╔═╡ 6f1225eb-8df1-4991-8292-c0b5f2c45c10
q₀L = [JointNormalFixedCov(randn(Qπ + 1, K)) for i = 1:(Qλ+1)]

# ╔═╡ 349ef71a-55f8-4c9a-8216-d6fbce8d42e2
q₀S = [JointNormalFixedCov(randn(Qπ + 1, K)) for j = 1:(Qσ+1)]

# ╔═╡ b2875ebb-5b3b-4dad-b1e6-f5b38b231b03
L

# ╔═╡ 27fa667c-4a76-4fed-9afc-7482dc1339a0
q₀λ = [Normal(Qλ) for lang in languages]

# ╔═╡ 3a5cadd7-d15b-4da3-8a0b-9e7678514ff0
q₀σ = [Normal(Qσ) for spk in speakers]

# ╔═╡ e343d7f3-9a50-4952-b429-40b982b2b49f
q₀π = [Normal(Qπ) for phone in phones]

# ╔═╡ 42ccdf0d-8a5f-404d-8965-cccc7f85d164
q₀θ = (q₀L, q₀S, q₀λ, q₀σ, q₀π)

# ╔═╡ cde6061d-57af-4aae-a964-4cafabfdd5f1
begin	
	local qθ = q₀θ
	
	loss = []
	for t in 1:10
		println("t: $t")
		μ = getstats(hgsm, qθ)
		(l, (∇μ,)) = withgradient(μ -> elbo(hgsm, (labels, data), qθ, μ) / N, μ)		
		qθ = newposterior(hgsm, qθ, ∇μ; lrate=1e-2)
		
		push!(loss, l)
	end
	#qW, qH = qθ
	loss
end

# ╔═╡ 4274c7f4-38d5-4bf0-87ae-417d715d9b3b
gradient(μ -> elbo(hgsm, (labels, data), q₀θ, μ) / N, getstats(hgsm, q₀θ))		

# ╔═╡ 72afd7c5-6abb-495d-9bea-b2ae6306327a
elbo(hgsm, (labels, data), q₀θ, getstats(hgsm, q₀θ))

# ╔═╡ b033529c-1a3f-44f3-b546-954b4000edd1
unpack(hgsm, q₀θ, getstats(hgsm, q₀θ))

# ╔═╡ 7b905154-4bd3-47ed-ae9e-2227f0420bd4


# ╔═╡ Cell order:
# ╟─6958bde2-c50f-4afd-8d08-7dfbfd97169a
# ╠═3cd07b90-9e21-4a74-8d99-a6375beb7d1e
# ╟─b83311ab-ca4f-432e-8627-5207517a81db
# ╠═c01de0cd-43f8-49cc-a8c9-ffbbc182d9e9
# ╠═e9a8feec-eb20-4c55-8669-6c6c085b4d85
# ╠═4fd12bad-b244-428e-8faa-f6000d0d9e3a
# ╠═dc2d02f7-2044-483c-99b1-b864c6940223
# ╠═29b92c17-36dd-49c5-afdf-21c2ed759fba
# ╠═32804074-8126-4328-9b2b-9785690c76e5
# ╟─8b535f17-7a55-42da-ba7a-c161bec9fd28
# ╠═4c69229d-a272-49eb-9dc1-484c53bad338
# ╠═3a303018-4560-4553-96f6-1600b0e1d6c6
# ╠═51d10c40-5dce-47e2-aeba-958beadda7b8
# ╟─5175712e-7f27-459e-8e6a-9bed325d6b4e
# ╠═add6cb90-47bd-40e8-bbbb-34938714a79b
# ╠═5415acd0-5e5f-40db-97eb-91b659a1ac5b
# ╟─763b5f95-db02-4ede-b632-848a646232d9
# ╠═c352df19-b653-43f3-b490-046447cea85a
# ╟─841a3f0e-8975-49d2-959e-afe832ed41e4
# ╠═b0761304-49df-4626-a677-488295dc786d
# ╟─b69c63d7-3f45-45d2-8278-05bc2fb5f3f1
# ╠═4154a267-326d-495c-aa60-156560aa5f72
# ╠═eebb75b8-8ece-4e26-972b-ed962ce1dd42
# ╠═b629f310-dbf7-4b6e-9c3d-199a31a1bfb5
# ╠═c47dcaf4-049a-45e1-853f-d589283b2038
# ╟─946af84f-a5f4-4780-a8c5-13d4acbb9a80
# ╠═f2f5e843-3505-421e-9c9c-71a564387bb2
# ╟─1dd9bfa7-b7c9-4075-9c6c-2f33871d5e0a
# ╠═6f1225eb-8df1-4991-8292-c0b5f2c45c10
# ╠═349ef71a-55f8-4c9a-8216-d6fbce8d42e2
# ╠═b2875ebb-5b3b-4dad-b1e6-f5b38b231b03
# ╠═27fa667c-4a76-4fed-9afc-7482dc1339a0
# ╠═3a5cadd7-d15b-4da3-8a0b-9e7678514ff0
# ╠═e343d7f3-9a50-4952-b429-40b982b2b49f
# ╠═42ccdf0d-8a5f-404d-8965-cccc7f85d164
# ╠═cde6061d-57af-4aae-a964-4cafabfdd5f1
# ╠═4274c7f4-38d5-4bf0-87ae-417d715d9b3b
# ╠═72afd7c5-6abb-495d-9bea-b2ae6306327a
# ╠═b033529c-1a3f-44f3-b546-954b4000edd1
# ╠═7b905154-4bd3-47ed-ae9e-2227f0420bd4
