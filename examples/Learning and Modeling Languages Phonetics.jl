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

### Getting the data

We download the data and store it in a local file.
"""

# ╔═╡ d1588a1c-babe-4223-bc6d-f353314ed2c0
localfile = "formants.tsv"

# ╔═╡ 262e0395-814f-4315-b25e-21c18a7a9a5f
begin
	# Database URL
	local dataurl = "https://raw.githubusercontent.com/lucasondel/LinguaLibreFormants/main/data/formants.tsv"

	# Download the data if it is not done already
	if ! isfile(localfile)
		run(`wget $dataurl`)
	end
end

# ╔═╡ 4bbdd3d2-eb62-4889-9c8a-20141c0b140a
md"""
### Data preparation

In a nutshell:
1. we extract the list of languages, speakers and phones
2. organized the data as a $3 \times N$ matrix where 3 is for 3 first formants and $N$ is the number of data points
3. for each data point we extract the annotation (language, speaker, phone)
"""

# ╔═╡ ee1d1c45-70f5-4fe7-a8e2-9134d45a2b95
begin
	local rawdata = readdlm(localfile)
	
	speakers = Set()
	languages = Set()
	phones = Set()
	labels = []
	data = []
	for (lang, spk, ipaphone, f1, f2, f3, duration) in eachrow(rawdata)
		# Remove the duration symbol
		ipaphone = replace(ipaphone, "ː" => "")
	
		lang, spk, ipaphone = Symbol(lang), Symbol(spk), Symbol(ipaphone)
		push!(data, vcat(f1, f2, f3))
		
		phone = (lang, ipaphone)
		spk = (lang, spk)
		
		push!(labels, (lang, spk, phone))
		push!(speakers, spk)
		push!(languages, lang)
		push!(phones, phone)
	end
	data = hcat(data...)
	labels = vcat(labels...)

	# Mean-variance normalization of the data
	local x̄ = sum(data, dims=2) / size(data, 2)
	local x̄² = sum(data .^ 2, dims=2) / size(data, 2)
	local var = x̄² - (x̄ .^ 2)
	data = (data .- x̄) ./ sqrt.(var)

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
end

# ╔═╡ 474abeb1-82f8-4a74-a03c-be1cf55df840
md"Some important constants:"

# ╔═╡ 31e56fed-e4f0-4c71-abb2-5c2eff5a9a47
D = size(data, 1)

# ╔═╡ 61cc1ab2-a615-4a9b-91f9-bc5909e29297
N = size(data, 2)

# ╔═╡ e08cc384-49dc-4585-a2ab-fbec9fcfe9d2
L = length(languages)

# ╔═╡ b8e7f92f-0065-46fe-ac6b-5adf29b3379f
S = length(speakers)

# ╔═╡ ef71661f-338e-4d57-86ed-fc2c308a744f
P = length(phones)

# ╔═╡ c01de0cd-43f8-49cc-a8c9-ffbbc182d9e9
begin
	md"""
	### Data statistics
	
	| \# languages | \# speakers  | \# phones |
	|:------------:|:------------:|:---------:|
	| $L           | $S           | $P        |

	"""
end

# ╔═╡ 8d416ab7-3ae7-4984-a7e6-1f56692d1b7b
begin
	mdstr = "| language | # speakers | phones |\n"
	mdstr *= "|:---------|:-----------:|:-------|\n"
	for lang in languages
		mdstr *= "| $lang | "

		ns = 0
		for (l, spk) in speakers
			if l == lang ns += 1end
		end
		mdstr *= "$ns | "
		
		for (l, phone) in phones
			if l == lang
				mdstr *= "$phone "
			end
		end
		mdstr *= "|\n"
	end
	Markdown.parse(mdstr)
end

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

# ╔═╡ 85e89c27-0371-42a9-8b3c-4fc677d1f7a7
md"""
### Training


"""

# ╔═╡ cde6061d-57af-4aae-a964-4cafabfdd5f1
begin	
	qθ = q₀θ
	
	loss = []
	for t in 1:1000
		μ = getstats(hgsm, qθ)
		(l, (∇μ,)) = withgradient(μ -> elbo(hgsm, groups, qθ, μ) / N, μ)		
		qθ = newposterior(hgsm, qθ, ∇μ; lrate=1e-2)
		
		push!(loss, l)
	end
	loss
end

# ╔═╡ 7175c939-bc9d-4b60-b145-b09cac018b7f
plot(loss[100:1:end], legend=false)

# ╔═╡ 4274c7f4-38d5-4bf0-87ae-417d715d9b3b
begin
	local qλ = qθ[3]
	local p = plot(legend=false)
	for (i, q) in enumerate(qλ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(2, 1, hxxᵀ) - x * x'
		annotate!(m[1], m[2], id2lang[i])
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ ef349252-cf35-42bd-b562-7ec28bef7cdd
begin
	local qσ = qθ[4]
	local p = plot(legend=false)
	for (i, q) in enumerate(qσ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(2, 1, hxxᵀ) - x * x'
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ b9bba539-ba31-420e-a718-a952db80d4c2
begin
	local qπ = qθ[5]
	local p = plot(legend=false, xlims=(-2, 2), ylims=(-2, 2))
	for (i, q) in enumerate(qπ)
		if i == 28 continue end
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(2, 1, hxxᵀ) - x * x'
		annotate!(m[1], m[2], id2phone[i][2])
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ ab7167f2-41c4-4944-b769-d7d27c9bda12
id2phone

# ╔═╡ 8b939634-e3f2-4a29-91de-4ed56787bc78
function minimum_divergence(hgsm, qθ)
	qL, qS, qλ, qσ, qπ = qθ

	μπ = map(μ, qπ)
	stdπ = stdparams.(qπ, μπ)
	
	μπ₀ = sum(μπ) / length(qπ)
	m₀, Σ₀ = stdparams(hgsm.model.pθ.pπ, μπ₀)
	L₀ = cholesky(Σ₀).L
	L₀⁻¹ = L₀

	[Normal(L₀⁻¹ * (m - m₀), Symmetric(L₀⁻¹ * Σ * L₀⁻¹')) for (m, Σ) in stdπ]
end

# ╔═╡ b44032e6-bdff-4f43-a225-70cddc2d89cd
minimum_divergence(hgsm, qθ)

# ╔═╡ d045db5e-e646-4f04-9df3-be877cd505f9
m, Σ = stdparams(qλ[1], unpack(qλ[1], sum(μ, qλ) / length(qλ)))

# ╔═╡ b518b13e-f470-4c4b-8c2e-e3b0803beb62
qθ[1][1].M

# ╔═╡ ec85819b-1b0f-4f0e-9602-46c98402bfdf
Lm⁻¹ = inv(Lm)

# ╔═╡ Cell order:
# ╟─6958bde2-c50f-4afd-8d08-7dfbfd97169a
# ╠═3cd07b90-9e21-4a74-8d99-a6375beb7d1e
# ╟─b83311ab-ca4f-432e-8627-5207517a81db
# ╠═d1588a1c-babe-4223-bc6d-f353314ed2c0
# ╠═262e0395-814f-4315-b25e-21c18a7a9a5f
# ╟─4bbdd3d2-eb62-4889-9c8a-20141c0b140a
# ╠═ee1d1c45-70f5-4fe7-a8e2-9134d45a2b95
# ╟─474abeb1-82f8-4a74-a03c-be1cf55df840
# ╠═31e56fed-e4f0-4c71-abb2-5c2eff5a9a47
# ╠═61cc1ab2-a615-4a9b-91f9-bc5909e29297
# ╠═e08cc384-49dc-4585-a2ab-fbec9fcfe9d2
# ╠═b8e7f92f-0065-46fe-ac6b-5adf29b3379f
# ╠═ef71661f-338e-4d57-86ed-fc2c308a744f
# ╟─c01de0cd-43f8-49cc-a8c9-ffbbc182d9e9
# ╟─8d416ab7-3ae7-4984-a7e6-1f56692d1b7b
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
# ╟─85e89c27-0371-42a9-8b3c-4fc677d1f7a7
# ╠═cde6061d-57af-4aae-a964-4cafabfdd5f1
# ╠═7175c939-bc9d-4b60-b145-b09cac018b7f
# ╠═4274c7f4-38d5-4bf0-87ae-417d715d9b3b
# ╠═ef349252-cf35-42bd-b562-7ec28bef7cdd
# ╠═b9bba539-ba31-420e-a718-a952db80d4c2
# ╠═ab7167f2-41c4-4944-b769-d7d27c9bda12
# ╠═8b939634-e3f2-4a29-91de-4ed56787bc78
# ╠═b44032e6-bdff-4f43-a225-70cddc2d89cd
# ╠═d045db5e-e646-4f04-9df3-be877cd505f9
# ╠═b518b13e-f470-4c4b-8c2e-e3b0803beb62
# ╠═ec85819b-1b0f-4f0e-9602-46c98402bfdf
