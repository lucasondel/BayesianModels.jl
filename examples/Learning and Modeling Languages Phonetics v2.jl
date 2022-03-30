### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 3cd07b90-9e21-4a74-8d99-a6375beb7d1e
begin
	using Revise
	using Pkg
	Pkg.activate("../")
	
	using BayesianModels
	using DelimitedFiles
	using ForwardDiff
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
	phones_per_lang = Dict()
	labels = []
	data = []
	for (lang, spk, sex, ipaphone, f1, f2, f3, duration) in eachrow(rawdata)
		# Remove the duration symbol
		ipaphone = replace(ipaphone, "ː" => "")
	
		lang, spk, ipaphone = Symbol(lang), Symbol(spk), Symbol(ipaphone)
		push!(data, vcat(f1, f2, f3))
		
		phone = ipaphone
		spk = (lang, sex, spk)

		l = get(phones_per_lang, lang, Set())
		push!(l, phone)
		phones_per_lang[lang] = l
		
		
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
1.  $\mathscr{L}_i \sim p(\mathscr{L}) \triangleq \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\{\mathscr{L}_0, \dots, \mathscr{L}_{Q_\lambda} \}$ → Language hyper-subspace bases of dimension $(Q_\pi + 1) \times K$
2.  $\mathscr{S}_j \sim p(\mathscr{S}) \triangleq \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\{\mathscr{S}_0, \dots, \mathscr{S}_{Q_\sigma} \}$ → Speaker hyper-subspace bases $(Q_\pi + 1) \times K$
3.  $\mathscr{P} = \sum_{i=0}^{Q_\lambda} \lambda_i \mathscr{L}_i + \sum_{j=0}^{Q_\sigma} \sigma_j \mathscr{S}_j$ where $\lambda_0 = \sigma_0 = 1$, → Phonetic subspace bases of dimension $(Q_\pi + 1) \times K$
"""

# ╔═╡ 763b5f95-db02-4ede-b632-848a646232d9
md"""
### Phone model 
1.  $\boldsymbol{\psi} = \mathscr{P}^\top \begin{bmatrix} \boldsymbol{\pi} \\ 1 \end{bmatrix}$, $\boldsymbol{\psi} \in \mathbb{R}^K$
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
Qσ = 10

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
\{\mathbf{\boldsymbol{\pi}_p} \}, \; &\forall p \in \{1, \dots, P\}
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
q₀L = [JointNormalFixedCov(i == Qλ + 1 ? zeros(Qπ + 1, K) : randn(Qπ + 1, K) / K)
	   for i = 1:(Qλ+1)]

# ╔═╡ 349ef71a-55f8-4c9a-8216-d6fbce8d42e2
q₀S = [JointNormalFixedCov(j == Qσ + 1 ? zeros(Qπ + 1, K) : randn(Qπ + 1, K) / K)
	   for j = 1:(Qσ+1)]

# ╔═╡ 27fa667c-4a76-4fed-9afc-7482dc1339a0
q₀λ = [Normal(zeros(Qλ), Matrix(I, Qλ, Qλ)) for lang in languages]

# ╔═╡ 3a5cadd7-d15b-4da3-8a0b-9e7678514ff0
q₀σ = [Normal(zeros(Qσ), Matrix(I, Qσ, Qσ)) for spk in speakers]

# ╔═╡ e343d7f3-9a50-4952-b429-40b982b2b49f
q₀π = [Normal(zeros(Qπ), Matrix(I, Qπ, Qπ)) for phone in phones]

# ╔═╡ 42ccdf0d-8a5f-404d-8965-cccc7f85d164
q₀θ = (q₀L, q₀S, q₀λ, q₀σ, q₀π)

# ╔═╡ ffe22379-2f91-4b54-8f74-413ac6d2ce08
md"""### Minimum Divergence"""

# ╔═╡ 4d374801-703c-4fb0-b77d-bbb41fbfd54b
function renorm_hyperbases(qM, qh, ph)
	μh = map(μ, qh)
	stdh = stdparams.(qh, μh)

	μh₀ = sum(μh) / length(qh)
	m₀, Σ₀ = stdparams(ph, μh₀)
	L₀ = cholesky(Σ₀).L
	L₀⁻¹ = inv(L₀)

	new_qh = [Normal(L₀⁻¹ * (m - m₀), Symmetric(L₀⁻¹ * Σ * L₀⁻¹')) for (m, Σ) in stdh]

	B = vec.(getproperty.(qM, (:M)))
	dims = size(qM[1].M)
	M, b = hcat(B[1:end-1]...), B[end] 
	new_M = reshape.(eachcol(M * L₀), dims...)
	new_b = reshape(b + M * m₀, dims...)

	new_qM = [JointNormalFixedCov(V) for V in [new_M..., new_b]]
	new_qM, new_qh
end

# ╔═╡ 546fd344-42e2-4881-9607-da99db000c33
function renorm_bases(qL, qS, qh, ph)
	μh = map(μ, qh)
	stdh = stdparams.(qh, μh)

	μh₀ = sum(μh) / length(qh)
	m₀, Σ₀ = stdparams(ph, μh₀)
	L₀ = cholesky(Σ₀).L
	L₀⁻¹ = inv(L₀)

	new_qh = [Normal(L₀⁻¹ * (m - m₀), Symmetric(L₀⁻¹ * Σ * L₀⁻¹')) for (m, Σ) in stdh]

	Ms = getproperty.(qL, (:M))
	new_Ms = map(Ms) do M
		W, b = M[1:end-1,:], M[end:end, :]
		vcat(L₀' * W, b + m₀' * W)
	end
	new_qL = [JointNormalFixedCov(V) for V in new_Ms]

	Ms = getproperty.(qS, (:M))
	new_Ms = map(Ms) do M
		W, b = M[1:end-1,:], M[end:end, :]
		vcat(L₀' * W, b + m₀' * W)
	end
	new_qS = [JointNormalFixedCov(V) for V in new_Ms]

	new_qL, new_qS, new_qh
end

# ╔═╡ 8b939634-e3f2-4a29-91de-4ed56787bc78
function minimum_divergence(hgsm, qθ)
	qL, qS, qλ, qσ, qπ = qθ

	qL2, qλ2 = renorm_hyperbases(qL, qλ, hgsm.model.pθ.pλ)
	qS2, qσ2 = renorm_hyperbases(qS, qσ, hgsm.model.pθ.pσ)
	qL2, qS2, qπ2 = renorm_bases(qL2, qS2, qπ, hgsm.model.pθ.pπ)
	
	qL2, qS2, qλ2, qσ2, qπ2
end

# ╔═╡ 85e89c27-0371-42a9-8b3c-4fc677d1f7a7
md"""
### Training


"""

# ╔═╡ 2822d23b-197d-4dcc-85d0-3df653fcf3b6
begin	
	qθ_md = q₀θ

	kls_md = []
	loss_md = []
	for t in 1:100000
		μ = getstats(hgsm, qθ_md)
		(l, (∇μ,)) = withgradient(μ -> elbo(hgsm, groups, qθ_md, μ) / N, μ)		

		qθ_md = newposterior(hgsm, qθ_md, ∇μ; lrate=1e-2)

		if t % 100 == 0
			qθ_md = minimum_divergence(hgsm, qθ_md)
		end

		push!(kls_md, kldiv(hgsm, qθ_md, getstats(hgsm, qθ_md)))
		push!(loss_md, l)
	end
	loss_md
end

# ╔═╡ 5162a02c-b803-4be5-bff8-801ec24001d3
plot(loss_md[1000:1:end], legend=false, linealpha=0.5)

# ╔═╡ 05c5717a-c0fa-47a4-9de9-c747b6fb3f93
md"""### Visualization"""

# ╔═╡ 4274c7f4-38d5-4bf0-87ae-417d715d9b3b
begin
	local qλ = qθ_md[3]
	local p = plot(legend=false)
	for (i, q) in enumerate(qλ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(Qλ, 1, hxxᵀ) - x * x'
		m = [m[1], m[2]]
		Σ = Σ[1:2,1:2]
		annotate!(m[1], m[2], id2lang[i])
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ aab42b7b-b269-4e6f-af74-f09ec9c6b862
begin
	local qσ = qθ_md[4]
	local p = plot(size=(800,800), legend=false, xlims=(-3, 3), ylims=(-3, 3))
	for (i, q) in enumerate(qσ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(Qσ, 1, hxxᵀ) - x * x'
		m = [m[1], m[2]]
		Σ = [Σ[1,1] Σ[1,2]; Σ[2, 1] Σ[2,2]] #Σ[1:2,1:2]
		#annotate!(m[1], m[2], speakers[i][1])
		color = speakers[i][2] == "male" ? :blue : :green
		plotnormal!(m, Σ, σ=2, fillcolor=color, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ b9bba539-ba31-420e-a718-a952db80d4c2
begin
	local qπ = qθ_md[5]
	local d1, d2 = 1, 2
	local p = plot(size=(600, 500), legend=false, xlabel="π₁", ylabel="π₂")
	for (i, q) in enumerate(qπ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(Qπ, 1, hxxᵀ) - x * x'
		m = [m[d1], m[d2]]
		Σ = [Σ[d1, d1] Σ[d1, d2]; Σ[d2, d1] Σ[d2, d2]]
		annotate!(m[1], m[2], id2phone[i])
		plotnormal!(m, Σ, σ=2, fillcolor=:blue, fillalpha=0.1, linealpha=0, label=false)
	end
	p
end

# ╔═╡ f7d8e930-3d7b-4135-bb64-a8b17367f683
qL, qS, qλ, qσ, qπ = qθ_md

# ╔═╡ cd347af2-ad3e-4e7d-b471-54b668511e69
function hgsm_η(hgsm::GSMPhonetic, qL, qS, λ, σ, π)
	L = getproperty.(qL, (:M,))
	S = getproperty.(qS, (:M,))
	P = sum(L .* vcat(λ, 1)) + sum(S .* vcat(σ, 1))
	ψ = P' * vcat(π, 1)
	Λμ, μᵀΛμ, Λ, logdetΛ = hgsm.f(ψ)
	vcat(Λμ, -(1/2)*diag(Λ), -vech(Λ, 1))
end

# ╔═╡ 7d457821-67c0-47e0-9c00-7bd2a6560cc8
hgsm_η(hgsm, qL, qS, qλ[1].μ, qσ[1].μ, qπ[1].μ)

# ╔═╡ bffb14d2-1173-4511-80b2-0676af0cd3f8
function hgsm_μ(hgsm::GSMPhonetic, η)
	L = length(η)
	D = Int((-(3/2) + sqrt((9/4) + 2L)))

	Λμ = η[1:D]
	diagΛ = -2 * η[D+1:2*D]
	vechΛ = - η[2*D+1:end]
	Λ = diagm(diagΛ) + CompressedSymmetric(D, 1, vechΛ)

	Σ = inv(Symmetric(Λ))
	μ = Σ * Λμ
	xxᵀ = Σ + μ * μ'
	vcat(μ, diag(xxᵀ), vech(xxᵀ, 1))
end

# ╔═╡ 3a4921ab-d8e4-463b-ae41-d9ac4b034c41
hgsm_μ(hgsm, [0., 0., -1, -1, 0.5])

# ╔═╡ c3f6fc47-2843-498d-8d86-0c99e8638ff4
hgsm_μ(hgsm, hgsm_η(hgsm, qL, qS, qλ[1].μ, qσ[1].μ, qπ[1].μ))

# ╔═╡ 5d54df87-0d8a-4371-93e9-cc20fd659974
function hgsm_A(hgsm::GSMPhonetic, η)
	L = length(η)
	D = Int((-(3/2) + sqrt((9/4) + 2L)))

	Λμ = η[1:D]
	diagΛ = -2 * η[D+1:2*D]
	vechΛ = - η[2*D+1:end]
	Λ = diagm(diagΛ) + CompressedSymmetric(D, 1, vechΛ)
	Σ = inv(Λ)
	
	-(1/2) * logdet(Λ) + (1/2) * Λμ' * (Σ * Λμ) 
end

# ╔═╡ f0d90bdc-aa48-4edf-a673-e6dc6d27e2e0
function trajectory_λ(hgsm, qL, qS, λ₁, λ₂, σ, π; step=1e-2)
	traj = [λ₁]
	dist = 0
	H = η -> ForwardDiff.hessian(η -> hgsm_A(hgsm, η), η)
	J = λ -> ForwardDiff.jacobian(λ -> hgsm_η(hgsm, qL, qS, λ, σ, π), λ)
	
	η₂ = hgsm_η(hgsm, qL, qS, λ₂, σ, π)
	
	while norm(traj[end] - λ₂) > 1e-1
		λ = traj[end]
		ηₜ = hgsm_η(hgsm, qL, qS, λ, σ, π)
		μₜ = hgsm_μ(hgsm, ηₜ)
		Jλ = J(λ)
		∇λ = (1/2) * Jλ' * (η₂ - ηₜ)
		#∇λ = -Jλ' * (ηₜ - (η₂ + μₜ))
		#∇λ = (λ₂ - λ)
		∇λ = ∇λ ./ norm(∇λ)
	
		G = Jλ' * Symmetric(H(ηₜ)) * Jλ
		v = ∇λ #inv(G) * ∇λ
		v = step * v ./ norm(v)
		dist += sqrt((step * ∇λ)' * G * (step * ∇λ))
		#dist += sqrt((step * ∇λ)' * G * (step * ∇λ))
		
		push!(traj, λ + v)
	end
	hcat(traj...), dist
end

# ╔═╡ 0d8c54e2-3677-4d5d-adc1-b6b3f482e409
t1, dist1 = trajectory_λ(hgsm, qL, qS, qλ[1].μ, qλ[6].μ, qσ[1].μ, qπ[1].μ)

# ╔═╡ b4c44e20-ffbc-4376-8cc6-3d9b1f64af9a
t2, dist2 = trajectory_λ(hgsm, qL, qS, qλ[6].μ, qλ[1].μ, qσ[1].μ, qπ[1].μ)

# ╔═╡ b317dbf6-2a60-400f-96d5-addd740be016
begin
	# Manual sorting so that languages from the same family appear together
	sorted_languages = [:spa, :ita, :fra, :rus, :pol, :deu]
	
	dist_lang = zeros(L, L)
	local idmap = Dict(i => findall(l -> l == lang, languages)[1] for (i, lang) in enumerate(sorted_languages))
	for i = 1:L, j = 1:L
		x, y = idmap[i], idmap[j]
		for phone in phones_per_lang[languages[x]]
			p = phone2id[phone]
			println(i, ": ", languages[x], " ", j, ": ", languages[y], " ", phones[p])
			traj, Δ = trajectory_λ(hgsm, qL, qS, qλ[x].μ, qλ[y].μ, zeros(Qσ), qπ[p].μ)
			dist_lang[i, j] += Δ / length(phones_per_lang[languages[x]])
		end
	end
end

# ╔═╡ 8d416ab7-3ae7-4984-a7e6-1f56692d1b7b
begin
	mdstr = "| language | # speakers | phones |\n"
	mdstr *= "|:---------|:-----------:|:-------|\n"
	for lang in sorted_languages
		mdstr *= "| $lang | "

		ns = 0
		for (l, spk) in speakers
			if l == lang ns += 1end
		end
		mdstr *= "$ns | "
		
		for phone in sort(collect(phones_per_lang[lang]))
			mdstr *= "$phone "
		end
		mdstr *= "|\n"
	end
	Markdown.parse(mdstr)
end

# ╔═╡ 69777c50-240b-4be7-a824-b5dd0eb5fb7c
languages

# ╔═╡ 00d323cd-b928-49cd-a590-0e8852ffb264
begin
	heatmap(
		dist_lang, 
		xticks=(1:6, sorted_languages), 
		yticks=(1:6, sorted_languages), 
		tickfont = (14,),
		xmirror=true,
		yflip=true, 
		c=:grays
	)
end 

# ╔═╡ 4d44afe7-b0dc-40b1-a00d-73ad346e0576
dist_lang

# ╔═╡ cea584da-2d0e-4c4f-a964-2f7254bf74aa
languages

# ╔═╡ 295da023-8cb9-4b13-99f8-fa7690c20b5c
begin
	local d1, d2 = 1, 2
	local qλ = qθ_md[3]
	local p = plot(legend=false)
	for (i, q) in enumerate(qλ)
		x, diagxxᵀ, hxxᵀ = unpack(q, μ(q))
		m = x
		Σ = diagm(diagxxᵀ) + CompressedSymmetric(Qλ, 1, hxxᵀ) - x * x'
		m = [m[d1], m[d2]]
		Σ = [Σ[d1, d1] Σ[d1, d2]; Σ[d2, d1] Σ[d2, d2]]
		annotate!(m[1], m[2], id2lang[i])
		plot!(t1[d1, :], t1[d2, :], linecolor=:black)
		plot!(t2[d1, :], t2[d2, :], linecolor=:black)
		plotnormal!(m, Σ, σ=2, fillalpha=0.4, linealpha=0, label=false)
	end
	p
end

# ╔═╡ d15d2ae4-01b4-4312-b921-2fac2d943036
function hgsm_llh(hgsm::GSMPhonetic, X, qL, qS, λ, σ, π)
	L = getproperty.(qL, (:M,))
	S = getproperty.(qS, (:M,))
	P = sum(L .* vcat(λ, 1)) + sum(S .* vcat(σ, 1))
	ψ = P' * vcat(π, 1)
	#Λμ, μᵀΛμ, Λ, logdetΛ = hgsm.f(ψ)
	#vcat(Λμ, -(1/2)*diag(Λ), -vech(Λ, 1))
	loglikelihood(hgsm.model, X, hgsm.f(ψ))
end

# ╔═╡ fa30a558-5bed-4ce1-8be2-ed55a3d7a9f8
function hgsm_stdparams(hgsm::GSMPhonetic, qL, qS, λ, σ, π)
	L = getproperty.(qL, (:M,))
	S = getproperty.(qS, (:M,))
	P = sum(L .* vcat(λ, 1)) + sum(S .* vcat(σ, 1))
	ψ = P' * vcat(π, 1)
	Λμ, μᵀΛμ, Λ, logdetΛ = hgsm.f(ψ)
	Σ = inv(Λ)
	μ = Σ * Λμ
	Σ, μ
end

# ╔═╡ a36c7fd9-6e47-4386-a9e0-cb9a9dc6a644
id2spk

# ╔═╡ 0f26901f-e592-4768-ac67-ee526a640ede
phone2id

# ╔═╡ 8509deeb-f156-4350-a21a-d1d70949091d
lang2id

# ╔═╡ 696d1d7d-68df-45c3-b5d9-0cae311e553a
repeat([2], 10)

# ╔═╡ 6836772d-67c1-475f-8d99-e11fc7f752a0
begin
	local rawdata = readdlm(localfile)
	local data = []

	#pdataraw = Dict("œ" => [], "ɛ" => [])
	#pdataraw = Dict("ã" => [], "õ" => [])
	#pdataraw = Dict("ẽ" => [], "õ" => [])
	pdataraw = Dict("$p" => [] for p in phones)
	testphones = collect(keys(pdataraw))
	for (lang, spk, sex, ipaphone, f1, f2, f3, duration) in eachrow(rawdata)
		# Remove the duration symbol
		ipaphone = replace(ipaphone, "ː" => "")
		
		lang, spk, ipaphone = lang, spk, ipaphone
		
		#if lang == "fra" && spk == "spk8" && sex == "male" && ipaphone in keys(pdataraw)
		if ipaphone in keys(pdataraw)
			push!(pdataraw[ipaphone], vcat(f1, f2, f3))
		end
		push!(data, vcat(f1, f2, f3))
	end
	
	pdata = Dict()
	for (k, v) in pdataraw
		pdata[k] = hcat(v...)
	end
	data = hcat(data...)
	
	local x̄ = sum(data, dims=2) / size(data, 2)
	local x̄² = sum(data .^ 2, dims=2) / size(data, 2)
	local var = x̄² - (x̄ .^ 2)
	for (k, v) in pdata
		pdata[k] = (v .- x̄) ./ sqrt.(var)
	end
	pdata 
end

# ╔═╡ 9c9d2747-b6db-4957-84be-3c327c4dc1df
begin
	local spk = 3
	local lang = 4
	#local testphones = ["ẽ", "õ"] #["œ", "ɛ"]
	p = plot(xlims=(-3, 3), ylims=(-3, 3))
	for phone in testphones
		i = phone2id[Symbol(phone)]
		Σ, m = hgsm_stdparams(hgsm, qL, qS, qλ[lang].μ, qσ[spk].μ, qπ[i].μ)
		plotnormal!(m[1:2], Σ[1:2,1:2], σ=2, fillalpha=0.4, linealpha=0, label="$phone")
	end

	for phone in testphones
		scatter!(pdata["$phone"][1,1:10], pdata["$phone"][2,:], label="$phone")
	end
	p
end

# ╔═╡ d711ad75-3fc6-4726-a506-962aaccb2de7
begin 
	local p = plot()
	for phone in testphones
		scatter!(pdata["$phone"][1,1:10], pdata["$phone"][2,1:10], label="$phone")
	end
	p
end

# ╔═╡ c80d9947-e02e-4fb0-bcb3-e29a4702688e
begin
	local spk = 3
	local i = phone2id[Symbol(testphones[1])]
	local j = phone2id[Symbol(testphones[2])]
	local errors = zeros(L)
	
	for lang in 1:L
	
		llh1 = vcat(
			hgsm_llh(hgsm, pdata[testphones[1]], qL, qS, qλ[lang].μ, qσ[spk].μ, qπ[i].μ),
			hgsm_llh(hgsm, pdata[testphones[1]], qL, qS, qλ[lang].μ, qσ[spk].μ, qπ[j].μ)
		)

		llh2 = vcat(
			hgsm_llh(hgsm, pdata[testphones[2]], qL, qS, qλ[lang].μ, qσ[spk].μ, qπ[i].μ),
			hgsm_llh(hgsm, pdata[testphones[2]], qL, qS, qλ[lang].μ, qσ[spk].μ, qπ[j].μ)
		)
		errors[lang] = ( sum(llh1[1,:] .> llh1[2,:]) + sum(llh2[1, :] .< llh2[2, :]) ) / (size(llh1, 2) + size(llh2, 2))
	end
	errors
end

# ╔═╡ 43692f77-f402-41c7-b22e-0b8a42bf8451
speakers

# ╔═╡ 6d1e2c27-4c08-4324-af85-45ed1c4950e0
local rawdata = readdlm(localfile)

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
# ╠═27fa667c-4a76-4fed-9afc-7482dc1339a0
# ╠═3a5cadd7-d15b-4da3-8a0b-9e7678514ff0
# ╠═e343d7f3-9a50-4952-b429-40b982b2b49f
# ╠═42ccdf0d-8a5f-404d-8965-cccc7f85d164
# ╟─ffe22379-2f91-4b54-8f74-413ac6d2ce08
# ╠═4d374801-703c-4fb0-b77d-bbb41fbfd54b
# ╠═546fd344-42e2-4881-9607-da99db000c33
# ╠═8b939634-e3f2-4a29-91de-4ed56787bc78
# ╟─85e89c27-0371-42a9-8b3c-4fc677d1f7a7
# ╠═2822d23b-197d-4dcc-85d0-3df653fcf3b6
# ╠═5162a02c-b803-4be5-bff8-801ec24001d3
# ╟─05c5717a-c0fa-47a4-9de9-c747b6fb3f93
# ╠═4274c7f4-38d5-4bf0-87ae-417d715d9b3b
# ╠═aab42b7b-b269-4e6f-af74-f09ec9c6b862
# ╠═b9bba539-ba31-420e-a718-a952db80d4c2
# ╠═f7d8e930-3d7b-4135-bb64-a8b17367f683
# ╠═cd347af2-ad3e-4e7d-b471-54b668511e69
# ╠═7d457821-67c0-47e0-9c00-7bd2a6560cc8
# ╠═bffb14d2-1173-4511-80b2-0676af0cd3f8
# ╠═3a4921ab-d8e4-463b-ae41-d9ac4b034c41
# ╠═c3f6fc47-2843-498d-8d86-0c99e8638ff4
# ╠═5d54df87-0d8a-4371-93e9-cc20fd659974
# ╠═f0d90bdc-aa48-4edf-a673-e6dc6d27e2e0
# ╠═0d8c54e2-3677-4d5d-adc1-b6b3f482e409
# ╠═b4c44e20-ffbc-4376-8cc6-3d9b1f64af9a
# ╠═b317dbf6-2a60-400f-96d5-addd740be016
# ╠═69777c50-240b-4be7-a824-b5dd0eb5fb7c
# ╠═00d323cd-b928-49cd-a590-0e8852ffb264
# ╠═4d44afe7-b0dc-40b1-a00d-73ad346e0576
# ╠═cea584da-2d0e-4c4f-a964-2f7254bf74aa
# ╠═295da023-8cb9-4b13-99f8-fa7690c20b5c
# ╠═d15d2ae4-01b4-4312-b921-2fac2d943036
# ╠═fa30a558-5bed-4ce1-8be2-ed55a3d7a9f8
# ╠═9c9d2747-b6db-4957-84be-3c327c4dc1df
# ╠═d711ad75-3fc6-4726-a506-962aaccb2de7
# ╠═c80d9947-e02e-4fb0-bcb3-e29a4702688e
# ╠═a36c7fd9-6e47-4386-a9e0-cb9a9dc6a644
# ╠═0f26901f-e592-4768-ac67-ee526a640ede
# ╠═8509deeb-f156-4350-a21a-d1d70949091d
# ╠═696d1d7d-68df-45c3-b5d9-0cae311e553a
# ╠═6836772d-67c1-475f-8d99-e11fc7f752a0
# ╠═43692f77-f402-41c7-b22e-0b8a42bf8451
# ╠═6d1e2c27-4c08-4324-af85-45ed1c4950e0
