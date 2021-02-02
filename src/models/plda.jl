# Probabilistic Linear Discriminant Analysis (PLDA) model.
#
# Lucas Ondel 2020

#######################################################################
# Model definition

"""
    struct PLDA{T,D,Q1,Q2}
        hprior_within_class
        hprior_across_class
        trans_within_class
        trans_across_class
        λ
    end

Standard PLDA model. `Q1` is the dimension of the within class subspace
and `Q2` is the dimension across class subspace.
"""
struct PLDA{T,D,Q1,Q2}
    hprior_within_class::Normal{T,Q1}
    hprior_across_class::Normal{T,Q2}
    trans_within_class::AffineTransform{T,D,Q1}
    trans_across_class::AffineTransform{T,D,Q2}
    λ::BayesParam{Gamma{T}}
end

function PLDA(T::Type{<:AbstractFloat}; datadim, latentdim_within_class,
              latentdim_across_class, pstrength = 1e-3, W_MAP = false)

    Q1, Q2 = latentdim_within_class, latentdim_across_class

    hprior_w = Normal(zeros(T, Q1), Symmetric(Matrix{T}(I, Q1, Q1)))
    hprior_a = Normal(zeros(T, Q2), Symmetric(Matrix{T}(I, Q2, Q2)))

    trans_within = AffineTransform(T, outputdim = datadim, inputdim = Q1,
                                   pstrength = pstrength, W_MAP = W_MAP)
    trans_across = AffineTransform(T, outputdim = datadim, inputdim = Q2,
                                   pstrength = pstrength, W_MAP = W_MAP)
    λprior = Gamma{T}(pstrength, pstrength)
    λposterior = Gamma{T}(pstrength, pstrength)
    λ = BayesParam(λprior, λposterior)

    PLDA{T,datadim,Q1,Q2}(hprior_w, hprior_a, trans_within, trans_across, λ)
end

function PLDA(;datadim, latentdim_within_class, latentdim_across_class,
              pstrength = 1e-3, W_MAP = false)
    PLDA(Float64; datadim = datadim, latentdim = latentdim,
         pstrength = pstrength, W_MAP = W_MAP)
end

#######################################################################
# Estimate the latent variables

function (m::PLDA{T,D,Q})(X::AbstractVector, z, uposts::Dict) where {T,D,Q}
    # Project the means in the data space
    μ = mean.([uposts[zᵢ] for zᵢ in z]) |> m.trans_across_class

    S₁, S₂ = hstats(m.trans_within_class, X - μ)
    λ̄ = mean(m.λ.posterior)
    S₁, S₂ = λ̄*S₁, λ̄*S₂

    Λ₀ = inv(m.hprior_within_class.Σ)
    Λ₀μ₀ = Λ₀ * m.hprior_within_class.μ
    Σ = Symmetric(inv(Λ₀ + S₂))
    [Normal(Σ * (Λ₀μ₀ + mᵢ), Σ) for mᵢ in S₁]
end

function (m::PLDA{T,D,Q})(X::AbstractVector, hposts) where {T,D,Q}

    # Project the embeddings in the data space
    μ = mean.(hposts) |> m.trans_within_class

    S₁, S₂ = hstats(m.trans_across_class, X - μ)
    λ̄ = mean(m.λ.posterior)
    S₁, S₂ = λ̄*S₁, λ̄*S₂

    Λ₀ = inv(m.hprior_across_class.Σ)
    Λ₀μ₀ = Λ₀ * m..hprior_across_class.μ
    Σ = Symmetric(inv(Λ₀ + S₂))
    [Normal(Σ * (Λ₀μ₀ + mᵢ), Σ) for mᵢ in S₁]
end

(m::PLDA)(t::Tuple) = m(t...)

#######################################################################
# Pretty print

function Base.show(io::IO, ::MIME"text/plain", model::PLDA)
    println(io, typeof(model), ":")
    println(io, "  trans_within_class: $(typeof(model.trans_within_class))")
    println(io, "  trans_across_class: $(typeof(model.trans_across_class))")
    println(io, "  λ: $(typeof(model.λ))")
end

#######################################################################
# Log-likelihood

function _llh_d(::PLDA, x, Tŵ, Tv̂, Tλ, Th, Tu)
    λ, lnλ = Tλ
    h, hhᵀ = Th
    u, uuᵀ = Tu

    # Extract the bias parameter
    wwᵀ = Tŵ[2][1:end-1, 1:end-1]
    w = Tŵ[1][1:end-1]
    μ = Tŵ[1][end]
    μ² = Tŵ[2][end, end]

    vvᵀ = Tv̂[2][1:end-1, 1:end-1]
    v = Tv̂[1][1:end-1]
    a = Tv̂[1][end]
    a² = Tv̂[2][end, end]

    x̄ = dot(w, h) + μ + dot(v, u) + a
    lognorm = (-log(2π) + lnλ - λ*(x^2))/2
    K = λ * (x̄*x - dot(w, h)*μ - dot(v, u)*a)
    K += -λ * (dot(vec(hhᵀ), vec(wwᵀ)) + dot(vec(uuᵀ), vec(vvᵀ)) + μ² + a²)/2
    K += -λ * ((dot(w, h) + μ) * (dot(v, u) + a))

    lognorm + K
end

function _llh(m::PLDA, x, Tŵs, Tv̂s, Tλ, Th, Tu)
    f = (a,b) -> begin
        xᵢ, Tŵᵢ, Tv̂ᵢ = b
        a + _llh_d(m, xᵢ, Tŵᵢ, Tv̂ᵢ, Tλ, Th, Tu)
    end
    foldl(f, zip(x, Tŵs, Tv̂s), init = 0)
end

function loglikelihood(m::PLDA, X, z, uposts)
    hposts = (X, z, uposts) |> m
    _llh.(
        [m],
        X,
        [[gradlognorm(w.posterior, vectorize = false) for w in m.trans_within_class.W]],
        [[gradlognorm(w.posterior, vectorize = false) for w in m.trans_across_class.W]],
        [gradlognorm(m.λ.posterior, vectorize = false)],
        [gradlognorm(p, vectorize = false) for p in hposts],
        [gradlognorm(uposts[zᵢ], vectorize = false) for zᵢ in z]
   ) - kldiv.(hposts, [m.hprior_within_class])
end

loglikelihood(m::PLDA, t::Tuple, uposts) = loglikelihood(m, t..., uposts)


#######################################################################
# Update of the precision parameter λ

function _λstats_d(::PLDA, x::Real, Tŵ, Tv̂, Th, Tu)
    h, hhᵀ = Th
    u, uuᵀ = Tu

    # Extract the bias parameter
    w = Tŵ[1][1:end-1]
    wwᵀ = Tŵ[2][1:end-1, 1:end-1]
    μ = Tŵ[1][end]
    μ² = Tŵ[2][end, end]

    v = Tv̂[1][1:end-1]
    vvᵀ = Tv̂[2][1:end-1, 1:end-1]
    a = Tv̂[1][end]
    a² = Tv̂[2][end, end]

    x̄ = dot(w, h) + μ + dot(v, u) + a
    Tλ₁ = -.5 * x.^2 + (x̄*x - dot(w, h)*μ - dot(v, u)*a)
    Tλ₁ += -(dot(vec(hhᵀ), vec(wwᵀ)) + dot(vec(uuᵀ), vec(vvᵀ)) + μ² + a²)/2
    Tλ₁ += -((dot(w, h) + μ) * (dot(v, u) + a))
    Tλ₂ = 1/2

    vcat(Tλ₁, Tλ₂)
end

function _λstats(m::PLDA, x::AbstractVector, Tŵ, Tv̂, Th, Tu)
    sum(_λstats_d.([m], x, Tŵ, Tv̂, [Th], [Tu]))
end

function λstats(m::PLDA, X, z, uposts, hposts)
    Tŵ = [gradlognorm(w.posterior, vectorize = false) for w in m.trans_within_class.W]
    Tv̂ = [gradlognorm(w.posterior, vectorize = false) for w in m.trans_across_class.W]
    Th = [gradlognorm(p, vectorize = false) for p in hposts]
    Tu = [gradlognorm(uposts[zᵢ], vectorize = false) for zᵢ in z]
    sum(_λstats.([m], X, [Tŵ], [Tv̂], Th, Tu))
end

function update_λ!(m::PLDA, accstats)::Nothing
    η₀ = naturalparam(m.λ.prior)
    update!(m.λ.posterior, η₀ + accstats)
    nothing
end


#######################################################################
# Update of the bases W (within class)

function wstats_within_class(m::PLDA, X, z, uposts, hposts)
    λ, _ = gradlognorm(m.λ.posterior, vectorize = false)
    μ = mean.([uposts[zᵢ] for zᵢ in z]) |> m.trans_across_class
    S₁, S₂ = wstats(m.trans_within_class, X-μ, hposts)
    [λ*S₁, λ*S₂]
end

update_W_within_class!(m::PLDA, accstats) = update_W!(m.trans_within_class, accstats)

#######################################################################
# Update of the bases W (across class)

function wstats_across_class(m::PLDA, X, z, uposts, hposts)
    λ, _ = gradlognorm(m.λ.posterior, vectorize = false)
    μ = mean.(hposts) |> m.trans_within_class
    posts = [uposts[zᵢ] for zᵢ in z]
    S₁, S₂ = wstats(m.trans_across_class, X-μ, posts)
    [λ*S₁, λ*S₂]
end

update_W_across_class!(m::PLDA, accstats) = update_W!(m.trans_across_class, accstats)

#######################################################################
# Update of the mean embeddings

function ustats(m::PLDA, X, z, uposts, hposts) where {T,D,Q}
    # Project the embeddings in the data space
    μ = mean.(hposts) |> m.trans_within_class

    S₁, S₂ = hstats(m.trans_across_class, X - μ)
    λ̄ = mean(m.λ.posterior)
    S₁, S₂ = λ̄*S₁, λ̄*S₂
    ustats = Dict()
    for key in keys(uposts)
        idxs = z .== key
        N = sum(idxs) # Number of points to the cluster `key`
        if N == 0
            [zero(S₁[1]), 0*S₂]
        else
            ustats[key] = [sum(S₁[idxs]), N*S₂]
        end
    end
    ustats
end

function update_u!(m::PLDA, uposts, accstats)::Nothing where {T,D,Q}
    Λ₀ = inv(m.hprior_across_class.Σ)
    Λ₀μ₀ = Λ₀ * m.hprior_across_class.μ
    for key in keys(accstats)
        S₁, S₂ = accstats[key]
        Σ = Symmetric(inv(Λ₀ + S₂))
        uposts[key] = Normal(Σ * (Λ₀μ₀ + S₁), Σ)
    end
    nothing
end

#######################################################################
# Training

"""
    fit!(model, dataloader, uposts[, epochs = 1, callback = x -> x])

Fit a PPCA model to a data set by estimating the variational posteriors
over the parameters.
"""
function fit!(model::PLDA, dataloader, uposts; epochs = 1, callback = x -> x)

    @everywhere dataloader = $dataloader

    for e in 1:epochs
        # Propagate the model to all the workers
        @everywhere model = $model
        @everywhere uposts = $uposts

        ###############################################################
        # Step 1: update the posterior of the bases (within class)
        waccstats_within = @distributed (+) for batch in dataloader
            X, z = batch

            # E-step: estimate the posterior of the embeddings
            hposts = (X, z, uposts) |> model

            # Accumulate statistics for the bases w
            wstats_within_class(model, X, z, uposts, hposts)
        end
        update_W_within_class!(model, waccstats_within)

        # Propagate the model to all the workers
        @everywhere model = $model
        @everywhere uposts = $uposts

        ###############################################################
        # Step 2: update the class mean embeddings
        reducer = (a,b) -> merge(+,a,b)
        uaccstats = @distributed reducer for batch in dataloader
            X, z = batch

            # E-step: estimate the posterior of the embeddings
            hposts = (X, z, uposts) |> model

            # Accumulate statistics for the bases w
            ustats(model, X, z, uposts, hposts)
        end
        update_u!(model, uposts, uaccstats)

        # Propagate the model to all the workers
        @everywhere model = $model
        @everywhere uposts = $uposts

        ###############################################################
        # Step 3: update the posterior of the bases (across class)
        waccstats_across = @distributed (+) for batch in dataloader
            X, z = batch

            # E-step: estimate the posterior of the embeddings
            hposts = (X, z, uposts) |> model

            # Accumulate statistics for the bases w
            wstats_across_class(model, X, z, uposts, hposts)
        end
        update_W_across_class!(model, waccstats_across)

        # Propagate the model to all the workers
        @everywhere model = $model
        @everywhere uposts = $uposts

        ###############################################################
        # Step 4: update the posterior of the precision λ
        λaccstats = @distributed (+) for batch in dataloader
            X, z = batch

            # E-step: estimate the posterior of the embeddings
            hposts = (X, z, uposts) |> model

            # Accumulate statistics for λ
            λstats(model, X, z, uposts, hposts)
        end
        update_λ!(model, λaccstats)

        # Notify the caller
        callback(e)
    end
end

