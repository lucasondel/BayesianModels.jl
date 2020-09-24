# Variational Bayes Inference for the PCCA model
#

#######################################################################
# inference for the PCCA model

function pgaugmentation(
    model::PCCAModel{D,K,Q},
    hposterior::Array{<:Normal}
) where {D,K,Q}

    # Expectation of the logistic regression bases.
    B = hcat([model.Bposterior[k].μ for k in 1:K-1]...)
    vec_BBᵀ = hcat([vec(model.Bposterior[k].Σ + β * β')
                    for (k,β) in enumerate(eachcol(B))]...)

    # Expectation correlation factors
    h = hcat([post.μ for post in hposterior]...)
    T = eltype(h)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = Matrix{T}(I, Q+1, Q+1) * 0
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        vec(Σ)
    end for post in hposterior]...)

    # Compute the expected sigmoid parameter squared
    ψ² = vec_BBᵀ' * ĥĥᵀ

    E_ω, KL = pgaugmentation(ψ²)
end

function loglikelihood(
    model::PCCAModel{D,K,Q},
    hposterior::Array{<:Normal},
    E_ω::Matrix{T},
    X::Matrix{T},
    z::Vector{<:Real}
) where {D,K,K_1,Q,T}

    N = size(X, 2)

    # Expectation correlation factors
    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    # Expectation of the linear regression bases
    w_Σs = [model.Wposterior[d].Σ for d in 1:D]
    W = hcat([model.Wposterior[d].μ for d in 1:D]...)
    WWᵀ = sum(w_Σs) + W * W'

    # Expectation of the logistic regress bases
    B = hcat([model.Bposterior[k].μ for k in 1:K-1]...)
    vec_BBᵀ = hcat([vec(model.Bposterior[k].Σ + β * β')
                    for (k, β) in enumerate(eachcol(B))]...)

    # Sufficient statistics for z₁, z₂, ...
    onehot = zeros(T, K-1, N)
    up_to = zeros(K-1, N)
    for i in 1:K-1 onehot[i, z .== i] .= 1 end
    for i in 1:K-1 up_to[i, z .> i] .= 1 end

    # Expectation λ
    λ = mean(model.λposterior)[1]

    # Compute the 1st natural parameters of h's posterior
    s1 = λ * W * X #.+ .5 * (B * (onehot .- up_to))

    # Compute the 2nd natural parameters of h's posterior
    E_BBᵀ = (vec_BBᵀ * ((onehot .+ up_to) .* E_ω))
    s2 = -.5 * (λ * reshape(WWᵀ, :, 1) .+ 0 * E_BBᵀ)

    lnλ = gradlognorm(model.λposterior)[2]
    llh = sum(eachrow(s1 .* ĥ)) + sum(eachrow(s2 .* ĥĥᵀ)) .+ .5 * D * lnλ
    llh += -.5 * λ * sum(eachrow(X .* X)) .- .5 * D * log(2π)
end

function inferh(
    model::PCCAModel{D,K,Q},
    E_ω::Matrix{T},
    X::Matrix{T},
    z::Vector{<:Real}
) where {D,K,K_1,Q,T}

    N = size(X, 2)

    # Expectation of the linear regression bases
    w_Σs = [model.Wposterior[d].Σ[1:end-1, 1:end-1] for d in 1:D]
    W = hcat([model.Wposterior[d].μ[1:end-1] for d in 1:D]...)
    WWᵀ = sum(w_Σs) + W * W'

    # Expectation of the logistic regress bases
    B = hcat([model.Bposterior[k].μ[1:end-1] for k in 1:K-1]...)
    vec_BBᵀ = hcat([vec(model.Bposterior[k].Σ[1:end-1, 1:end-1] + β * β')
                    for (k, β) in enumerate(eachcol(B))]...)

    # Sufficient statistics for z₁, z₂, ...
    onehot = zeros(T, K-1, N)
    up_to = zeros(K-1, N)
    for i in 1:K-1 onehot[i, z .== i] .= 1 end
    for i in 1:K-1 up_to[i, z .> i] .= 1 end

    # Expectation λ
    λ = mean(model.λposterior)[1]

    # Compute the 1st natural parameters of h's posterior
    Λ₀ = inv(model.hprior.Σ)
    Λ₀μ₀ = Λ₀ * model.hprior.μ
    Λμs = Λ₀μ₀ .+ λ * W * X #.+ .5 * (B * onehot - B * up_to)

    # Compute the 2nd natural parameters of h's posterior
    E_BBᵀ = vec_BBᵀ * ((onehot .+ up_to) .* E_ω)
    Λs = Λ₀ .+ λ * reshape(WWᵀ, Q, Q, 1) .+ 0 * reshape(E_BBᵀ, Q, Q, :)
    Σs = [inv(Λs[:, :, n]) for n in 1:N]
    μs = [Σs[n] * c for (n, c) in enumerate(eachcol(Λμs))];

    [Normal(μs[n], Σs[n]) for n in 1:N]
end

function stats_W(
    model::PCCAModel{D,K,Q},
    hposterior::Array{<:Normal},
    X::Matrix{T}
) where {D,K,Q,T}
    λ = mean(model.λposterior)[1]

    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    s1 = λ * ĥ * X'
    s2 = -.5 * λ * reduce(+, eachcol(ĥĥᵀ))
    vcat(s1, repeat(s2', size(s1, 2))')
end

function stats_λ(
    model::PCCAModel{D,K,Q},
    hposterior::Array{<:Normal},
    X::Matrix{T}
) where {D,K,Q,T}
    N = size(X, 2)

    # Expectation H
    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    # Expectation W
    w_Σs = [model.Wposterior[d].Σ for d in 1:D]
    W = hcat([model.Wposterior[d].μ for d in 1:D]...)
    vec_WWᵀ = vec(sum(w_Σs) + W * W')

    # Return the statistic vector
    s1 = -.5 * sum(X .* X) + sum(W' * ĥ .* X) .- .5 .* sum(ĥĥᵀ .* vec_WWᵀ)
    s2 = N * D / 2
    vcat([s1], [s2])
end

function stats_B(
    model::PCCAModel{D,K,Q},
    hposterior::Array{<:Normal},
    E_ω::Matrix{T},
    z::Vector{<:Real}
) where {D,K,K_1,Q,T}

    N = length(z)

    # Expectation H
    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    # Sufficient statistics for z₁, z₂, ...
    onehot = zeros(T, K-1, N)
    up_to = zeros(K-1, N)
    for i in 1:K-1 onehot[i, z .== i] .= 1 end
    for i in 1:K-1 up_to[i, z .> i] .= 1 end

    s1 = .5 * (onehot .- up_to) * ĥ'
    s2 = (E_ω .* (onehot .+ up_to)) * ĥĥᵀ'
    hcat(s1, s2)
end

#######################################################################
# inference for the PPCA model

function loglikelihood(
    model::PPCAModel{D,Q},
    hposterior::Array{<:Normal},
    X::Matrix{T},
) where {D,Q,T}

    N = size(X, 2)

    # Expectation correlation factors
    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    # Expectation of the linear regression bases
    w_Σs = [model.Wposterior[d].Σ for d in 1:D]
    W = hcat([model.Wposterior[d].μ for d in 1:D]...)
    WWᵀ = sum(w_Σs) + W * W'

    # Expectation λ
    η_λ = gradlognorm(model.λposterior)

    # Compute the 1st natural parameters of h's posterior
    s1 = W * X

    # Compute the 2nd natural parameters of h's posterior
    s2 = -.5 * reshape(WWᵀ, :)

    # Sufficient statistics for λ
    stats = vcat(-.5 * sum(eachrow(X .* X))' + sum(eachrow(s1 .* ĥ))' + s2' * ĥĥᵀ,
                 .5 * D * ones(T, 1, N))

    return stats' * η_λ .- .5 * D * log(2π)
end

function inferh(
    model::PPCAModel{D,Q},
    X::Matrix{T},
) where {D,Q,T}

    N = size(X, 2)

    # Expectation of the linear regression bases
    w_Σs = [model.Wposterior[d].Σ[1:end-1, 1:end-1] for d in 1:D]
    W = hcat([model.Wposterior[d].μ for d in 1:D]...)
    μ = W[end, :]
    W = W[1:end-1, :]
    WWᵀ = sum(w_Σs) + W * W'

    # Expectation λ
    λ = mean(model.λposterior)[1]

    # Compute the 1st natural parameters of h's posterior
    Λ₀ = inv(model.hprior.Σ)
    Λ₀μ₀ = Λ₀ * model.hprior.μ
    Λμs = Λ₀μ₀ .+ λ * W * (X .-  μ)

    # Compute the 2nd natural parameters of h's posterior
    Λ = Λ₀ .+ λ * WWᵀ
    Σ = inv(Λ)
    μs = [Σ * μ for (n, μ) in enumerate(eachcol(Λμs))];

    [Normal(μs[n], Σ) for n in 1:N]
end

function stats_λ(
    model::PPCAModel{D,Q},
    hposterior::Array{<:Normal},
    X::Matrix{T}
) where {D,Q,T}
    N = size(X, 2)

    # Expectation correlation factors
    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)

    # Expectation of the linear regression bases
    w_Σs = [model.Wposterior[d].Σ for d in 1:D]
    W = hcat([model.Wposterior[d].μ for d in 1:D]...)
    WWᵀ = sum(w_Σs) + W * W'

    # Expectation λ
    η_λ = gradlognorm(model.λposterior)

    # Compute the 1st natural parameters of h's posterior
    s1 = W * X

    # Compute the 2nd natural parameters of h's posterior
    s2 = -.5 * reshape(WWᵀ, :)

    # Sufficient statistics for λ
    stats = vcat(-.5 * sum(eachrow(X .* X))' + sum(eachrow(s1 .* ĥ))' + s2' * ĥĥᵀ,
                 .5 * D * ones(T, 1, N))

    sum(eachcol(stats))
end

function stats_W(
    model::PPCAModel{D,Q},
    hposterior::Array{<:Normal},
    X::Matrix{T}
) where {D,Q,T}
    λ = mean(model.λposterior)[1]

    h = hcat([post.μ for post in hposterior]...)
    ĥ = vcat(h, ones(T, 1, size(h, 2)))
    ĥĥᵀ = hcat([begin
        Σ = zeros(T, Q+1, Q+1)
        Σ[1:end-1, 1:end-1] = post.Σ + post.μ * post.μ'
        Σ[end, 1:end-1] = post.μ
        Σ[1:end-1, end] = post.μ
        Σ[end, end] = 1
        vec(Σ)
    end for post in hposterior]...)
    #return ĥĥᵀ

    s1 = sum([λ * ĥₙ * xₙ' for (ĥₙ, xₙ) in zip(eachcol(ĥ), eachcol(X))])
    s2 = sum(eachcol(-.5 * λ * ĥĥᵀ))
    return vcat(s1, repeat(s2', size(s1, 2))')
    #vcat(s1, s2)
    #return s1, s2
    #s1, s2, ĥ, ĥĥᵀ
    #vcat(s1, repeat(s2', size(s1, 2))')
end

