
using Random
using Plots
using BasicDataLoaders

using Revise
using BayesianModels

#######################################################################
# Config

D = 2           # Dimension of the data.
K = 5           # Number of Gaussian components in the model.
epochs = 1000   # Number of epochs.
lrate = 0.1     # Learning rate.
batchsize = 10  # Number of samples per batch
lograte = 5     # Output a log message every `lograte` udpate

#######################################################################

X = collect(eachcol(randn(2, 100) .+ 1))

normals = [NormalDiag(datadim = D, init_noise_std = 0.1) for k in 1:K]
model = Mixture(components = normals)

elbos = [elbo(model, X)]
@info "epoch = 0 𝓛 = $(elbos[end])"

n_updates = 0
for epoch in 1:epochs
    global n_updates

    dl = DataLoader(shuffle(X), batchsize = batchsize)
    for (i, Xᵢ) in enumerate(dl)
        gradstep(∇elbo(model, Xᵢ, stats_scale = 1), lrate = lrate)
        n_updates += 1

        if n_updates % lograte == 0
            push!(elbos, elbo(model, X))
            @info "epoch = $epoch batch = $(i)/$(length(dl)) 𝓛 = $(elbos[end])"
        end
    end
end

plot(elbos[2:end], label = "ELBO")

