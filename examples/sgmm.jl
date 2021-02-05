
using Plots
using BasicDataLoaders

using Revise
using BayesianModels

#######################################################################
# Config

D = 2           # Dimension of the data.
K = 5           # Number of Gaussian components in the model.
epochs = 1000     # Number of epochs.
lrate = 0.1     # Learning rate.

#######################################################################

X = collect(eachcol(randn(2, 100) .+ 1))

normals = [NormalDiag(datadim = D, init_noise_std = 0.1) for k in 1:K]
model = Mixture(components = normals)

elbos = [elbo(model, X)]
@info "epoch = 0 𝓛 = $(elbos[end])"

for epoch in 1:epochs
    gradstep(∇elbo(model, X, stats_scale = 1), lrate = lrate)
    push!(elbos, elbo(model, X))
    @info "epoch = $epoch 𝓛 = $(elbos[end])"
end

plot(elbos, label = "ELBO")

