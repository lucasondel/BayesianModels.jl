# PPCA

Julia package for Probabilistic Principal Components Analysis as
described in ["Variational Principal Components Analysis"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-vpca-icann-99.pdf).

## Usage

You first need to create a `PPCAModel`:
```julia
julia> model = PPCAModel(datadim = 2, latentdim = 2)
PPCAModelHP{Float64,2,2}:
  αprior:
    ExpFamilyDistributions.Gamma{Float64}:
      α = 0.001
      β = 0.001
  wprior:
    ExpFamilyDistributions.Normal{Float64,3}:
      μ = [0.0, 0.0, 0.0]
      Σ = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
  hprior:
    ExpFamilyDistributions.Normal{Float64,2}:
      μ = [0.0, 0.0]
      Σ = [1.0 0.0; 0.0 1.0]
  λprior:
    ExpFamilyDistributions.Gamma{Float64}:
      α = 0.001
      β = 0.001
```
In practice, you should set `latentdim` between `1` and `datadim - 1`
(included). If you set, as in the example above, `latentdim` greater or
equal to `datadim`, the model will automatically shrink the extra
bases to zero during training (see illustration).

Then, you need to initialize the variational posteriors. This is easily
done by:
```julia
julia> θposts = θposteriors(model)
Dict{Symbol,Any} with 3 entries:
  :α => ExpFamilyDistributions.Gamma{Float64}[Gamma{Float64}:…
  :w => ExpFamilyDistributions.Normal{Float64,2}[Normal{Float64,2}:…
  :λ => Gamma{Float64}:…
```

For training, you need to provide a data loader from the
[BasicDataLoaders](https://github.com/lucasondel/BasicDataLoaders)
package to access the data, and simply call `fit!`:
```julia
julia> dl = ... # Data loader
julia> fit!(model, dl, θposts, epochs = 10)
```
The training will be run in parallel if there are several workers
available. To add workers, see
[`addprocs`](https://docs.julialang.org/en/v1/stdlib/Distributed/).

Finally, to extract the latent posteriors:
```julia
julia> X = dl[1] # Use the 1st batch of the data loader
julia> hposteriors(model, X, θposts)
10-element Array{Normal{Float64,2},1}:
 Normal{Float64,2}:
  μ = [1.7512436452299205, -9.956494453759758e-8]
  Σ = [0.028248509216655084 5.1464934731583015e-8; 5.1464934731583015e-8 0.9657413079963144]
 Normal{Float64,2}:
  μ = [-1.7016493202424885, 9.674530645698072e-8]
  Σ = [0.028248509216655084 5.1464934731583015e-8; 5.1464934731583015e-8 0.9657413079963144]
...
```

For a complete example, have a look at the [example jupyter notebook](https://github.com/BUTSpeechFIT/PPCA/blob/master/examples/PPCA.ipynb).
![Alt Text](https://github.com/BUTSpeechFIT/PPCA/blob/master/examples/demo.gif)
