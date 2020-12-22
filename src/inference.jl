# PPCA - Implementation of the Variational Bayes Inference for the
# PCCA model
#
# Lucas Ondel, 2020

function _fit!(model::AbstractPPCAModel, dataloader, θposts, epochs, callback)

    @everywhere dataloader = $dataloader

    # NOTE: By 1 epoch we mean TWO passes over the data, one pass to
    # update the bases and the other to update the precision parameter

    for e in 1:epochs
        # Propagate the model and the posteriors to all the workers
        @everywhere θposts = $θposts
        @everywhere model = $model

        ###############################################################
        # Step 1: update the posterior of the bases
        waccstats = @distributed (+) for X in dataloader
            # E-step: estimate the posterior of the embeddings
            hposts = hposteriors(model, X, θposts)

            # Accumulate statistics for the bases w
            wstats(model, X, θposts, hposts)
        end
        wposteriors!(model, θposts, waccstats)

        # Propagate the update of the posterior to all the workers
        @everywhere θposts = $θposts

        ###############################################################
        # Step 2: update the posterior of the precision λ
        λaccstats = @distributed (+) for X in dataloader
            # E-step: estimate the posterior of the embeddings
            hposts = hposteriors(model, X, θposts)

            # Accumulate statistics for the bases w
            λstats(model, X, θposts, hposts)
        end

        # M-step 2: update the posterior of the precision parameter λ
        λposterior!(model, θposts, λaccstats)

        # Notify the caller
        callback(e)
    end
end

function fit!(model::PPCAModel, dataloader, θposts; epochs = 1, callback = x -> x)
    _fit!(model, dataloader, θposts, epochs, callback)
end

function fit!(model::PPCAModelHP, dataloader, θposts; epochs = 1, callback = x -> x)
    function modified_callback(e)
        αposteriors!(model, θposts, αstats(model, θposts))
        callback(e)
    end
    _fit!(model, dataloader, θposts, epochs, modified_callback)
end

"""
    fit!(model, dataloader, θposts[, epochs = 1, callback = x -> x])

Fit a PPCA model to a data set by estimating the variational posteriors
over the parameters.
"""
fit!

