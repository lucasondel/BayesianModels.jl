# PPCA - Implementation of the Variational Bayes Inference for the
# PCCA model
#
# Lucas Ondel, 2020

"""
    fit!(model, dataloader, [, epochs = 1, callback = x -> x])

Fit a PPCA model to a data set by estimating the variational posteriors
over the parameters.
"""
function fit!(model::PPCAModel, dataloader, epochs, callback)

    @everywhere dataloader = $dataloader

    # NOTE: By 1 epoch we mean TWO passes over the data, one pass to
    # update the bases and the other to update the precision parameter

    for e in 1:epochs
        # Propagate the model to all the workers
        @everywhere model = $model

        ###############################################################
        # Step 1: update the posterior of the bases
        waccstats = @distributed (+) for X in dataloader
            # E-step: estimate the posterior of the embeddings
            hposts = X |> model

            # Accumulate statistics for the bases w
            wstats(model, X, hposts)
        end
        update_W!(model, waccstats)

        # Propagate the model to all the workers
        @everywhere model = $model

        ###############################################################
        # Step 2: update the posterior of the precision λ
        λaccstats = @distributed (+) for X in dataloader
            # E-step: estimate the posterior of the embeddings
            hposts = X |> model

            # Accumulate statistics for λ
            λstats(model, X, hposts)
        end

        # M-step 2: update the posterior of the precision parameter λ
        update_λ!(model, λaccstats)

        # M-step 3: update the posterior of the precision parameter λ
        update_α!(model, αstats(model))

        # Notify the caller
        callback(e)
    end
end

