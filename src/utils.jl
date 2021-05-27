# SPDX-License-Identifier: MIT

macro cache(cache_dict::Symbol, expr::Expr)
    @assert expr.head == :(=) "can only cache assignment expression"
    sym = QuoteNode(expr.args[1])
    return quote
        local retval = $(esc(expr))
        $(esc(cache_dict))[$sym] = retval
        retval
    end
end


function logsumexp(x; dims=:)
    xmax = maximum(x, dims=dims)
    xmax + log.(sum(exp.(x .- xmax), dims=dims))
end

function init_gpu()
    device_reset!()

    for device in devices()
        try
            device!(device)
            @goto device_selected
        catch e
            @debug "$device is busy, trying the next one"
        end
    end
    if isnothing(selected_device)
        throw(ErrorException("could not set a working device, all devices appear to be busy"))
    end

    @label device_selected
    @info "using cuda device $(device())"
    return nothing
end

function _reallocate!(obj, allocator, visited = Set())
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        T = typeof(prop)
        if T <: EFD.AbstractParameter
            param = EFD.reallocate(prop, allocator)
            setproperty!(obj, name, param)
        elseif T <: BayesianParameter
            push!(visited, prop.prior)
            push!(visited, prop.posterior)
            newprior = _reallocate!(prop.prior, allocator, visited)
            newposterior = _reallocate!(prop.posterior, allocator, visited)
            setproperty!(obj, name, BayesianParameter(newprior, newposterior))
        else prop ∉ visited
            push!(visited, prop)
            _reallocate!(prop, allocator, visited)
        end
    end
    obj
end

gpu!(obj) = _reallocate!(obj, CuArray, Set())
cpu!(obj) = _reallocate!(obj, Array, Set())
