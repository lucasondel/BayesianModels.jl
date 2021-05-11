# SPDX-License-Identifier: MIT

abstract type BMObject end

const BMObjectList{N,T<:BMObject} = NTuple{N,T}

function iscomposite(obj::T) where T<:BMObject
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        if typeof(prop) <: BMObject
            return true
        elseif typeof(prop) <: BMObjectList
            return true
        end
    end
    return false
end

function Base.show(io::IO, mime::MIME"text/plain", obj::BMObject)
    indent = get(io, :indent, 0)
    prefix = get(io, :prefix, "")
    parents = get(io, :parents, [])

    if ! iscomposite(obj)
        println(io, " "^indent, prefix, obj)
        return
    end

    println(io, " "^indent, prefix, obj, " (")
    for name in fieldnames(typeof(obj))
        prop = getproperty(obj, name)
        if typeof(prop) <: BMObject
            io2 = IOContext(io, :indent => indent+2, :prefix => "($(name)): ")
            show(io2, mime, prop)
        elseif typeof(prop) <: BMObjectList
            println(io, " "^(indent+2), "($name): [")
            for (i, param) in enumerate(prop)
                io2 = IOContext(io, :indent => indent+4, :prefix => "($(i)): ")
                show(io2, mime, param)
            end
            println(io, " "^(indent+2), "]")
        end
    end
    println(io, " "^indent, ")")
end
