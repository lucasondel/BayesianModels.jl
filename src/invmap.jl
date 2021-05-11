# SPDX-License-Identifier: MIT

# Invertible function
#
# Lucas Ondel 2021

struct InvertibleMap <: Function
    f::Function
    f_inv::Function
end

function (imap::InvertibleMap)(x)
    imap.f(x)
end

Base.literal_pow(::typeof(^), imap::InvertibleMap, ::Val{-1}) = imap.f_inv

