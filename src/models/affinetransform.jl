# Probabilistic Affine Transform, elementary component of PPCA,
# PLDA,...
#
# Lucas Ondel 2021

#######################################################################
# Model definition

"""
    struct AffineTransform{D,Q}
        W
        b
    end

Affine transform.
"""
struct AffineTransform{D,Q} <: BMObject
    W::BayesParamList
    b::T where T<:AbstractBayesParam
end


function (trans::AffineTransform)(X::AbstractVector)
    bases = mean.([w.posterior for w in trans.W])
    _affine_transform.([bases], X)
end

