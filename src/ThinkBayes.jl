module ThinkBayes

export CatDist, pmf_from_seq, pmf_with_probs, mult_likelihood, max_prob, pdf, maximum, getindex, probs

import Distributions, Distributions.probs
import Base.copy, Base.getindex, Base.values
using DataFrames

struct CatDist
    values::Vector
    dist::Distributions.Categorical
end

getindex(d::CatDist, prob) = pdf(d, prob)
copy(d::CatDist) = CatDist(copy(d.values), Distributions.Categorical(copy(probs(d))))
values(d::CatDist) = d.values

function probs(d::CatDist)
    Distributions.probs(d.dist)
end

function pdf(d::CatDist, prob)
    index=findfirst(isequal(prob), d.values)
    if index==nothing
        return 0
    end
    Distributions.probs(d.dist)[index]
end

function maximum(d::CatDist)
    Distributions.maximum(d.dist)
end

function max_prob(d::CatDist)
    (mp, index)=findmax(identity, (probs(d)))
    d.values[index]
end

function pmf_with_probs(seq, probs)::CatDist
    a = [fill(x,y) for (x,y) in zip(seq, probs)]
    pmf_from_seq([(a...)...])
end

function pmf_from_seq(seq)::CatDist
    df=DataFrame(a=seq)
    len=length(seq)
    g=groupby(df, :a)
    d=[(first(x).a, nrow(x)/len) for x in g]
    CatDist([x[1] for x in d], Distributions.Categorical([x[2] for x in d]))
end

function pmf_new_probs(d::CatDist, probs)::CatDist
    CatDist(d.values, Distributions.Categorical(probs))
end

function mult_likelihood(d::CatDist, likelihood)::CatDist
    pmf_new_probs(d, normalize(probs(d).*likelihood))
end

function normalize(probs)
    probs./sum(probs)
end

end
