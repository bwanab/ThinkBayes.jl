module ThinkStats

export CatDist, pmf_from_seq, mult_likelihood, max_prob, pdf, maximum, getindex, probs

using Distributions
using DataFrames

struct CatDist
    values::Vector
    dist::Categorical
end

Base.getindex(d::CatDist, prob) = pdf(d, prob)

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

function pmf_from_seq(seq)::CatDist
    df=DataFrame(a=seq)
    len=length(seq)
    g=groupby(df, :a)
    d=[(first(x).a, nrow(x)/len) for x in g]
    CatDist([x[1] for x in d], Categorical([x[2] for x in d]))
end

function pmf_new_probs(d::CatDist, probs)::CatDist
    CatDist(d.values, Categorical(probs))
end

function mult_likelihood(d::CatDist, likelihood)::CatDist
    pmf_new_probs(d, normalize(probs(d).*likelihood))
end

function normalize(probs)
    probs./sum(probs)
end

end
