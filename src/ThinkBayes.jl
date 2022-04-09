module ThinkBayes

export CatDist, pmf_from_seq, mult_likelihood, max_prob, min_prob, prob_ge, prob_le, 
    binom_pmf, normalize, add_dist, sub_dist, mult_dist, make_binomial, loc,
    update_binomial
# from Base:
export getindex, copy, values, show, (*), (==)
# from Distributions:
export probs, pdf, cdf, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, modes, mode, skewness, kurtosis, entropy, mgf, cf
# from Plot:
export plot, plot!

import Plots: plot, plot!, bar


import Distributions
import Distributions:  probs, pdf, cdf, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, modes, mode, skewness, kurtosis, entropy, mgf, cf

import Base: copy, getindex, values, show, (*), (==)

using DataFrames
using Interpolations

struct CatDist
    values::Vector
    dist::Distributions.Categorical
end

# Base:

getindex(d::CatDist, prob) = pdf(d, prob)
copy(d::CatDist) = CatDist(copy(d.values), Distributions.Categorical(copy(probs(d))))
values(d::CatDist) = d.values
function show(io::IO, d::CatDist)
    a=DataFrame(a=values(d), b=probs(d))
    show(a)
end
(*)(d::CatDist, likelihood) = mult_likelihood(d, likelihood)
(==)(x::CatDist, y::CatDist) = (x.values == y.values) && (probs(x) == probs(y))


# Plots
nplot=1
function plot(d::CatDist; xaxis="xs", yaxis="ys", label="y1", plot_title="plot")
    global nplot=1
    plot(values(d), probs(d), xaxis=xaxis, yaxis=yaxis, label=label, plot_title=plot_title)
end

function plot!(d::CatDist; label=nothing)
    global nplot += 1
    if label===nothing
        label="y"*string(ThinkBayes.nplot)
    end
    plot!(values(d), probs(d), label=label)
end

function bar(d::CatDist; xaxis=("xs"), yaxis=("ys"), label="y1", plot_title="bar plot")
    bar(values(d), probs(d), xaxis=xaxis, yaxis=yaxis, label=label, plot_title=plot_title)
end

# DataFrame

function loc(df, val)
    df[findfirst(==(val), df.Index), :]
end

# Distributions

function probs(d::CatDist)
    probs(d.dist)
end

findindex(d::CatDist, x) = findfirst(isequal(x), d.values)

function pdf(d::CatDist, x)
    index = findindex(d, x)
    if index === nothing
        return 0
    end
    pdf(d.dist, index)
end

function logpdf(d::CatDist, x)
    index = findindex(d, x)
    if index === nothing
        return 0
    end
    logpdf(d.dist, index)
end

function cdf(d::CatDist, x)
    index = findindex(d, x)
    if index === nothing
        return 0.0
    end
    cdf(d.dist, index)
end

maximum(d::CatDist) = d.values[maximum(d.dist)]
minimum(d::CatDist) = d.values[minimum(d.dist)]
rand(d::CatDist) = d.values[rand(d.dist)]
sampler(d::CatDist) = sampler(d.dist)
quantile(d::CatDist, r) = d.values[quantile(d.dist, r)]
insupport(d::CatDist, r) = insupport(d.dist, r)
mean(d::CatDist) = mean(d.dist)
var(d::CatDist) = var(d.dist)
mode(d::CatDist) = d.values[mode(d.dist)]
modes(d::CatDist) = [d.values[x] for x in modes(d.dist)]
skewness(d::CatDist) = skewness(d.dist)
kurtosis(d::CatDist) = kurtosis(d.dist)
entropy(d::CatDist) = entropy(d.dist)
entropy(d::CatDist, r::Real) = entropy(d.dist, r)
mgf(d::CatDist, r) = mgf(d.dist, r)
cf(d::CatDist, r) = cf(d.dist, r)




# CatDist:

function max_prob(d::CatDist)
    (mp, index)=findmax(identity, (probs(d)))
    d.values[index]
end

function min_prob(d::CatDist)
    (mp, index)=findmin(identity, (probs(d)))
    d.values[index]
end


function pmf_from_seq(seq; counts=nothing)::CatDist
    if counts!==nothing
        a = [fill(x,y) for (x,y) in zip(seq, counts)]
        seq=[(a...)...]
    end
    df=DataFrame(a=seq)
    len=length(seq)
    g=groupby(df, :a)
    d=[(first(x).a, nrow(x)/len) for x in g]
    CatDist([x[1] for x in d], Distributions.Categorical([x[2] for x in d]))
end

function pmf_from_seq(seq, probs::Array{Float64})::CatDist
    CatDist(seq, Distributions.Categorical(probs))
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

function prob_ge(d::CatDist, threshold)
    sum(probs(d)[values(d).>=threshold])
end

function prob_le(d::CatDist, threshold)
    sum(probs(d)[values(d).<=threshold])
end


function binom_pmf(k::Number, n::Number, ps::AbstractVector)
    [pdf(Distributions.Binomial(n, p), k) for p in ps]
end

function binom_pmf(k::Number, ns::AbstractVector, p::Number)
    [pdf(Distributions.Binomial(n, p), k) for n in ns]
end

function make_binomial(n, p)
    """Make a binomial Pmf."""
    binom=Distributions.Binomial(n, p)
    ks=[pdf(binom, x) for x in 0:n]
    pmf_from_seq(0:n, ks)
end

function update_binomial(pmf::CatDist, data)
    (k, n) = data
    xs = values(pmf)
    likelihood=binom_pmf(k, n, xs)
    pmf*=likelihood
end


function convolve(p1::CatDist, p2::CatDist, func)
    d = [(func(q1, q2), (p1 * p2)) 
          for (q1, p1) in zip(values(p1), probs(p1))
                for (q2, p2) in zip(values(p2), probs(p2))]
    df = DataFrame(qs=[q for (q, p) in d], ps=[p for (q, p) in d])
    g = groupby(df, :qs)
    d = [(first(x).qs, sum(x.ps)) for x in g]
    CatDist([x[1] for x in d], Distributions.Categorical([x[2] for x in d]))
end

function add_dist(p1::CatDist, p2::CatDist)
    convolve(p1, p2, +)
end

function add_dist(p1::CatDist, n::Number)
    pmf_from_seq(values(p1).+n, probs(p1))
end

function sub_dist(p1::CatDist, p2::CatDist)
    convolve(p1, p2, -)
end

function sub_dist(p1::CatDist, n::Number)
    pmf_from_seq(values(p1).-n, probs(p1))
end

function mult_dist(p1::CatDist, p2::CatDist)
    convolve(p1, p2, *)
end

function mult_dist(p1::CatDist, n::Number)
    pmf_from_seq(values(p1).*n, probs(p1))
end

# cmf
export CMF, cmf, cmfs

struct CMF
    d:: DataFrame
    interp::Any
end

function cmf(pmf::CatDist)
    cmfs = [cdf(pmf, x) for x in values(pmf)]
    interp = LinearInterpolation(Interpolations.deduplicate_knots!(cmfs), values(pmf))
    CMF(DataFrame(Index=values(pmf), cmf=cmfs), interp)
end

function values(cmf::CMF) 
    cmf.d.Index
end

function cmfs(cmf::CMF)
    cmf.d.cmf
end

function cmf(d::CMF, x)
    loc(d.d, x).cmf
end

function quantile(d::CMF, x)
    d.interp(x)
end

function plot(d::CMF; xaxis="xs", yaxis="ys", label="y1", plot_title="plot")
    global nplot=1
    plot(values(d), cmfs(d), xaxis=xaxis, yaxis=yaxis, label=label, plot_title=plot_title)
end

function plot!(d::CMF; label=nothing)
    global nplot += 1
    if label===nothing
        label="y"*string(ThinkBayes.nplot)
    end
    plot!(values(d), cmfs(d), label=label)
end

end
