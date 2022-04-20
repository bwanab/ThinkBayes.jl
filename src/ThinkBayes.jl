module ThinkBayes

export CatDist, pmf_from_seq, mult_likelihood, max_prob, min_prob, 
    prob_ge, prob_le, prob_gt, prob_lt, prob_eq,
    binom_pmf, normalize, add_dist, sub_dist, mult_dist, make_binomial, loc,
    update_binomial, credible_interval, make_pmf, make_df_from_seq_pmf, 
    make_mixture, make_poisson_pmf, update_poisson, make_exponential_pmf, make_gamma_pmf,
    expo_pdf
# from Base:
export getindex, copy, values, show, (+), (*), (==), (^), (-), (/), isapprox
# from Distributions:
export probs, pdf, cdf, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, std, modes, mode, skewness, kurtosis, entropy, mgf, cf
# from Plot:
export plot, plot!

import Plots: plot, plot!, bar


import Distributions
import Distributions:  probs, pdf, cdf, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, modes, mode, skewness, kurtosis, entropy, mgf, cf, std

import Base: copy, getindex, values, show, (+), (*), (==), (^), (-), (/), isapprox

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
    a=DataFrame(Values=values(d), Probs=probs(d))
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
    idx = findfirst(==(val), df.Index)
    if idx === nothing
        return nothing
    end
   df[idx, :]
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
#TODO: fix mean should be the interpolation of the value at the mean, not the index into the distribution.
#      like mode, but interpolated instead of the actual value.
# Note: I decided to take the simpler route of implementing it like in
#       empiricaldist.py from Allen Downey's ThinkBayes
mean(d::CatDist) = sum(values(d) .* probs(d))
function var(d::CatDist)
    m = mean(d)
    dd = values(d) .- m
    sum(dd .^ 2 .* probs(d))
end
std(d::CatDist) = var(d) ^ 0.5
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

function make_poisson_pmf(lamda, vals)
    dist = Distributions.Poisson(lamda)
    ps = normalize([pdf(dist, v) for v in vals])
    pmf_from_seq(vals, ps)
end

function make_exponential_pmf(lambda::Float64, vals::Vector{Float64})
    λ = 0.000001
    if lambda > 0
        λ = lambda
    end
    e = Distributions.Exponential(1/λ)
    pmf_from_seq(vals, normalize([pdf(e, v) for v in vals]))
end

function make_exponential_pmf(lambda::Float64, high::Number)
    qs = LinRange(0, high, 101)
    make_exponential_pmf(lambda, [q for q in qs])
end

function  expo_pdf(lambdas::Vector{Float64}, val::Float64)
    [pdf(Distributions.Exponential(1/lambda), val) for lambda in lambdas]
end

function update_poisson(p::CatDist, data)
    k = data
    lambdas = values(p)
    likelihood = [pdf(Distributions.Poisson(lambda), k) for lambda in lambdas]
    p * likelihood
end

function make_gamma_pmf(alpha::Float64, high::Number; n::Int64 = 101)
    vals = [x for x in LinRange(0, high, n)]
    g = Distributions.Gamma(alpha)
    ps = [pdf(g, v) for v in vals];
    pmf_from_seq(vals, normalize(ps))
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

function prob_gt(d::CatDist, threshold)
    sum(probs(d)[values(d).>threshold])
end

function prob_le(d::CatDist, threshold)
    sum(probs(d)[values(d).<=threshold])
end
function prob_lt(d::CatDist, threshold)
    sum(probs(d)[values(d).<threshold])
end

"""
Two functions that do the same thing: Compute the probability
that one pmf is greater than another. I'd expected them to be
about the same speed, but the second seems much faster:

For two pmfs of 101 rows, doing the computation 10000 times takes 
7.4 seconds for prob_gt_old and 2.2 seconds for prob_gt.
"""
function prob_gt_old(d1::CatDist, d2::CatDist)
    sum([p1 * p2
        for (q1, p1) in items(d1)
            for (q2, p2) in items(d2)
                if q1 > q2])
end

function prob_gt(d1::CatDist, d2::CatDist)
    prod(x::Tuple) = x[1] * x[2]
    gt(x::Tuple) = x[1] > x[2]
    g = broadcast(gt, collect(Iterators.product(values(d1), values(d2))))
    p = broadcast(prod, collect(Iterators.product(probs(d1), probs(d2))))
    sum(g .* p)
end

function prob_lt(d1::CatDist, d2::CatDist)
    prob_gt(d2, d1)
end

function prob_eq(d1::CatDist, d2::CatDist)
    1 - (prob_gt(d1, d2) + prob_gt(d2, d1))
end

items(d::CatDist) = [x for x in zip(values(d), probs(d))]

function binom_pmf(k::Number, n::Number, ps::AbstractVector)
    [pdf(Distributions.Binomial(n, p), k) for p in ps]
end

function binom_pmf(k::Number, ns::AbstractVector, p::Number)
    [pdf(Distributions.Binomial(n, p), k) for n in ns]
end

function binom_pmf(ks::AbstractVector, n::Number, p::Number)
    [pdf(Distributions.Binomial(n, p), k) for k in ks]
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
          for (q1, p1) in items(p1)
                for (q2, p2) in items(p2)]
    df = DataFrame(qs=[q for (q, p) in d], ps=[p for (q, p) in d])
    g = groupby(df, :qs)
    d = [(first(x).qs, sum(x.ps)) for x in g]
    CatDist([x[1] for x in d], Distributions.Categorical([x[2] for x in d]))
end

add_dist(p1::CatDist, p2::CatDist) = convolve(p1, p2, +)
add_dist(p1::CatDist, n::Number) = pmf_from_seq(values(p1).+n, probs(p1))
sub_dist(p1::CatDist, p2::CatDist) = convolve(p1, p2, -)
sub_dist(p1::CatDist, n::Number) = pmf_from_seq(values(p1).-n, probs(p1))
mult_dist(p1::CatDist, p2::CatDist) = convolve(p1, p2, *)
mult_dist(p1::CatDist, n::Number) = pmf_from_seq(values(p1).*n, probs(p1))
div_dist(p1::CatDist, p2::CatDist) = convolve(p1, p2, /)
div_dist(p1::CatDist, n::Number) = pmf_from_seq(values(p1)./n, probs(p1))

function dist_op(p1::CatDist, p2::CatDist, func)
    vs = vcat(values(p1), values(p2)) |> sort |> unique
    qs = [func(pdf(p1, v), pdf(p2, v)) for v in vs]
    (vs, qs)
end
function dist_op(p1::CatDist, n::Number, func)
    vs = values(p1)
    qs = func.(probs(p1), n)
    (vs, qs)
end
function dist_op(p1::CatDist, vqs::Tuple{Vector, Vector}, func)
    (vs, qs) = vqs
    vs1 = vcat(values(p1), vs) |> sort |> unique
    qs1 = [func(pdf(p1, v), qs[v]) for v in vs]
    (vs1, qs1)
end
function dist_op(vqs::Tuple{Vector, Vector}, n::Number, func)
    (vs, qs) = vqs
    qsx = func.(qs, n)
    (vs, qsx)
end
function dist_op(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}, func)
    (vs1, qs1) = vqs1
    (vs2, qs2) = vqs2
    vsx = vcat(vs1, vs2) |> sort |> unique
    qsx = [func(qs1[v], qs2[v]) for v in vs]
    (vsx, qsx)
end

(+)(p1::CatDist, p2::CatDist) = dist_op(p1, p2, +)
(+)(p1::CatDist, n::Number) = dist_op(p1, n, +)
(+)(p1::CatDist, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, +)
(+)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, +)
(+)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, +)
(-)(p1::CatDist, p2::CatDist) = dist_op(p1, p2, -)
(-)(p1::CatDist, n::Number) = dist_op(p1, n, -)
(-)(p1::CatDist, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, -)
(-)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, -)
(-)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, -)
(*)(p1::CatDist, p2::CatDist) = dist_op(p1, p2, *)
(*)(p1::CatDist, n::Number) = dist_op(p1, n, *)
(*)(p1::CatDist, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, *)
(*)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, *)
(*)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, *)
(/)(p1::CatDist, p2::CatDist) = dist_op(p1, p2, /)
(/)(p1::CatDist, n::Number) = dist_op(p1, n, /)
(/)(p1::CatDist, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, /)
(/)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, /)
(/)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, /)
make_pmf(vqs::Tuple{Vector, Vector}) = pmf_from_seq(vqs[1], normalize(vqs[2]))

function credible_interval(p1::CatDist, x::Number)
    low = (1.0 - x) / 2.0
    high = 1.0 - low
    quantile(p1, [low, high])
end

function make_df_from_seq_pmf(seq::Vector{CatDist})
    a = [[pdf(x, i) for i in values(x)] for x in seq]
    max_len = length.(a) |> maximum
    a1 = [vcat(x, fill(0, max_len - length(x))) for x in a]
    a2  = reshape(reduce(vcat, a1), max_len, length(seq)) |> transpose
    DataFrame([a2[x,:] for x in 1:length(seq)], :auto)
end

"""
This is the original make_mixture. I leave it here to show how
it could be done with the components, but it's replaced below by 
a version from Distributions.
"""
function make_mixture(pmf, pmf_seq)
    a = [probs(x) for x in pmf_seq]
    max_len = length.(a) |> maximum
    a1 = [vcat(x, fill(0, max_len - length(x))) for x in a]
    a1 = reshape(reduce(vcat, a1), max_len, length(pmf_seq))
    ps = a1 * probs(pmf)
    pmf_from_seq(1:length(ps), ps)
end 

"""

"""
function make_mixture_should_work(pmf::CatDist, pmf_seq::Vector{CatDist})::CatDist
    vs = collect(Iterators.flatten([values(d) for d in pmf_seq])) |> sort |> unique
    m = Distributions.MixtureModel(Distributions.Categorical, [probs(p.dist) for p in pmf_seq], probs(pmf))
    pmf_from_seq(vs, normalize([pdf(m, v+1) for v in vs]))
end


abstract type AbstractDistFunction end
# CDF
export CDF, make_cdf, cdfs, make_pdf, max_dist, min_dist, cdf_from_seq

struct CDF <: AbstractDistFunction
    d:: DataFrame
    q_interp::Any
    c_interp::Any
end

cdf_from_seq(vs:: Vector) = sort(vs) |> pmf_from_seq |> make_cdf

function make_cdf(pmf::CatDist)
    make_cdf(values(pmf), [cdf(pmf, x) for x in values(pmf)])
end

function make_cdf(vs, cs::Vector{Float64})
    q_interp = LinearInterpolation(Interpolations.deduplicate_knots!(cs), vs, extrapolation_bc=Line())
    c_interp = LinearInterpolation(vs, cs, extrapolation_bc = Line())
    CDF(DataFrame(Index=vs, cdf=cs), q_interp, c_interp)
end


function values(cdf::CDF) 
    cdf.d.Index
end

function cdfs(cdf::CDF)
    cdf.d.cdf
end

function cdf(d::CDF, x)
    row = loc(d.d, x)
    if row === nothing
        return d.c_interp(x)
    end
    row.cdf
end

function quantile(d::CDF, x)
    d.q_interp(x)
end

mean(c::CDF) = make_pdf(c) |> mean
var(c::CDF) = make_pdf(c) |> var
std(c::CDF) = make_pdf(c) |> std
function prob_x(c::CDF, x, func)
    p = make_pdf(c)
    func(p, x)
end
prob_le(c::CDF, x) = prob_x(c, x, prob_le)
prob_lt(c::CDF, x) = prob_x(c, x, prob_lt)
prob_ge(c::CDF, x) = prob_x(c, x, prob_ge)
prob_gt(c::CDF, x) = prob_x(c, x, prob_gt)

median(c::CDF) = cdf(c, 0.5)

function plot(d::AbstractDistFunction; xaxis="xs", yaxis="ys", label="y1", plot_title="plot")
    global nplot=1
    plot(values(d), cdfs(d), xaxis=xaxis, yaxis=yaxis, label=label, plot_title=plot_title)
end

function plot!(d::AbstractDistFunction; label=nothing)
    global nplot += 1
    if label===nothing
        label="y"*string(ThinkBayes.nplot)
    end
    plot!(values(d), cdfs(d), label=label)
end

function bar(d::AbstractDistFunction; xaxis=("xs"), yaxis=("ys"), label="y1", plot_title="bar plot")
    bar(values(d), cdfs(d), xaxis=xaxis, yaxis=yaxis, label=label, plot_title=plot_title)
end


function credible_interval(p1::AbstractDistFunction, x::Number)
    low = (1.0 - x) / 2.0
    high = 1.0 - low
    [quantile(p1, low), quantile(p1, high)]
end

function isapprox(p1::AbstractDistFunction, p2::AbstractDistFunction)
    (values(p1) ≈ values(p2)) && (cdfs(p1) ≈ cdfs(p2))
end

function make_pdf(p1::AbstractDistFunction)
    p = cdfs(p1)
    ps = vcat(first(p), diff(p))
    pmf_from_seq(values(p1), ps)
end

function (^)(p1::AbstractDistFunction, x::Number)
    make_cdf(values(p1), cdfs(p1).^x)
end

max_dist(p1::AbstractDistFunction, x::Number) =  p1^x

function min_dist(p1::AbstractDistFunction, x::Number)
    p_gt = make_ccdf(p1);
    prob_gt6 = p_gt^x
    make_cdf(prob_gt6)
end


function show(io::IO, p1::CDF)
    show(p1.d)
end


# CCDF

export make_ccdf

struct CCDF <: AbstractDistFunction
    d:: DataFrame
    q_interp::Any
    c_interp::Any
end

function make_ccdf(p1::CDF)::CCDF
    vs = values(p1)
    cs = 1 .- cdfs(p1)
    make_ccdf(vs, cs)
end
 
function make_ccdf(vs, cs::Vector{Float64})
    knots = Interpolations.deduplicate_knots!(reverse(cs))
    q_interp = LinearInterpolation(knots, reverse(vs), extrapolation_bc=Line())
    c_interp = LinearInterpolation(vs, cs, extrapolation_bc = Line())
    CCDF(DataFrame(Index=vs, cdf=cs), q_interp, c_interp)
end

function make_cdf(p1::CCDF)::CDF
    vs = values(p1)
    cs = 1 .- cdfs(p1)
    make_cdf(vs, cs)
end

function (^)(p1::CCDF, x::Number)
    make_ccdf(values(p1), cdfs(p1).^x)
end
function values(cdf::CCDF) 
    cdf.d.Index
end

function cdfs(cdf::CCDF)
    cdf.d.cdf
end

function cdf(d::CCDF, x)
    row = loc(d.d, x)
    if row === nothing
        return d.c_interp(x)
    end
    row.cdf
end

function quantile(d::CCDF, x)
    d.q_interp(x)
end

function show(io::IO, p1::CCDF)
    show(p1.d)
end

end
