module ThinkBayes

export Pmf, pmf_from_seq, mult_likelihood, max_prob, min_prob,
    prob_ge, prob_le, prob_gt, prob_lt, prob_eq,
    binom_pmf, normalize, add_dist, sub_dist, mult_dist, div_dist, make_binomial, loc, df_to_matrix,
    update_binomial, credible_interval, make_pmf, make_df_from_seq_pmf, 
    make_mixture, make_poisson_pmf, update_poisson, make_exponential_pmf, 
    make_weibull_pmf, cdf_from_dist,
    make_gamma_pmf, make_normal_pmf, pmf_from_dist, pmf_from_tuples,
    expo_pdf, kde_from_sample, kde_from_pmf, items, outer
    
# from Base:
export getindex, setindex!, copy, values, show, (+), (*), (==), (^), (-), (/), isapprox, transpose
# from Distributions:
export probs, pdf, cdf, plot, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, std, modes, mode, skewness, kurtosis, entropy, mgf, cf
# from Plot:
export plot, plot!, contour, contour!, surface, surface!, bar, bar!

import Plots: plot, plot!, bar, bar!, heatmap, heatmap!, contour, contour!, surface, surface!

import Images: colorview
import ImageTransformations: imresize
import Colors: RGB

import Distributions
import Distributions:  probs, pdf, cdf, maximum, minimum, rand, sampler, logpdf, quantile, insupport,
    mean, var, modes, mode, skewness, kurtosis, entropy, mgf, cf, std, UnivariateDistribution

import Base: copy, getindex, setindex!, values, show, display, (+), (*), (==), (^), (-), (/), isapprox

using DataFrames
import DataFrames: stack, unstack
using Interpolations
using KernelDensity

struct Pmf
    values::Vector
    dist::Distributions.Categorical
end

# Base:

getindex(d::Pmf, val) = pdf(d, val)
function setindex!(d::Pmf, new_prob, val)
   idx = findfirst(values(d).==val)
    ps = probs(d)
    ps[idx] = new_prob
    new_ps = normalize(ps)
    for i in 1:length(ps)
        ps[i] = new_ps[i]
    end
end
copy(d::Pmf) = Pmf(copy(d.values), Distributions.Categorical(copy(probs(d))))
values(d::Pmf) = d.values
function show(io::IO, d::Pmf)
    a=DataFrame(Values=values(d), Probs=probs(d))
    show(io, a)
end
function show(io::IO, ::MIME"text/plain", d::Pmf)
    a=DataFrame(Values=values(d), Probs=probs(d))
    show(io, "text/plain", a)
end
function show(io::IO, ::MIME"text/html", d::Pmf)
    a=DataFrame(Values=values(d), Probs=probs(d))
    show(io, "text/html", a)
end
#display(d::Pmf) = show(d)
(*)(d::Pmf, likelihood) = mult_likelihood(d, likelihood)
(==)(x::Pmf, y::Pmf) = (x.values == y.values) && (probs(x) == probs(y))


# Plots
nplot=1
function plot(d::Pmf; kwargs...)
    global nplot=1
    plot(values(d), probs(d); kwargs...)
end

function plot!(d::Pmf; label=nothing, kwargs...)
    global nplot += 1
    if label===nothing
        label="y"*string(ThinkBayes.nplot)
    end
    plot!(values(d), probs(d); label=label, kwargs...)
end

function bar(d::Pmf; kwargs...)
    bar(values(d), probs(d); kwargs...)
end
function bar!(d::Pmf; kwargs...)
    bar!(values(d), probs(d); kwargs...)
end

function plot(bs::Vector{Pmf}; kwargs...)
	bs_len = length(bs)
	xs = values(bs[1])
	xlen = length(xs)
	ps = reshape(reduce(vcat, [probs(b) for b in bs]), xlen, bs_len)
	titles = reshape(["machine"*string(i) for i in 1:bs_len], 1, bs_len)
	#bar(xs, ps, label=titles, layout=(2,2))
	plot(xs, ps, label=titles, layout=(2,2); kwargs...)
end

function plot(d::UnivariateDistribution; kwargs...)
    low = mean(d) - 2 * std(d)
    high = mean(d) + 2 * std(d)
    plot(pmf_from_dist(range(low, high, length=51), d); kwargs...)
end


# DataFrame

function loc(df::AbstractDataFrame, val)
    idx = findfirst(==(val), df.Index)
    if idx === nothing
        return nothing
    end
   df[idx, :]
end

"""
From a grouped data frame, extract the group names and the values of a given
column as a dictionary of lists.
"""
function collect_vals(gdf::GroupedDataFrame{DataFrame}, idxc, valc)
    Dict([(first(g)[idxc], g[!, valc]) for g in gdf])
end

"""
From a grouped data frame, extract the group names and the result of a function
on the group as a dictionary.
"""
function collect_func(gdf::GroupedDataFrame{DataFrame}, f, idxc, valc)
    groups = combine(gdf, valc => f)
    Dict([(k, groups[findfirst(==(k), groups[!, idxc]), valc*"_"*string(f)]) for k in collect(groups[!, idxc])])
end

# given a dataframe of numbers, return a matrix of those values
df_to_matrix(df::AbstractDataFrame) = reduce(hcat, eachcol(df))

# Distributions

function probs(d::Pmf)
    probs(d.dist)
end

findindex(d::Pmf, x) = findfirst(isequal(x), d.values)

function pdf(d::Pmf, x)
    index = findindex(d, x)
    if index === nothing
        return 0
    end
    pdf(d.dist, index)
end

function logpdf(d::Pmf, x)
    index = findindex(d, x)
    if index === nothing
        return 0
    end
    logpdf(d.dist, index)
end

function cdf(d::Pmf, x)
    index = findindex(d, x)
    if index === nothing
        return 0.0
    end
    cdf(d.dist, index)
end

maximum(d::Pmf) = d.values[maximum(d.dist)]
minimum(d::Pmf) = d.values[minimum(d.dist)]
rand(d::Pmf) = d.values[rand(d.dist)]
function rand(d::Pmf, dim::Int64)
    vals = values(d)
    [vals[rv] for rv in rand(d.dist, dim)]
end
function rand(d::Pmf, dims::Tuple{Vararg{Int64, N}} where N)
    vals = values(d)
    [vals[rv] for rv in rand(d.dist, dims)]
end
sampler(d::Pmf) = Distributions.AliasTable(probs(d))
quantile(d::Pmf, r) = d.values[quantile(d.dist, r)]
insupport(d::Pmf, r) = insupport(d.dist, r)
#TODO: fix mean should be the interpolation of the value at the mean, not the index into the distribution.
#      like mode, but interpolated instead of the actual value.
# Note: I decided to take the simpler route of implementing it like in
#       empiricaldist.py from Allen Downey's ThinkBayes
mean(d::Pmf) = sum(values(d) .* probs(d))
function var(d::Pmf)
    m = mean(d)
    dd = values(d) .- m
    sum(dd .^ 2 .* probs(d))
end
std(d::Pmf) = var(d) ^ 0.5
mode(d::Pmf) = d.values[mode(d.dist)]
modes(d::Pmf) = [d.values[x] for x in modes(d.dist)]
skewness(d::Pmf) = skewness(d.dist)
kurtosis(d::Pmf) = kurtosis(d.dist)
entropy(d::Pmf) = entropy(d.dist)
entropy(d::Pmf, r::Real) = entropy(d.dist, r)
mgf(d::Pmf, r) = mgf(d.dist, r)
cf(d::Pmf, r) = cf(d.dist, r)




# Pmf:

function max_prob(d::Pmf)
    (mp, index)=findmax(identity, (probs(d)))
    d.values[index]
end

function min_prob(d::Pmf)
    (mp, index)=findmin(identity, (probs(d)))
    d.values[index]
end


function pmf_from_seq(seq; counts=nothing)::Pmf
    if counts!==nothing
        a = [fill(x,y) for (x,y) in zip(seq, counts)]
        seq=[(a...)...]
    end
    df=DataFrame(a=seq)
    len=length(seq)
    g=groupby(df, :a)
    d=[(first(x).a, nrow(x)/len) for x in g]
    Pmf([x[1] for x in d], Distributions.Categorical([x[2] for x in d]))
end

function pmf_from_tuples(seq)
    pmf_from_seq([x for (x, y) in seq], [y for (x,y) in seq])
end

function make_poisson_pmf(lamda, vals)
    dist = Distributions.Poisson(lamda)
    pmf_from_dist(vals, dist)
end

"""
make_weibull_pmf(lam, k) is set up to be similar to weilbull_dist in 
chapter 14 of ThinkBayes2 but returning a pmf.
"""
function make_weibull_pmf(λ, k, vals)
    dist = Distributions.Weibull(k, λ)
    # pmf_from_dist(vals, dist) # can't do this since pdf(dist, 0) can === Inf 
    p = [pdf(dist, x) for x in vals]
    probs = map(x -> x === Inf ? 0.0 : x, p)
    pmf_from_seq(vals, normalize(probs))
end

function make_exponential_pmf(lambda::Float64, vals::Vector{Float64})
    λ = 0.000001
    if lambda > 0
        λ = lambda
    end
    e = Distributions.Exponential(1/λ)
    pmf_from_dist(vals, e)
end

function make_exponential_pmf(lambda::Float64, high::Number)
    qs = LinRange(0, high, 101)
    make_exponential_pmf(lambda, [q for q in qs])
end

function  expo_pdf(lambdas::Vector{Float64}, val::Float64)
    [pdf(Distributions.Exponential(1/lambda), val) for lambda in lambdas]
end

function update_poisson(p::Pmf, data)
    k = data
    lambdas = values(p)
    likelihood = [pdf(Distributions.Poisson(lambda), k) for lambda in lambdas]
    p * likelihood
end

function make_gamma_pmf(alpha::Float64, high::Number; n::Int64 = 101)
    vals = [x for x in LinRange(0, high, n)]
    g = Distributions.Gamma(alpha)
    ps = [pdf(g, v) for v in vals];
    pmf_from_dist(vals, g)
end

function make_normal_pmf(mu::Float64 = 0.0, sigma::Float64 = 1.0, low::Number = -5 , high::Number = 5, n::Int64 = 101)
    vals = [x for x in LinRange(low, high, n)]
    make_normal_pmf(vals, mu=mu, sigma=sigma)
end

function make_normal_pmf(vals; mu::Float64 = 0.0, sigma::Float64 = 1.0)
    g = Distributions.Normal(mu, sigma)
    pmf_from_dist(vals, g)
end

function kde_from_sample(d, q_min, q_max, q_n)
    k = kde(d, boundary=(q_min, q_max), npoints=q_n)
    ik = InterpKDE(k)
    qs = [x for x in LinRange(q_min, q_max, q_n)]
    ps = [pdf(ik, x) for x in qs]
    # this next step must be done because the ps for very low numbers
    # seems to alternate signs which is a no-no for Distributions.Categorical
    ps = normalize([x < 1e-12 ? 0 : x for x in ps])
    pmf_from_seq(qs, normalize(ps))
end

function kde_from_pmf(pmf:: Pmf, n=nothing)
    vals = values(pmf)
    n = n === nothing ? length(vals) : n
    kde_from_sample(rand(pmf, n), minimum(vals), maximum(vals), n)
end

function pmf_from_dist(vals, dist::UnivariateDistribution)
    ps = [pdf(dist, v) for v in vals]
    pmf_from_seq(vals, normalize(ps))
end

function pmf_from_seq(seq, probs::Array{Float64})::Pmf
    Pmf(seq, Distributions.Categorical(probs))
end


function pmf_new_probs(d::Pmf, probs)::Pmf
    Pmf(d.values, Distributions.Categorical(probs))
end

function mult_likelihood(d::Pmf, likelihood)::Pmf
    pmf_new_probs(d, normalize(probs(d).*likelihood))
end

function normalize(probs)
    probs./sum(probs)
end

function prob_ge(d::Pmf, threshold)
    sum(probs(d)[values(d).>=threshold])
end

function prob_gt(d::Pmf, threshold)
    sum(probs(d)[values(d).>threshold])
end

function prob_le(d::Pmf, threshold)
    sum(probs(d)[values(d).<=threshold])
end
function prob_lt(d::Pmf, threshold)
    sum(probs(d)[values(d).<threshold])
end

"""
Two functions that do the same thing: Compute the probability
that one pmf is greater than another. I'd expected them to be
about the same speed, but the second seems much faster:

For two pmfs of 101 rows, doing the computation 10000 times takes 
7.4 seconds for prob_gt_old and 2.2 seconds for prob_gt.
"""
function prob_gt_old(d1::Pmf, d2::Pmf)
    sum([p1 * p2
        for (q1, p1) in items(d1)
            for (q2, p2) in items(d2)
                if q1 > q2])
end

function prob_gt(d1::Pmf, d2::Pmf)
    prod(x::Tuple) = x[1] * x[2]
    gt(x::Tuple) = x[1] > x[2]
    g = broadcast(gt, collect(Iterators.product(values(d1), values(d2))))
    p = broadcast(prod, collect(Iterators.product(probs(d1), probs(d2))))
    sum(g .* p)
end

function prob_lt(d1::Pmf, d2::Pmf)
    prob_gt(d2, d1)
end

function prob_eq(d1::Pmf, d2::Pmf)
    1 - (prob_gt(d1, d2) + prob_gt(d2, d1))
end

items(d::Pmf) = [x for x in zip(values(d), probs(d))]

function binom_pmf(k::Number, n::Number, ps::AbstractVector)
    [pdf(Distributions.Binomial(n, p), k) for p in ps]
end

function binom_pmf(k::Number, ns::AbstractVector, p::Number)
    [pdf(Distributions.Binomial(n, p), k) for n in ns]
end

function binom_pmf(ks::AbstractVector, n::Number, p::Number)
    [pdf(Distributions.Binomial(n, p), k) for k in ks]
end

function binom_pmf(k::Number, n::Number, p::Number)
    pdf(Distributions.Binomial(n, p), k)
end

function make_binomial(n, p)
    """Make a binomial Pmf."""
    binom=Distributions.Binomial(n, p)
    ks=[pdf(binom, x) for x in 0:n]
    pmf_from_seq(0:n, ks)
end

function update_binomial(pmf::Pmf, data)
    (k, n) = data
    xs = values(pmf)
    likelihood=binom_pmf(k, n, xs)
    pmf*=likelihood
end


function convolve(d1::Pmf, d2::Pmf, func)
    d = sort([(func(q1, q2), (p1 * p2)) 
          for (q1, p1) in items(d1)
                for (q2, p2) in items(d2)])
    df = DataFrame(qs=[round(q, digits=8) for (q, p) in d], ps=[p for (q, p) in d])
    g = groupby(df, :qs)
    d = [(first(x).qs, sum(x.ps)) for x in g]
    Pmf([q for (q, p) in d], Distributions.Categorical([p for (q, p) in d]))
end

add_dist(p1::Pmf, p2::Pmf) = convolve(p1, p2, +)
add_dist(p1::Pmf, n::Number) = pmf_from_seq(values(p1).+n, probs(p1))
sub_dist(p1::Pmf, p2::Pmf) = convolve(p1, p2, -)
sub_dist(p1::Pmf, n::Number) = pmf_from_seq(values(p1).-n, probs(p1))
mult_dist(p1::Pmf, p2::Pmf) = convolve(p1, p2, *)
mult_dist(p1::Pmf, n::Number) = pmf_from_seq(values(p1).*n, probs(p1))
div_dist(p1::Pmf, p2::Pmf) = convolve(p1, p2, /)
div_dist(p1::Pmf, n::Number) = pmf_from_seq(values(p1)./n, probs(p1))

function dist_op(p1::Pmf, p2::Pmf, func)
    vs = vcat(values(p1), values(p2)) |> sort |> unique
    qs = [func(pdf(p1, v), pdf(p2, v)) for v in vs]
    (vs, qs)
end
function dist_op(p1::Pmf, n::Number, func)
    vs = values(p1)
    qs = func.(probs(p1), n)
    (vs, qs)
end
function dist_op(p1::Pmf, vqs::Tuple{Vector, Vector}, func)
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

(+)(p1::Pmf, p2::Pmf) = dist_op(p1, p2, +)
(+)(p1::Pmf, n::Number) = dist_op(p1, n, +)
(+)(p1::Pmf, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, +)
(+)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, +)
(+)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, +)
(-)(p1::Pmf, p2::Pmf) = dist_op(p1, p2, -)
(-)(p1::Pmf, n::Number) = dist_op(p1, n, -)
(-)(p1::Pmf, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, -)
(-)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, -)
(-)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, -)
(*)(p1::Pmf, p2::Pmf) = dist_op(p1, p2, *)
(*)(p1::Pmf, n::Number) = dist_op(p1, n, *)
(*)(p1::Pmf, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, *)
(*)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, *)
(*)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, *)
(/)(p1::Pmf, p2::Pmf) = dist_op(p1, p2, /)
(/)(p1::Pmf, n::Number) = dist_op(p1, n, /)
(/)(p1::Pmf, vqs::Tuple{Vector, Vector}) = dist_op(p1, vqs, /)
(/)(vqs::Tuple{Vector, Vector}, n::Number) = dist_op(vqs, n, /)
(/)(vqs1::Tuple{Vector, Vector}, vqs2::Tuple{Vector, Vector}) = dist_op(vqs1, vqs2, /)
make_pmf(vqs::Tuple{Vector, Vector}) = pmf_from_seq(vqs[1], normalize(vqs[2]))

function credible_interval(p1::Pmf, x::Number)
    low = (1.0 - x) / 2.0
    high = 1.0 - low
    quantile(p1, [low, high])
end

function make_df_from_seq_pmf(seq::Vector{Pmf})
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

Well, it should be replaced, but I can't make the "better one" work :(
"""
function make_mixture(pmf, pmf_seq)
    a = [probs(x) for x in pmf_seq]
    max_len = length.(a) |> maximum
    a1 = [vcat(x, fill(0, max_len - length(x))) for x in a]
    a1 = reshape(reduce(vcat, a1), max_len, length(pmf_seq))
    ps = a1 * probs(pmf)
    pmf_from_seq(values(pmf), ps)
end 

"""

"""
function make_mixture_should_work(pmf::Pmf, pmf_seq::Vector{Pmf})::Pmf
    vs = collect(Iterators.flatten([values(d) for d in pmf_seq])) |> sort |> unique
    m = Distributions.MixtureModel(Distributions.Categorical, [probs(p.dist) for p in pmf_seq], probs(pmf))
    pmf_from_seq(vs, normalize([pdf(m, v+1) for v in vs]))
end

"""
outer(f, x, y) where f is a function of two variables like +, *, >, etc
               x, y are vectors
        
does x' f y which produces a matrix of length(y) rows and length(x) columns, then 
transforms this to a length(x) vector of vectors of the rows.

e.g.
x = [1,3,5]
y = [2,4]
outer(*, x, y) => [[2, 6, 10], [4, 12, 20]]
"""
function outer(f, x, y)
    d = f.(x', y)
end

"""
Joint distributions.
"""

export Joint, make_joint, visualize_joint, visualize_joint!, column, row, joint_to_df,
    collect_vals, collect_func, marginal, stack, unstack, index, columns

struct Joint
    M::Matrix
    xs::Vector 
    ys::Vector
end

"""
Exactly like outer(f, x, y) but works with pmfs instead of vectors.
Returns a Joint structure that contains the results as a Matrix
where along with the xs and ys.
"""
function make_joint(f, x_pmf, y_pmf)
    x_p = probs(x_pmf)
	y_p = probs(y_pmf)
	v = outer(f, x_p, y_p)
	Joint(v, values(x_pmf), values(y_pmf))
end

function column(j::Joint, x_val)
    index = findfirst(round.(j.xs, digits=3) .≈ x_val)
    j.M[:,index]
end

function row(j::Joint, y_val)
    index = findfirst(round.(j.ys, digits=3) .≈ y_val)
    j.M[index,:]
end

columns(j::Joint) = j.ys
index(j::Joint) = j.xs

function joint_to_df(j::Joint)
    DataFrame(hcat(j.ys, j.M), vcat(["Index"], string.(round.(j.xs, digits=3))))
end

function marginal(joint::Joint, dim)
	sums = sum(joint.M, dims=dim)
	pmf_from_seq(dim==1 ? joint.xs : joint.ys, vec(sums))
end

function mult_likelihood(j::Joint, likelihood)
    Joint(normalize(j.M .* likelihood), j.xs, j.ys)
end

(*)(d::Joint, likelihood) = mult_likelihood(d, likelihood)

(+)(d1::Joint, d2::Joint) = Joint(d1.M .+ d2.M, d1.xs, d2.xs)

"""
stack(j::Joint)

flatten a joint distribution to two vectors.
The first is a vector of tuples of all combinations of the xs, ys
The second is a vector of the equivalent probabilities.

So, the value of (x, y) in the first vector is the same index as 
the probability of the xth row and the yth column.

Thus, a joint that looks like:

    1   2
3  0.5  0.2
4  0.1  0.4

will look like:
([(1,3), (1,4) (2,3), (2,4)], [0.5, 0.1, 0.2, 0.4])
"""
function stack(j::Joint)
    vals = [(x, y) for x in j.xs for y in j.ys]
    probs = vec(j.M)
    (vals, probs)
end

"""
The inverse of stack.

For qs [(1,3), (2,3)]
"""
function unstack(qs, vs)
    xs = unique([x for (x,y) in qs])
    ys = unique([y for (x,y) in qs])
    M = reshape(vs, length(ys), length(xs))
    Joint(M, xs, ys)
end

transpose(j::Joint) = Joint(transpose(j.M), columns(j), index(j))

function show(io::IO, j::Joint)
    show(joint_to_df(j))
end
function show(io::IO, ::MIME"text/plain", j::Joint)
    show(io, "text/plain", joint_to_df(j))
end
function show(io::IO, ::MIME"text/html", j::Joint)
    show(io, "text/html", joint_to_df(j))
end

(==)(j1::Joint, j2::Joint) = (index(j1) == index(j2)) && (columns(j1) == columns(j2)) && (j1.M == j2.M)

"""
A simple visualization. 
TODO: enhancement opportunity!
"""
visualize_joint_old(joint::DataFrame) = visualize_joint(df_to_matrix(joint))

function visualize_joint_old(M::Matrix{Float64})
    r, c = size(M)
    # flip 90° right
    M = reshape([M[i, j] for i in r:-1:1 for j in 1:c], r, c)
    Mx = 1 .- (M * (1 / maximum(M)))
    M2 = zeros(size(M))
    fill!(M2, 1.0)
    imresize(colorview(RGB, Mx, Mx, M2), ratio = 5)
end

function visualize_joint!(joint::Joint; c = :greys, xaxis="XS", yaxis="YS", normalize=false, alpha=1.0, is_contour=false)
    visualize_joint(joint, c=c, xaxis=xaxis, yaxis=yaxis, normalize=normalize, secondary=true, alpha=alpha, is_contour=is_contour)
end
function visualize_joint(joint::Joint; c = :greys, xaxis="XS", yaxis="YS", normalize=false, secondary=false, alpha=1.0, is_contour=false)
    M = joint.M
    ys = joint.ys
    xs = joint.xs
    visualize_joint(M, xs=xs, ys=ys, c=c, xaxis=xaxis, yaxis=yaxis, normalize=normalize, secondary=secondary, alpha=alpha, is_contour=is_contour)
end

function visualize_joint!(M::AbstractMatrix; xs = missing, ys=missing, c = :greys, xaxis="XS", yaxis="YS", normalize=false, alpha=1.0, is_contour=false)
    visualize_joint(M, xs=xs, ys=ys, c=c, xaxis=xaxis, yaxis=yaxis, normalize=normalize, secondary=true, alpha=alpha, is_contour=is_contour)
end    
function visualize_joint(M::AbstractMatrix; xs = missing, ys=missing, c = :greys, xaxis="XS", yaxis="YS", normalize=false, secondary=false, alpha=1.0, is_contour=false)
    rows,cols = size(M)
    txs = xs === missing ? (1:rows) : xs
    tys = ys === missing ? (1:cols) : ys
    if normalize
        M = M .* (1 / maximum(M))
    end
    if secondary
        if is_contour
            contour!(txs, tys, M, c=c, xaxis=xaxis, yaxis=yaxis, alpha=alpha)
        else
            heatmap!(txs, tys, M, c=c, xaxis=xaxis, yaxis=yaxis, alpha=alpha)
        end
    else
        if is_contour
            contour(txs, tys, M, c=c, xaxis=xaxis, yaxis=yaxis, alpha=alpha)
        else
            heatmap(txs, tys, M, c=c, xaxis=xaxis, yaxis=yaxis, alpha=alpha)
        end
    end
end

function contour(j::Joint; kwargs...)
    contour(j.xs, j.ys, j.M; kwargs...)
end

function contour!(j::Joint; kwargs...)
    contour!(j.xs, j.ys, j.M; kwargs...)
end

function surface(j::Joint; kwargs...)
    surface(j.xs, j.ys, j.M; kwargs...)
end

abstract type AbstractDistFunction end

"""
CDF
"""

export CDF, make_cdf, cdfs, make_pdf, max_dist, min_dist, cdf_from_seq

struct CDF <: AbstractDistFunction
    d:: DataFrame
    q_interp::Any
    c_interp::Any
end

cdf_from_seq(vs) = sort(vs) |> pmf_from_seq |> make_cdf

function make_cdf(pmf::Pmf)
    make_cdf(values(pmf), [cdf(pmf, x) for x in values(pmf)])
end

function make_cdf(vs, cs::Vector{Float64})
    q_interp = LinearInterpolation(Interpolations.deduplicate_knots!(cs), vs, extrapolation_bc=Line())
    c_interp = LinearInterpolation(vs, cs, extrapolation_bc = Line())
    CDF(DataFrame(Index=vs, cdf=cs), q_interp, c_interp)
end

function cdf_from_dist(vs, dist)
    make_cdf(vs, cdf(dist, vs))
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

function plot(d::AbstractDistFunction; kwargs...)
    global nplot=1
    plot(values(d), cdfs(d); kwargs...)
end

function plot!(d::AbstractDistFunction; label=nothing, kwargs...)
    global nplot += 1
    if label===nothing
        label="y"*string(ThinkBayes.nplot)
    end
    plot!(values(d), cdfs(d), label=label; kwargs...)
end

function bar(d::AbstractDistFunction; kwargs...)
    bar(values(d), cdfs(d), kwargs...)
end
function bar!(d::AbstractDistFunction; kwargs...)
    bar!(values(d), cdfs(d), kwargs...)
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
function show(io::IO, ::MIME"text/plain", p1::CDF)
    show(io, "text/plain", p1.d)
end
function show(io::IO, ::MIME"text/html", p1::CDF)
    show(io, "text/html", p1.d)
end

"""
CCDF
"""

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
