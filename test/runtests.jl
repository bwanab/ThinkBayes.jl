module Tst

#include("../src/ThinkBayes.jl")
#using .ThinkBayes
using ThinkBayes
using DataFrames
using Distributions: Beta
using Plots
using Test

@testset "Test ThinkBayes" begin
    hypos=0:100
    prior=pmf_from_seq(hypos)
    likelihood_vanilla=hypos./100
    posterior1=mult_likelihood(prior, likelihood_vanilla)
    posterior1x=prior * likelihood_vanilla
    @test posterior1==posterior1x
    plot([probs(prior), probs(posterior1)])
    savefig("posterior1.svg")
    posterior2=mult_likelihood(posterior1, likelihood_vanilla)
    plot([probs(posterior1), probs(posterior2)])
    savefig("posterior2.svg")
    likelihood_chocolate=1 .- hypos./100
    posterior3=mult_likelihood(posterior2, likelihood_chocolate)
    plot([probs(posterior2), probs(posterior3)])
    savefig("posterior3.svg")
    @test max_prob(posterior3)==67
    @test pdf(posterior3, 67)≈ 0.01777821782178218
    @test pdf(posterior3, 105)==0
    @test posterior3[67]==pdf(posterior3, 67)
    @test values(posterior3)==0:100
    t=pmf_from_seq([6, 8, 12], counts=[1, 2, 3])
    @test t[8]≈1/3
    @test pmf_from_seq(values(t), probs(t))[8]≈1/3
    @test cdf(posterior3, 67)≈ 0.6074666666666667
    @test maximum(posterior3)==100
    @test minimum(posterior3)==0
    @test rand(posterior3)>=0
    #@test sampler(posterior3)
    @test logpdf(posterior3, 67)≈-4.029781288915619
    @test quantile(posterior3, 1/3)==51
    @test insupport(posterior3, 1/3)==false
    @test mean(posterior3)≈ 59.995999999999995
    @test var(posterior3)≈ 399.879984
    @test modes(posterior3)==[67]
    @test mode(posterior3)≈67
    @test skewness(posterior3)≈ -0.2860714196611638
    @test kurtosis(posterior3)≈ -0.6430360893518259
    @test entropy(posterior3)≈ 4.369729458150812
    #@test mgf(posterior3)≈3
    #@test cf
    @test sum(binom_pmf(140, 250, range(0, 1, length=101)))≈0.39840637450199445
    @test normalize(1:10)[1]≈1/55
    d=pmf_from_seq(1:6)
    @test mean(reduce(add_dist, fill(d, 3))) ≈ 10.5
    @test probs(sub_dist(d, d))[4] ≈ 0.1111111111111111
    xs = (values(d), [x for x in 0.1:0.01:0.15])
    @test probs(d - xs |> make_pmf)[3] ≈ 0.18666666666666668
    @test values(sub_dist(d, d))[4] ≈ -2
    @test probs(mult_dist(d, d))[6] ≈ 0.1111111111111111
    @test probs(d * xs |> make_pmf)[6] ≈ 0.2
    @test values(mult_dist(d, d))[6] ≈ 6
    @test values(d / xs |> make_pmf)[6] ≈ 6
    @test probs(make_binomial(4, 0.5))[3] ≈ 0.375
    c = make_cdf(d)
    @test cdf(c, 3) ≈ 0.5
    @test cdf(c, 3.5) ≈ 0.5833333333333
    @test quantile(c, 0.5) == 3
    @test mean(c) ≈ mean(d)
    @test var(c) ≈ var(d)
    @test std(c) ≈ std(d)
    @test prob_le(c, 3) ≈ prob_le(d, 3)
    @test prob_le(c, 3) ≈ 0.5
    d1 = make_pdf(c)
    @test values(d) == values(d1)
    @test probs(d) ≈ probs(d1)
    cc = make_ccdf(c)
    @test cdf(cc, 3) ≈ 0.5
    @test cdf(cc, 3.5) ≈ 0.41666666666
    c1 = make_cdf(cc)
    @test c ≈ c1
    d1 = pmf_from_seq(3:8)
    d2 = pmf_from_seq(1:2:12, counts=[1, 2, 3, 3, 2, 1])
    @test prob_gt(d1, d2) ≈ 0.38888888888888895
    pdf_n = make_normal_pmf(LinRange(-5, 5, 50))
    @test round(mean(pdf_n), digits = 8) ≈ 0.0
    @test round(std(pdf_n), digits = 4) ≈ 1.0
    beta_dist = Beta(2, 2)
    pdf_beta = pmf_from_dist(LinRange(0, 1, 50), beta_dist)
    @test mean(pdf_beta) ≈ 0.5
    @test round(std(pdf_beta), digits=5) ≈ 0.22342
    x_pmf = make_normal_pmf(range(-5, 5, 51))
    y_pmf = make_normal_pmf(range(0, 10, 51), mu=5.0, sigma=2.0)
    j = make_joint(*, x_pmf, y_pmf)
    @test round(sum(column(j, 1.0)), digits=4) ≈ 0.0484
    @test round(sum(row(j, 1.0)), digits=4) ≈ 0.0055
    m1 = marginal(j, 1)
    @test round(pdf(m1, 1.0), digits=4) ≈ 0.0484
    m2 = marginal(j, 2)
    @test round(pdf(m2, 1.0), digits=4) ≈ 0.0055
    @test round(sum(row(j + j, 1.0)), digits=4) ≈ 0.0712
    @test round(sum(column(j + j, 1.0)), digits=4) ≈ 0.0968
    likelihood = reshape(range(0, 10, length=51*51), 51, 51)
    @test round(sum(row(j * likelihood, 1.0)), digits=4) ≈ 0.0054
    @test round(sum(column(j * likelihood, 1.0)), digits=4) ≈ 0.0579
    df = DataFrame(a=vcat(fill("this", 6), fill("that",6)), b=1:12)
    gdf = groupby(df, "a")
    @test collect_vals(gdf, "a", "b")["this"] == collect(1:6)
    @test collect_vals(gdf, "a", "b")["that"] == collect(7:12)
    d = collect_func(gdf, pmf_from_seq, "a", "b")
    @test round(pdf(d["this"], 4), digits=4) ≈ 0.1667
    @test round(pdf(d["that"], 10), digits=4) ≈ 0.1667
    a = [1,2]
    b = [4,5,6]
    j = Joint(outer(*, a, b), a, b)
    qs,ps = stack(j)
    j2 = unstack(qs, ps)
    @test j2 == j
    d1, d2, d3 = 20, 30, 50
    pmf1 = pmf_from_seq(1:d1, normalize(rand(d1)))
    pmf2 = pmf_from_seq(1:d2, normalize(rand(d2)))
    pmf3 = pmf_from_seq(1:d3, normalize(rand(d3)))
    joint2 = make_joint(*, pmf2, pmf3)
    vs1, probs1 = stack(joint2)
    joint2_pmf = pmf_from_seq(vs1, probs1)
    joint3 = make_joint(*, pmf1, joint2_pmf)
    vs2, probs2 = stack(joint3)
    joint3_pmf = pmf_from_seq(vs2, probs2)
    m1, m2, m3 = marginals3(joint3_pmf)
    @test m1 ≈ pmf1
    @test m2 ≈ pmf2
    @test m3 ≈ pmf3
    """
    Note that the following is a set of experiments to think about how to 
    make a Joint of 3 (or more) parameters/dimensions work.
    """
    M3 = reshape(probs(joint3_pmf), 50,30,20)
    #@test reshape(sum(sum(M3, dims=1), dims=2), d3) ≈ probs(m1)
    #@test reshape(sum(sum(M3, dims=1), dims=3), d2) ≈ probs(m2)
    #@test reshape(sum(sum(M3, dims=2), dims=3), d1) ≈ probs(m3)
    @test [sum(M3[:,:,x]) for x in 1:d1] ≈ probs(m1)
    @test [sum(M3[:,x,:]) for x in 1:d2] ≈ probs(m2)
    @test [sum(M3[x,:,:]) for x in 1:d3] ≈ probs(m3)

    """
    Using BenchmarkTools, I've determined that for arrays of significant
    size the view method works best.
    """

    @test [sum(view(M3, :,:,x)) for x in 1:d1] ≈ probs(m1)
    @test [sum(view(M3, :,x,:)) for x in 1:d2] ≈ probs(m2)
    @test [sum(view(M3, x,:,:)) for x in 1:d3] ≈ probs(m3)

    """
    This is how one might compute the 3rd paramter 3d matrix of a given Joint and a Pmf.
    """

    M4 = permutedims(reshape(outer(*, reshape(joint2.M, prod(size(joint2.M))), probs(pmf1)), d1,d3,d2), (1,3,2))
    # @test joint3.M == M4
    @test [sum(view(M4, :,:,x)) for x in 1:d3] ≈ probs(m3)
    @test [sum(view(M4, x,:,:)) for x in 1:d1] ≈ probs(m1)
    @test [sum(view(M4, :,x,:)) for x in 1:d2] ≈ probs(m2)
end
end
