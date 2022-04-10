module Tst

include("../src/ThinkBayes.jl")
using .ThinkBayes
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
    @test mean(posterior3)≈ 60.995999999999995
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
    @test mean(reduce(add_dist, fill(d, 3))) ≈ 8.5
    @test probs(sub_dist(d, d))[4] ≈ 0.1111111111111111
    @test values(sub_dist(d, d))[4] ≈ -2
    @test probs(mult_dist(d, d))[6] ≈ 0.1111111111111111
    @test values(mult_dist(d, d))[6] ≈ 6
    @test probs(make_binomial(4, 0.5))[3] ≈ 0.375
    c = make_cdf(d)
    @test cdf(c, 3) ≈ 0.5
    @test cdf(c, 3.5) ≈ 0.5833333333333
    @test quantile(c, 0.5) == 3
    d1 = make_pdf(c)
    @test values(d) == values(d1)
    @test probs(d) ≈ probs(d1)
    cc = make_ccdf(c)
    @test cdf(cc, 3) ≈ 0.5
    @test cdf(cc, 3.5) ≈ 0.41666666666
    c1 = make_cdf(cc)
    @test c ≈ c1
end
end
