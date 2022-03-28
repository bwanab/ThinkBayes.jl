module Tst

#include("ThinkStats.jl")
using ThinkStats
using Plots
using Test

@testset "Test ThinkStats" begin
    hypos=0:100
    prior=pmf_from_seq(hypos)
    likelihood_vanilla=hypos./100
    posterior1=mult_likelihood(prior, likelihood_vanilla)
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
    @test pdf(posterior3, 67)≈0.01777821782178218
    @test pdf(posterior3, 105)==0
    @test posterior3[67]==pdf(posterior3, 67)
    @test values(posterior3)==0:100
    @test pmf_with_probs([6, 8, 12], [1, 2, 3])[8]≈0.333333333333333
end
end
