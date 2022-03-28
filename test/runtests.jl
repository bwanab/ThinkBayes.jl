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
    t=pmf_from_seq([6, 8, 12], counts=[1, 2, 3])
    @test t[8]≈1/3
    @test pmf_from_seq(values(t), probs(t))[8]≈1/3
end
end
