module Tst

include("ThinkStats.jl")
import .ThinkStats
using Plots

hypos=0:100
prior=ThinkStats.pmf_from_seq(hypos)
likelihood_vanilla=hypos./100
posterior1=ThinkStats.mult_likelihood(prior, likelihood_vanilla)
plot([ThinkStats.probs(prior), ThinkStats.probs(posterior1)])
savefig("posterior1.svg")
posterior2=ThinkStats.mult_likelihood(posterior1, likelihood_vanilla)
plot([ThinkStats.probs(posterior1), ThinkStats.probs(posterior2)])
savefig("posterior2.svg")
likelihood_chocolate=1 .- hypos./100
posterior3=ThinkStats.mult_likelihood(posterior2, likelihood_chocolate)
plot([ThinkStats.probs(posterior2), ThinkStats.probs(posterior3)])
savefig("posterior3.svg")
print("max posterior3=",ThinkStats.max_prob(posterior3))

end
