module Tst

include("ThinkStats.jl")
using .ThinkStats
using Plots

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
println("max posterior3=",max_prob(posterior3))
println("pdf 67 of posterior3=",pdf(posterior3, 67))
println("pdf 105 of posterior3=",pdf(posterior3, 105))
println("pdf 67 of posterior3=",posterior3[67])

end
