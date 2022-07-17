### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d56f1f09-3661-4f45-93ef-67847ef7c32b
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ 75b5a3f6-6bc0-4974-8280-78d887d4d6c5
begin
	using DataFrames, CSV, Plots, Distributions, PlutoUI, GraphViz, LinearAlgebra, CategoricalArrays
	using Mamba: Model, Stochastic, NUTS, setsamplers!, graph2dot, mcmc, predict
	
end

# ╔═╡ c79c78b4-020a-11ed-2145-330672983f9c
md"# MCMC

## The World Cup Problem"

# ╔═╡ 59569740-92cd-4451-9fc9-c0c2afd43df0
TableOfContents()

# ╔═╡ dfa9e70a-fb23-4836-aaaa-78c4238706be
md"## Grid Approximation"

# ╔═╡ 3f61c4f7-e31a-4073-b707-5c32b7658bfb
begin
	alpha = 1.4
	prior_dist = Gamma(alpha)
end

# ╔═╡ ed06502f-c33f-41dc-860a-9e1107d0a00e
lams = 0:14

# ╔═╡ 511b9ef4-3b43-4de1-aad8-1bdf7c27f787
prior_pmf = pmf_from_dist(lams, prior_dist);

# ╔═╡ 7da29c34-26fb-4372-b09f-93559de91df2
likelihood = [pdf(Poisson(l), 4) for l in lams]

# ╔═╡ dfdacad3-bd7f-44fb-8ba9-b72768961fb9
posterior = prior_pmf * likelihood;

# ╔═╡ b2f17e4b-1b83-489c-ab4b-5da6ea692a0f
md"## Prior Predictive Distribution"

# ╔═╡ c696b370-c61b-4554-b618-c2e1cd648dc1
sample_prior = rand(prior_dist, 1000)

# ╔═╡ acb901f6-de48-41c2-a557-a42685de076f
sample_prior_pred = [rand(Poisson(s)) for s in sample_prior]

# ╔═╡ 72408f03-6f96-4575-929d-9a688de8aca8
bar(pmf_from_seq(sample_prior_pred), xaxis=((0:11)))

# ╔═╡ 1127ff19-172c-4848-b474-6a933abfdd29
line = Dict{Symbol, Any}(
  :x => [1],
  :goals => [4]
)


# ╔═╡ d8b58d1c-8bfa-4185-bf66-1fff56cff4e4
model = Model(
  goals = Stochastic(1,
    (lam) -> Poisson(lam * 1.0),
    false
  ),
  lam = Stochastic(() -> Gamma(1.4, 1))
);

# ╔═╡ 6c9f8c9c-1d6d-4ecc-896e-352db02afccb
scheme = [NUTS([:lam])];

# ╔═╡ e40f4882-f198-44a1-bb23-1abe147a8ec9
setsamplers!(model, scheme);

# ╔═╡ 3aab27f5-6fe2-4fb9-ac48-84ee80b1259a
inits = [
	Dict{Symbol, Any}(
		:lam => rand(Gamma(1.4, 1)),
		:goals => line[:goals]
	)
	for i in 1:3
]

# ╔═╡ 88f5d3ad-6ff7-4f0b-b4e2-060f6b320bc4


# ╔═╡ 3cc7f9f2-0496-41cb-8d62-ebcf57664857
GraphViz.Graph(graph2dot(model))

# ╔═╡ a8e7aae9-5836-4dcd-a9d9-b60b41585a05
sim = mcmc(model, line, inits, 2000, burnin=1000, chains=1)

# ╔═╡ 20762045-c064-4855-bf97-4da826739b63
x = predict(sim)

# ╔═╡ cb7d5d80-e47e-44c3-9a02-9589f3445e8c
preds = reshape(x.value, 1000)

# ╔═╡ 7842faa9-ee48-4e45-b7f9-18b76dd889a4
begin
	Plots.plot(cdf_from_seq(sample_prior), label="prior")
	Plots.plot!(cdf_from_seq(preds), label="Mamba")
	Plots.plot!(make_cdf(posterior), label="Grid")
end

# ╔═╡ 6c7f4d90-3722-48cb-b670-f70fc1d8866b
mean(preds), std(preds)

# ╔═╡ 58d74942-ac13-4137-86b5-3d8690433db6
mean(posterior), std(posterior)

# ╔═╡ 8287fb37-92bb-4c82-9eb3-7e6881cfceb6
histogram(preds)

# ╔═╡ cdd7c26e-8517-4f45-80bf-ced6056a8a16
pred_pmf = pmf_from_seq(preds)

# ╔═╡ 7d495113-7d9b-4d94-9ec4-b1faa2f68dd2
posterior

# ╔═╡ 67f33a03-0477-47db-815a-c9bc1b73306d
max_prob(posterior), max_prob(pred_pmf), mean(posterior), mean(pred_pmf)

# ╔═╡ 055f1d15-f169-446d-83ae-a06c998af9b0
begin
	plot(pred_pmf, xaxis=0:14)
	plot!(posterior)
end

# ╔═╡ f9440f1c-ad4f-4c49-9c10-38173bf69305
begin
	plot(make_cdf(pred_pmf), seriestype=:steppre, label="Mamba samples")
	plot!(make_cdf(posterior), seriestype=:steppre, label="grid samples")
end

# ╔═╡ Cell order:
# ╟─c79c78b4-020a-11ed-2145-330672983f9c
# ╠═d56f1f09-3661-4f45-93ef-67847ef7c32b
# ╠═75b5a3f6-6bc0-4974-8280-78d887d4d6c5
# ╠═59569740-92cd-4451-9fc9-c0c2afd43df0
# ╟─dfa9e70a-fb23-4836-aaaa-78c4238706be
# ╠═3f61c4f7-e31a-4073-b707-5c32b7658bfb
# ╠═ed06502f-c33f-41dc-860a-9e1107d0a00e
# ╠═511b9ef4-3b43-4de1-aad8-1bdf7c27f787
# ╠═7da29c34-26fb-4372-b09f-93559de91df2
# ╠═dfdacad3-bd7f-44fb-8ba9-b72768961fb9
# ╟─b2f17e4b-1b83-489c-ab4b-5da6ea692a0f
# ╠═c696b370-c61b-4554-b618-c2e1cd648dc1
# ╠═acb901f6-de48-41c2-a557-a42685de076f
# ╠═72408f03-6f96-4575-929d-9a688de8aca8
# ╠═1127ff19-172c-4848-b474-6a933abfdd29
# ╠═d8b58d1c-8bfa-4185-bf66-1fff56cff4e4
# ╠═6c9f8c9c-1d6d-4ecc-896e-352db02afccb
# ╠═e40f4882-f198-44a1-bb23-1abe147a8ec9
# ╠═3aab27f5-6fe2-4fb9-ac48-84ee80b1259a
# ╠═88f5d3ad-6ff7-4f0b-b4e2-060f6b320bc4
# ╠═3cc7f9f2-0496-41cb-8d62-ebcf57664857
# ╠═a8e7aae9-5836-4dcd-a9d9-b60b41585a05
# ╠═20762045-c064-4855-bf97-4da826739b63
# ╠═cb7d5d80-e47e-44c3-9a02-9589f3445e8c
# ╠═7842faa9-ee48-4e45-b7f9-18b76dd889a4
# ╠═6c7f4d90-3722-48cb-b670-f70fc1d8866b
# ╠═58d74942-ac13-4137-86b5-3d8690433db6
# ╠═8287fb37-92bb-4c82-9eb3-7e6881cfceb6
# ╠═cdd7c26e-8517-4f45-80bf-ced6056a8a16
# ╠═7d495113-7d9b-4d94-9ec4-b1faa2f68dd2
# ╠═67f33a03-0477-47db-815a-c9bc1b73306d
# ╠═055f1d15-f169-446d-83ae-a06c998af9b0
# ╠═f9440f1c-ad4f-4c49-9c10-38173bf69305
