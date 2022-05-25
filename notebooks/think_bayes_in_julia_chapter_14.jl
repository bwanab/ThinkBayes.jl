### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 11bfe214-db8f-11ec-24d9-21c6b862ea54
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ a0bf7ab3-541d-4182-ba67-5de432a53ff9
using DataFrames, CSV, Plots, Distributions, PlutoUI

# ╔═╡ b240adda-091b-45ac-878d-302f7b18d610
TableOfContents()

# ╔═╡ f1824111-27c6-4027-a3b3-0552f2a154d4
md"## The Weibull Distribution"

# ╔═╡ 5e16f035-5003-4b6e-851b-79eb47716120
dist = Weibull(0.8, 3)

# ╔═╡ 90b7bc4c-9cd5-4ecc-b4f6-a3e9c70072b0
wd = cdf_from_dist(range(0.12, 12.12, 101), dist);

# ╔═╡ 9e7c28b0-b11e-4800-bf51-c63db77bf4c0
plot(wd)

# ╔═╡ 69cf8c0d-7faf-4a36-8da3-96d2588e029a
data=rand(dist, 10)

# ╔═╡ 1cc86501-f32c-408d-a1db-693e51a996ea
lams = range(0.1, 10.1, 101)

# ╔═╡ 0916bfe2-447f-4a7e-9d27-6c5553ddbabf
prior_lam = pmf_from_seq(lams);

# ╔═╡ 954da934-c9c1-4eb5-a7fd-449eb4ecd44c
begin
	ks = range(0.1, 5.1, 101)
	prior_k = pmf_from_seq(ks)
end;

# ╔═╡ edc59f76-d891-4388-ae1c-bcab84c44c65
prior = make_joint(*, prior_lam, prior_k);

# ╔═╡ ac201ad5-cace-43d9-8dd3-efe9271e4c96
md"The following d is a replication of the data in ThinkBayes2 to test my functions"

# ╔═╡ 36a22839-995d-4f9d-b0a4-743e66d5c864
d = [0.1285542 , 2.95534628, 0.07899838, 2.39963209, 0.22675277,
       3.4159887 , 7.78133028, 1.93757717, 9.03248382, 5.75612021]

# ╔═╡ 55a7646f-69ef-4d06-ab08-bb7d86c43fb2
function update_weibull(prior, data)
	wd_func(l, k) = prod([pdf(Weibull(k, l), x) for x in data])
	likelihood = outer(wd_func, index(prior), columns(prior))
	prior * likelihood
end

# ╔═╡ 693f95b6-82d4-4862-aaee-cfbd905a8c1e
posterior = update_weibull(prior, d);

# ╔═╡ 0d59c956-7125-4d9d-86a7-51f55869b462
contour(posterior, size=(800, 500), title="Posterior joint distribution of weibull parameters", xaxis=("λ"), yaxis=("κ"))

# ╔═╡ 87f1970f-93e4-4839-af25-7c9ccc44dfcc
md"## Marginal Distributions"

# ╔═╡ 74fc9c1d-affb-486d-b57a-f51cb723943e
begin
	posterior_lam = marginal(posterior, 1)
	posterior_k = marginal(posterior, 2)
end;

# ╔═╡ c3c534ff-44ae-4dbc-9d64-e1eda361ea20
begin
	plot(posterior_lam, label="λ")
	vline!([3])
end

# ╔═╡ 4fde7da5-8af1-4ad2-9d93-21e726404cb0
begin
	plot(posterior_k, label="κ")
	vline!([0.8])
end

# ╔═╡ 5f18fac1-ca3f-4da7-8b5b-29206edc8a8f
credible_interval(posterior_lam, 0.9)

# ╔═╡ 792530e7-0530-4780-a920-9fe8ac63b478
credible_interval(posterior_k, 0.9)

# ╔═╡ a9c8c1da-daa4-44a2-8d08-cd2af7ff5307
md"## Incomplete Data"

# ╔═╡ Cell order:
# ╠═11bfe214-db8f-11ec-24d9-21c6b862ea54
# ╠═a0bf7ab3-541d-4182-ba67-5de432a53ff9
# ╠═b240adda-091b-45ac-878d-302f7b18d610
# ╟─f1824111-27c6-4027-a3b3-0552f2a154d4
# ╠═5e16f035-5003-4b6e-851b-79eb47716120
# ╠═90b7bc4c-9cd5-4ecc-b4f6-a3e9c70072b0
# ╠═9e7c28b0-b11e-4800-bf51-c63db77bf4c0
# ╠═69cf8c0d-7faf-4a36-8da3-96d2588e029a
# ╠═1cc86501-f32c-408d-a1db-693e51a996ea
# ╠═0916bfe2-447f-4a7e-9d27-6c5553ddbabf
# ╠═954da934-c9c1-4eb5-a7fd-449eb4ecd44c
# ╠═edc59f76-d891-4388-ae1c-bcab84c44c65
# ╟─ac201ad5-cace-43d9-8dd3-efe9271e4c96
# ╠═36a22839-995d-4f9d-b0a4-743e66d5c864
# ╠═55a7646f-69ef-4d06-ab08-bb7d86c43fb2
# ╠═693f95b6-82d4-4862-aaee-cfbd905a8c1e
# ╠═0d59c956-7125-4d9d-86a7-51f55869b462
# ╟─87f1970f-93e4-4839-af25-7c9ccc44dfcc
# ╠═74fc9c1d-affb-486d-b57a-f51cb723943e
# ╠═c3c534ff-44ae-4dbc-9d64-e1eda361ea20
# ╠═4fde7da5-8af1-4ad2-9d93-21e726404cb0
# ╠═5f18fac1-ca3f-4da7-8b5b-29206edc8a8f
# ╠═792530e7-0530-4780-a920-9fe8ac63b478
# ╟─a9c8c1da-daa4-44a2-8d08-cd2af7ff5307
