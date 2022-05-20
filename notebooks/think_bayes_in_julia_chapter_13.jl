### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 4207b7f9-775a-4358-909e-32c217362770
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 99f426c0-d54e-11ec-2e4c-a3fb9fe71e8c
using DataFrames, CSV, Plots, Distributions

# ╔═╡ 92d83875-5ad2-4d6d-9122-e6a98464f2cb
df = DataFrame(CSV.File(download("https://github.com/AllenDowney/ThinkBayes2/raw/master/data/drp_scores.csv"), missingstring=["NA"], delim='\t', header=22, skipto=23));

# ╔═╡ 6c04a8f8-3fd4-4a40-87de-ff37160ddcb0
gdf = groupby(df, "Treatment");

# ╔═╡ 72eb658b-4592-424e-b709-230ec8770191
md"Now, make a cdf from the responses in each group"

# ╔═╡ ad71941f-c9a3-4180-844f-ddb1faa59f53
cdf_groups = combine(gdf, :Response => cdf_from_seq);

# ╔═╡ a445cc70-fe1f-47c9-a324-18ddcc7c2881
md"And make a dictionary of the results"

# ╔═╡ 6e616132-37a8-495c-9374-ce4e28f506e1
#responses = Dict([(k, cdf_groups[findfirst(==(k), cdf_groups.Treatment), "Response_cdf_from_seq"]) for k in collect(cdf_groups.Treatment)]);
responses = collect_vals(gdf, "Treatment", "Response")

# ╔═╡ 84691d70-de1b-4839-9969-48b7fc7322ce
begin
	plot(xaxis=("Score"), yaxis=("CDF"))
	for k in keys(responses)
		plot!(cdf_from_seq(responses[k]), label=k)
	end
	plot!()
end

# ╔═╡ f6b3aa0f-78fc-46b4-8e1e-4f5451365022
md"## Estimating Parameters"

# ╔═╡ 53c2fcb6-c70a-4edf-aa75-21775a405355
prior_mu = pmf_from_seq(range(20, 80, 101));

# ╔═╡ b348b226-3b93-4ce6-af5c-ca9c47af85f7
prior_sigma = pmf_from_seq(range(5, 30, 101));

# ╔═╡ b70f49c8-b713-47a8-8499-3c0f60e256cf
prior = make_joint(*, prior_mu, prior_sigma);

# ╔═╡ 642b575c-55fd-43bd-ab17-50e6cc8093b6
begin
	val_map = collect_vals(gdf, "Treatment", "Response")
	data = val_map["Control"]
end

# ╔═╡ 8d00d242-20e6-49e7-b5c5-21b3a2dfbff1
md"## Likelihood"

# ╔═╡ afa9c807-552f-4efa-89ea-1afb105fbd0b
begin
	mk_dense(x, y) = pdf.(Normal(x, y), data)
	densities = outer(mk_dense, prior.xs, prior.ys)
	likelihood = prod.(densities)
	size(likelihood)
end

# ╔═╡ 3f175a6d-a62f-4ece-ba52-fdea28c3e482
function update_norm(prior, data)
	mk_dense(x, y) = pdf.(Normal(x, y), data)
	densities = outer(mk_dense, prior.xs, prior.ys)
	likelihood = prod.(densities)
	#normalize(prior.M .* likelihood)
	prior * likelihood
end

# ╔═╡ e283ac78-ce8f-4c62-8bc0-47502edf18d9
posterior_control = update_norm(prior, val_map["Control"]);

# ╔═╡ 228ff1e9-5b41-46c6-afa1-206b137aa0b1
posterior_treated = update_norm(prior, val_map["Treated"]);

# ╔═╡ fcf15bc1-11f7-484e-88b7-6baf907a70e6
begin
	plot()
	visualize_joint!(posterior_treated, alpha=1.0, is_contour=true, c=:blues)
	visualize_joint!(posterior_control, alpha=1.0, is_contour=true, c=:reds)
	plot!()
end

# ╔═╡ 9e80e64d-0961-4f32-91d9-0ff8c12b859f
plotly()

# ╔═╡ c41b015a-b70f-4201-961c-1f501854aeb2
surface(posterior_treated + posterior_control)

# ╔═╡ 632714d4-82b8-453e-9727-74a71a72a3fc
md"## Posterior Marginal Distributions"

# ╔═╡ 293a2adb-a02b-4364-8121-c8fc8502ca81
pmf_mean_control = ThinkBayes.marginal(posterior_control, 1);

# ╔═╡ 2c3a1ddd-370e-4e76-b955-6cacc07116b3
pmf_mean_treated = ThinkBayes.marginal(posterior_treated, 1);

# ╔═╡ 750182de-1170-44cb-8c5f-2d94eb1a9988
begin
	plot(pmf_mean_control, label="Control")
	plot!(pmf_mean_treated, label="Treated")
end

# ╔═╡ 9b78cd80-57b2-4b11-a4e8-b7dc48af7d11
prob_gt(pmf_mean_treated, pmf_mean_control)

# ╔═╡ a99563ad-53c5-4a71-884d-19ae56b78c93
md"## Distribution of Differences"

# ╔═╡ 19e1c017-4b4e-438f-b485-53068fbe10e7
pmf_diff = sub_dist(pmf_mean_treated, pmf_mean_control);

# ╔═╡ e7dde5db-a9a7-4eb0-8bdb-92c58abaf78c
length(values(pmf_diff))

# ╔═╡ 96d4b76b-b0b7-4048-8b4a-19c3ea8d7250
plot(pmf_diff, xaxis=("Difference in population means"), yaxis=("PDF"))

# ╔═╡ 29c0003b-2d70-43b3-86bf-bf82606f8dba
cdf_diff = make_cdf(pmf_diff);

# ╔═╡ ed61c2a4-2bba-4878-90e0-a05123ec3ec8
plot(cdf_diff, xaxis=("Difference in population means"), yaxis=("PDF"))

# ╔═╡ 36a582fc-3d19-4d39-82bf-851bc46233c6
mean(pmf_diff)

# ╔═╡ d374722d-c468-46f5-adc1-50a04ce43df1
credible_interval(pmf_diff, 0.9)

# ╔═╡ b0887200-149f-43cf-bc58-0312252eb63c
md"## Using Summary Statistics"

# ╔═╡ ac474237-5b83-46aa-af2a-58756eb8d74c
begin
	mu = 42
	sigma = 17
end

# ╔═╡ 27308e14-305c-42ed-9138-8f3c8e85d021
begin
	n = 20
	m = 41
	s = 18
end

# ╔═╡ 6addd7af-96a1-4a45-b98a-66506ca75799
dist_m = Normal(mu, sigma/√n)

# ╔═╡ e9e6daf7-811e-4347-b0a8-a885060ac82b
like1 = pdf(dist_m, m)

# ╔═╡ 61cb6cb9-d270-43d1-a262-56b1c3122464
t = n * s^2 / sigma^2

# ╔═╡ 6820f9e3-535e-4447-9b4c-22f39b41d7c7
dist_s = Chisq(n-1)

# ╔═╡ c9dfa5cc-c7c7-4876-9121-2bfe6a0e9aa3
like2 = pdf(dist_s, t)

# ╔═╡ f7c3bb84-7ac9-48a5-9013-4bd83c148b04
like = like1 * like2

# ╔═╡ 11ceaf81-463a-493f-96dc-693dbf3a35e7
md"## Update with Summary Statistics"

# ╔═╡ f18042e9-a775-4bb0-b87f-a44fa4ec532d
summary = Dict([(name, (length(values(r)), mean(r), std(r))) for (name, r) in    pairs(responses)]) 

# ╔═╡ 227df8be-b546-442f-a998-447cefb8992e
typeof(responses["Treated"])

# ╔═╡ Cell order:
# ╠═99f426c0-d54e-11ec-2e4c-a3fb9fe71e8c
# ╠═4207b7f9-775a-4358-909e-32c217362770
# ╠═92d83875-5ad2-4d6d-9122-e6a98464f2cb
# ╠═6c04a8f8-3fd4-4a40-87de-ff37160ddcb0
# ╟─72eb658b-4592-424e-b709-230ec8770191
# ╠═ad71941f-c9a3-4180-844f-ddb1faa59f53
# ╟─a445cc70-fe1f-47c9-a324-18ddcc7c2881
# ╠═6e616132-37a8-495c-9374-ce4e28f506e1
# ╠═84691d70-de1b-4839-9969-48b7fc7322ce
# ╟─f6b3aa0f-78fc-46b4-8e1e-4f5451365022
# ╠═53c2fcb6-c70a-4edf-aa75-21775a405355
# ╠═b348b226-3b93-4ce6-af5c-ca9c47af85f7
# ╠═b70f49c8-b713-47a8-8499-3c0f60e256cf
# ╠═642b575c-55fd-43bd-ab17-50e6cc8093b6
# ╟─8d00d242-20e6-49e7-b5c5-21b3a2dfbff1
# ╠═afa9c807-552f-4efa-89ea-1afb105fbd0b
# ╠═3f175a6d-a62f-4ece-ba52-fdea28c3e482
# ╠═e283ac78-ce8f-4c62-8bc0-47502edf18d9
# ╠═228ff1e9-5b41-46c6-afa1-206b137aa0b1
# ╠═fcf15bc1-11f7-484e-88b7-6baf907a70e6
# ╠═9e80e64d-0961-4f32-91d9-0ff8c12b859f
# ╠═c41b015a-b70f-4201-961c-1f501854aeb2
# ╟─632714d4-82b8-453e-9727-74a71a72a3fc
# ╠═293a2adb-a02b-4364-8121-c8fc8502ca81
# ╠═2c3a1ddd-370e-4e76-b955-6cacc07116b3
# ╠═750182de-1170-44cb-8c5f-2d94eb1a9988
# ╠═9b78cd80-57b2-4b11-a4e8-b7dc48af7d11
# ╟─a99563ad-53c5-4a71-884d-19ae56b78c93
# ╠═19e1c017-4b4e-438f-b485-53068fbe10e7
# ╠═e7dde5db-a9a7-4eb0-8bdb-92c58abaf78c
# ╠═96d4b76b-b0b7-4048-8b4a-19c3ea8d7250
# ╠═29c0003b-2d70-43b3-86bf-bf82606f8dba
# ╠═ed61c2a4-2bba-4878-90e0-a05123ec3ec8
# ╠═36a582fc-3d19-4d39-82bf-851bc46233c6
# ╠═d374722d-c468-46f5-adc1-50a04ce43df1
# ╟─b0887200-149f-43cf-bc58-0312252eb63c
# ╠═ac474237-5b83-46aa-af2a-58756eb8d74c
# ╠═27308e14-305c-42ed-9138-8f3c8e85d021
# ╠═6addd7af-96a1-4a45-b98a-66506ca75799
# ╠═e9e6daf7-811e-4347-b0a8-a885060ac82b
# ╠═61cb6cb9-d270-43d1-a262-56b1c3122464
# ╠═6820f9e3-535e-4447-9b4c-22f39b41d7c7
# ╠═c9dfa5cc-c7c7-4876-9121-2bfe6a0e9aa3
# ╠═f7c3bb84-7ac9-48a5-9013-4bd83c148b04
# ╟─11ceaf81-463a-493f-96dc-693dbf3a35e7
# ╠═f18042e9-a775-4bb0-b87f-a44fa4ec532d
# ╠═227df8be-b546-442f-a998-447cefb8992e
