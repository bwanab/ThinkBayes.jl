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
responses = Dict([(k, cdf_groups[findfirst(==(k), cdf_groups.Treatment), "Response_cdf_from_seq"]) for k in collect(cdf_groups.Treatment)]);

# ╔═╡ 84691d70-de1b-4839-9969-48b7fc7322ce
begin
	plot(xaxis=("Score"), yaxis=("CDF"))
	for k in keys(responses)
		plot!(responses[k], label=k)
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
end

# ╔═╡ e283ac78-ce8f-4c62-8bc0-47502edf18d9
posterior_control = update_norm(prior, val_map["Control"])

# ╔═╡ 228ff1e9-5b41-46c6-afa1-206b137aa0b1
posterior_treated = update_norm(prior, val_map["Treated"])

# ╔═╡ fcf15bc1-11f7-484e-88b7-6baf907a70e6
begin
	plot()
	visualize_joint!(posterior_treated, xs=prior.xs, ys=prior.ys, alpha=1.0, is_contour=true, normalize=true, c=:blues)
	visualize_joint!(posterior_control, xs=prior.xs, ys=prior.ys, alpha=1.0, is_contour=true, normalize=true, c=:reds)
	plot!()
end

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
