### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 15b0a8d0-ce4f-11ec-2954-3d08b2e754ed
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ cafc7577-2a2a-43da-8514-20569cc30160
using DataFrames, CSV, Distributions, Plots

# ╔═╡ 8d10dd3f-4580-49d6-857e-885cf6077266
md"# Chapter 12 - Classification"

# ╔═╡ 5dd18246-3ce4-4610-8e7d-4380f8959a4d
df = DataFrame(CSV.File(download("https://github.com/allisonhorst/palmerpenguins/raw/master/inst/extdata/penguins_raw.csv"), missingstring=["NA"]))

# ╔═╡ 77fdcfe8-956c-466c-a0fe-32f324a089cb
dropmissing!(df, ["Flipper Length (mm)", "Culmen Length (mm)"])

# ╔═╡ 20d2c2ff-bfd9-4cc3-98a5-0886559bf374
df.Species2 = first.(split.(df.Species))

# ╔═╡ cfbc5e96-3dfc-4ca2-8ecd-bc9226eaeb18
function make_cdf_map(df, colname; by="Species2")
	gd = groupby(df, [by]);
	d = [(first(g).Species2, g[!, colname]) for g in gd]
	Dict([(s, make_cdf(pmf_from_seq(sort([v for v in vals if v !== missing])))) for (s, vals) in d])
end

# ╔═╡ 6ea82220-ebc9-4fc7-ab6c-0f79d2cfd5e4
cdf_map = make_cdf_map(df, "Culmen Length (mm)");

# ╔═╡ c1a235a7-8668-46a5-a3b4-dc752b7c9110
function plot_cdfs(df, colname; by="Species2")
	function p(i, k, cdf_map)
		if i == 1
			return plot(cdf_map[k], label=k)
		else
			return plot!(cdf_map[k], label=k)
		end
	end
	cdf_map = make_cdf_map(df, colname, by=by)
	last([p(i, k, cdf_map) for (i, k) in enumerate(keys(cdf_map))])
end

# ╔═╡ 3d36c6b0-80dc-48ad-8794-2d9c3975fa30
plot_cdfs(df, "Culmen Length (mm)")

# ╔═╡ 16aef67d-c069-4e47-83d1-b2ac1625d18c
plot_cdfs(df, "Flipper Length (mm)")

# ╔═╡ 2f014da1-2f5a-476d-b711-6a3e156a62d1
plot_cdfs(df, "Culmen Depth (mm)")

# ╔═╡ 3ad37fe8-5e3c-480f-80cf-f19a515d1810
plot_cdfs(df, "Body Mass (g)")

# ╔═╡ a6ca9f64-d63a-42a5-8956-e6d01d6b62a6
md"## Normal Models"

# ╔═╡ e875a719-28ef-4461-940a-a6129bb385e1
function make_norm_map(df, colname; by="Species2")
	m = make_cdf_map(df, colname)
	Dict([(k, Normal(mean(v), std(v))) for (k, v) in pairs(m)])
end

# ╔═╡ 89483d4e-eed0-4057-9709-8f18e25caa65
flipper_map = make_norm_map(df, "Flipper Length (mm)")

# ╔═╡ 2d8463d3-49e3-48eb-8459-a7f5b0739e52
begin
	data = 193
	pdf(flipper_map["Adelie"], data)
end

# ╔═╡ 22046bc9-ad93-4e70-9656-a8d21df9ccbc
begin
	hypos = sort(collect(keys(flipper_map)))
	likelihood = [pdf(flipper_map[hypo], data) for hypo in hypos]
	hypos, likelihood
end

# ╔═╡ a25d5245-e316-4937-a7c9-4603722bc043
md"## The Update"

# ╔═╡ f82bbc9f-73e5-44c4-a663-0a7dd312d050
prior = pmf_from_seq(hypos)

# ╔═╡ c57885aa-0606-4f82-a000-e32530c0deb6
posterior = prior * likelihood

# ╔═╡ d7d818c4-d824-41e9-81a3-3b4b6530a5de
function update_penguin(prior, data, norm_map)
	hypos = values(prior)
	likelihood = [pdf(norm_map[hypo], data) for hypo in hypos]
	prior * likelihood
end

# ╔═╡ 59bbcbf0-2929-4b06-8976-1be71bfab901
posterior1 = update_penguin(prior, 193, flipper_map)

# ╔═╡ 22d863ba-bea3-4756-98b5-eab932e0f0ea
culmen_map = make_norm_map(df, "Culmen Length (mm)")

# ╔═╡ 6a3a2855-4d7e-43b1-9af4-87a6d5a8e431
posterior2 = update_penguin(prior, 48, culmen_map)

# ╔═╡ 08be9ec4-4e51-4f0a-9b89-3f3aa78c282d
function update_naive(prior, data_seq, norm_maps)
	posterior = copy(prior)
	for (data, norm_map) in zip(data_seq, norm_maps)
		posterior = update_penguin(posterior, data, norm_map)
	end
	posterior
end

# ╔═╡ e9eac7e7-b962-43fe-96a1-12952518b42d
begin
	col_names = ["Flipper Length (mm)", "Culmen Length (mm)"]
	data_seq = [193, 48]
	norm_maps = [flipper_map, culmen_map]
	posterior3 = update_naive(prior, data_seq, norm_maps)
end

# ╔═╡ 6d79a183-e982-44bc-a97b-21a279dbfdbf
max_prob(posterior3)

# ╔═╡ 076d1f45-2b1b-445d-905c-fd6ea3e0fe65
begin
	posteriors = [update_naive(prior, values(df[r, col_names]), norm_maps) for r in 1:nrow(df) if !any(isequal.(missing, values(df[r, col_names])))]
	probs = max_prob.(posteriors)
end

# ╔═╡ 3845d82a-0777-48c3-acdf-2fc9b88ab3fb
df.Classification = max_prob.(posteriors)

# ╔═╡ 1a0f0668-77e0-454e-80e8-84ae2e06ade7
same = df.Species2 .== df.Classification

# ╔═╡ fe3db091-c725-41ec-85f1-df62b19b73eb
sum(same) / first(size(posteriors))

# ╔═╡ a3f71fa5-be19-4a1a-8ba0-8573a87a3a23
function accuracy(df)
	valid = nrow(df)
	same = df.Species2 .== df.Classification
	sum(same) / valid
end

# ╔═╡ 3935b12a-c50b-46b7-b64b-8a11a7215335
accuracy(df)

# ╔═╡ eeafc3d6-9bc2-415a-ab40-a719ff9302f5
md"## Joint Distributions"

# ╔═╡ 978f66ce-b020-4857-8e6e-f0f431434d66
function scatterplot(df, var1, var2; by="Species2", all_secondary=false)
	function p(i, d1, d2)
		if i == 1 && !all_secondary
			return scatter(d1[2], d2[2], label=d1[1])
		else
			return scatter!(d1[2], d2[2], label=d1[1])
		end
	end
	gd = groupby(df, [by]);
	d1 = [(first(g).Species2, collect(g[!, var1])) for g in gd]
	d2 = [(first(g).Species2, collect(g[!, var2])) for g in gd]
	last([p(i, x, y) for (i, (x, y)) in enumerate(zip(d1,d2))])
end

# ╔═╡ 3d3e0438-1311-44ed-9f2e-7259383cb4d0
begin
	var1 = col_names[1]
	var2 = col_names[2]
	scatterplot(df, var1, var2)
end

# ╔═╡ 51ff7c73-d304-440d-aba4-61743f97bb3b
function make_pmf_norm(dist, sigmas=3, n=101)
	mu, sigma = mean(dist), std(dist)
	low = mu - sigmas * sigma
	high = mu + sigmas * sigma
	qs = LinRange(low, high, n)
	pmf_from_dist(qs, dist)
end

# ╔═╡ a725825d-0c60-404f-8c33-380c96cd2b62
begin
	joint_map = Dict()
	for species in hypos
		pmf1 = make_pmf_norm(flipper_map[species])
		pmf2 = make_pmf_norm(culmen_map[species])
		joint_map[species] = make_joint(*, pmf1, pmf2)
	end
end

# ╔═╡ 2917ae4a-72f6-4ff5-83b5-83b990553f12
function plot_joints(joint_map)
	function p(i, k)
		if i == 1
			return visualize_joint(joint_map[k])
		else
			return ThinkBayes.visualize_joint!(joint_map[k])
		end
	end
	last([p(i, k) for (i, k) in enumerate(keys(joint_map))])
end

# ╔═╡ 823958fd-d86b-4270-89e4-4ccb47575789
md"__This visualization is an enhancement opportunity!__"

# ╔═╡ 2de4cf7e-c7ae-4db5-8af4-8f49992c3a31
begin
	visualize_joint(joint_map["Adelie"], is_contour=true)
	visualize_joint!(joint_map["Gentoo"], is_contour=true)
	visualize_joint!(joint_map["Chinstrap"], is_contour=true)
	scatterplot(df, col_names[1], col_names[2], all_secondary=true)
end

# ╔═╡ c0526259-d963-4d6f-9fdf-cd3ed09deb1f
md"## Multivariate Normal Distribution"

# ╔═╡ f65ab28d-c2d3-4a92-a7a5-d19bddbbc574
features = hcat(df[!, var1], df[!, var2]);

# ╔═╡ 77af8680-7506-4bda-a3f5-aca36aa3feaa
means=vec(mean(features, dims=1))

# ╔═╡ 5cdb2f13-7c15-4f92-bd15-9e44ea42c0e5
covar = cov(features)

# ╔═╡ 1a49eb60-1cc3-403e-a24c-9ecdaf4052f0
MvNormal(means, covar)

# ╔═╡ 61ad48bf-d029-4ac1-a23e-ef0b272cb572
function make_multinorm_map(df, colnames; by="Species2")
	gd = groupby(df, [by])
	multinorm_map = Dict()
	for g in gd
    	species = first(g).Species2
	    features = reduce(hcat, [g[!, c] for c in colnames])
		means = vec(mean(features, dims=1))
		covar = cov(features)
		multinorm_map[species] = MvNormal(means, covar)
	end
	multinorm_map
end

# ╔═╡ 4346993d-a902-4963-a879-4b115ce978a0
multinorm_map = make_multinorm_map(df, col_names)

# ╔═╡ 75ce51d1-5a5a-4e7c-96bd-33d859290f4c
md"## Visualizing a Multivariate Normal Distribution"

# ╔═╡ 7896da48-3c4b-4770-a3a0-6ddec858231a
begin
	norm1 = flipper_map["Adelie"]
	norm2 = culmen_map["Adelie"]
	multinorm = multinorm_map["Adelie"]
end

# ╔═╡ 234b3c23-456a-40c7-96dc-b4072570d38d
begin
	pmf1 = make_pmf_norm(norm1)
	pmf2 = make_pmf_norm(norm2)
end;

# ╔═╡ aa7d06a9-99d6-424a-867f-6bc2f7861680
begin
	mn_pdf(x, y) = pdf(multinorm, [x, y])
	densities = outer(mn_pdf, values(pmf1), values(pmf2))
end;

# ╔═╡ 3c7511ee-fd7c-4285-a312-da903f052563
visualize_joint(densities, xs=values(pmf1), ys=values(pmf2), alpha=1.0, is_contour=true)

# ╔═╡ 1edbdb38-5182-4ece-8c36-5c4cd6511af4
function make_multi_joint(norm1, norm2, multinorm)
	pmf1 = make_pmf_norm(norm1)
	pmf2 = make_pmf_norm(norm2)
	mn_pdf(x, y) = pdf(multinorm, [x, y])
	(outer(mn_pdf, values(pmf1), values(pmf2)), pmf1, pmf2)
end

# ╔═╡ 45f8aa52-ffd7-4da7-8c00-b8ff1ec66272
begin
	viz = Dict()
	for (i, species) in enumerate(hypos)
		norm1 = flipper_map[species]
		norm2 = culmen_map[species]
		multinorm = multinorm_map[species]
		M, pmf1, pmf2 = make_multi_joint(norm1, norm2, multinorm)
		viz[species] = (M, values(pmf1), values(pmf2))
	end
	function vj(s, fst)
		densities, xs, ys = viz[s]
		if fst
			visualize_joint(densities, xs=xs, ys=ys, alpha=1.0, is_contour=true)
		else
			visualize_joint!(densities, xs=xs, ys=ys, alpha=1.0, is_contour=true)
		end
	end
	vj("Adelie", true)
	vj("Gentoo", false)
	vj("Chinstrap", false)
	scatterplot(df, var1, var2, all_secondary=true)
end

# ╔═╡ 357bf6ab-51c8-47c4-846f-e74881107c17
md"## A Less Naive Classifier"

# ╔═╡ f4d84625-1cdf-421d-8951-0a3b915d649c
update_penguin(prior, [193, 48], multinorm_map)

# ╔═╡ 494a72d7-536a-404f-b79b-b76df69e89d6
begin
	flr(x) = floor(Int, x)
	vect(t::Tuple) = [x for x in t]
	posteriors2 = [update_penguin(prior, vect(flr.(values(df[r, col_names]))), multinorm_map) for r in 1:nrow(df) if !any(isequal.(missing, values(df[r, col_names])))]
	probs2 = max_prob.(posteriors2)
	df.Classification = max_prob.(posteriors2)
	accuracy(df)
end

# ╔═╡ Cell order:
# ╟─8d10dd3f-4580-49d6-857e-885cf6077266
# ╠═cafc7577-2a2a-43da-8514-20569cc30160
# ╠═15b0a8d0-ce4f-11ec-2954-3d08b2e754ed
# ╠═5dd18246-3ce4-4610-8e7d-4380f8959a4d
# ╠═77fdcfe8-956c-466c-a0fe-32f324a089cb
# ╠═20d2c2ff-bfd9-4cc3-98a5-0886559bf374
# ╠═cfbc5e96-3dfc-4ca2-8ecd-bc9226eaeb18
# ╠═6ea82220-ebc9-4fc7-ab6c-0f79d2cfd5e4
# ╠═c1a235a7-8668-46a5-a3b4-dc752b7c9110
# ╠═3d36c6b0-80dc-48ad-8794-2d9c3975fa30
# ╠═16aef67d-c069-4e47-83d1-b2ac1625d18c
# ╠═2f014da1-2f5a-476d-b711-6a3e156a62d1
# ╠═3ad37fe8-5e3c-480f-80cf-f19a515d1810
# ╟─a6ca9f64-d63a-42a5-8956-e6d01d6b62a6
# ╠═e875a719-28ef-4461-940a-a6129bb385e1
# ╠═89483d4e-eed0-4057-9709-8f18e25caa65
# ╠═2d8463d3-49e3-48eb-8459-a7f5b0739e52
# ╠═22046bc9-ad93-4e70-9656-a8d21df9ccbc
# ╟─a25d5245-e316-4937-a7c9-4603722bc043
# ╠═f82bbc9f-73e5-44c4-a663-0a7dd312d050
# ╠═c57885aa-0606-4f82-a000-e32530c0deb6
# ╠═d7d818c4-d824-41e9-81a3-3b4b6530a5de
# ╠═59bbcbf0-2929-4b06-8976-1be71bfab901
# ╠═22d863ba-bea3-4756-98b5-eab932e0f0ea
# ╠═6a3a2855-4d7e-43b1-9af4-87a6d5a8e431
# ╠═08be9ec4-4e51-4f0a-9b89-3f3aa78c282d
# ╠═e9eac7e7-b962-43fe-96a1-12952518b42d
# ╠═6d79a183-e982-44bc-a97b-21a279dbfdbf
# ╠═076d1f45-2b1b-445d-905c-fd6ea3e0fe65
# ╠═3845d82a-0777-48c3-acdf-2fc9b88ab3fb
# ╠═1a0f0668-77e0-454e-80e8-84ae2e06ade7
# ╠═fe3db091-c725-41ec-85f1-df62b19b73eb
# ╠═a3f71fa5-be19-4a1a-8ba0-8573a87a3a23
# ╠═3935b12a-c50b-46b7-b64b-8a11a7215335
# ╟─eeafc3d6-9bc2-415a-ab40-a719ff9302f5
# ╠═978f66ce-b020-4857-8e6e-f0f431434d66
# ╠═3d3e0438-1311-44ed-9f2e-7259383cb4d0
# ╠═51ff7c73-d304-440d-aba4-61743f97bb3b
# ╠═a725825d-0c60-404f-8c33-380c96cd2b62
# ╠═2917ae4a-72f6-4ff5-83b5-83b990553f12
# ╟─823958fd-d86b-4270-89e4-4ccb47575789
# ╠═2de4cf7e-c7ae-4db5-8af4-8f49992c3a31
# ╟─c0526259-d963-4d6f-9fdf-cd3ed09deb1f
# ╠═f65ab28d-c2d3-4a92-a7a5-d19bddbbc574
# ╠═77af8680-7506-4bda-a3f5-aca36aa3feaa
# ╠═5cdb2f13-7c15-4f92-bd15-9e44ea42c0e5
# ╠═1a49eb60-1cc3-403e-a24c-9ecdaf4052f0
# ╠═61ad48bf-d029-4ac1-a23e-ef0b272cb572
# ╠═4346993d-a902-4963-a879-4b115ce978a0
# ╟─75ce51d1-5a5a-4e7c-96bd-33d859290f4c
# ╠═7896da48-3c4b-4770-a3a0-6ddec858231a
# ╠═234b3c23-456a-40c7-96dc-b4072570d38d
# ╠═aa7d06a9-99d6-424a-867f-6bc2f7861680
# ╠═3c7511ee-fd7c-4285-a312-da903f052563
# ╠═1edbdb38-5182-4ece-8c36-5c4cd6511af4
# ╠═45f8aa52-ffd7-4da7-8c00-b8ff1ec66272
# ╟─357bf6ab-51c8-47c4-846f-e74881107c17
# ╠═f4d84625-1cdf-421d-8951-0a3b915d649c
# ╠═494a72d7-536a-404f-b79b-b76df69e89d6
