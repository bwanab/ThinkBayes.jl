### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 4207b7f9-775a-4358-909e-32c217362770
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 99f426c0-d54e-11ec-2e4c-a3fb9fe71e8c
using DataFrames, CSV, Plots, Distributions, PlutoUI

# ╔═╡ a6de7ccf-6512-4c38-b577-ade3fab54c27
TableOfContents()

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
responses = collect_vals(gdf, "Treatment", "Response");

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
end;

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

# ╔═╡ 013be8cb-989d-48af-b028-63159d3da220
begin
	contour(posterior_treated, c=:blues)
	contour!(posterior_control, c=:reds)
end

# ╔═╡ 9e80e64d-0961-4f32-91d9-0ff8c12b859f
plotly()

# ╔═╡ c41b015a-b70f-4201-961c-1f501854aeb2
surface(posterior_treated + posterior_control)

# ╔═╡ 632714d4-82b8-453e-9727-74a71a72a3fc
md"## Posterior Marginal Distributions"

# ╔═╡ 293a2adb-a02b-4364-8121-c8fc8502ca81
pmf_mean_control = marginal(posterior_control, 1);

# ╔═╡ 2c3a1ddd-370e-4e76-b955-6cacc07116b3
pmf_mean_treated = marginal(posterior_treated, 1);

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
n₁, m₁, s₁ = summary["Control"]

# ╔═╡ 9a5512de-cc0c-4f6d-a715-cbf9b7aea978
likefunc(m,s) = pdf(Normal(m, s/√n₁), m₁)

# ╔═╡ 048f526a-376f-43f5-ad3e-352e23066414
like₁ = outer(likefunc, prior.xs, prior.ys);

# ╔═╡ f8e88d6d-653b-4e8e-9d26-28817ccbf13c
size(like₁)

# ╔═╡ 1b19b45e-3697-4be3-ab53-95f4c4282a3d
begin
	like2func(m, s) = pdf(Chisq(n₁ - 1), n * s₁^2 / s^2)
	like₂ = outer(like2func, prior.xs, prior.ys)
	size(like₂)
end;

# ╔═╡ e411b5cf-91c6-4d13-8a6e-dc7875a30223
function update_norm_summary(prior, data)
	n₁, m₁, s₁ = data
	like1func(m,s) = pdf(Normal(m, s/√n₁), m₁)
	like₁ = outer(like1func, prior.xs, prior.ys);
	like2func(m, s) = pdf(Chisq(n₁ - 1), n₁ * s₁^2 / s^2)
	like₂ = outer(like2func, prior.xs, prior.ys)
	prior * (like₁ .* like₂)
end

# ╔═╡ 34a07d31-e02d-49f7-b4b1-a258068ccf18
begin
	posterior_control2 = update_norm_summary(prior, summary["Control"])
	posterior_treated2 = update_norm_summary(prior, summary["Treated"])
end;

# ╔═╡ 28d6452f-d9ed-4961-8dc5-9f265b883976
begin
	contour(posterior_treated, c=:reds)
	contour!(posterior_control, c=:blues)
end

# ╔═╡ 439e9db4-2006-4ac8-8a8f-35d7a02f9664
md"## Comparing Marginals"

# ╔═╡ c8708e9b-885d-4725-ab8f-54ed4c70ae07
begin
	pmf_mean_control2 = marginal(posterior_control2, 1)
	pmf_mean_treated2 = marginal(posterior_treated2, 1)
end;

# ╔═╡ d8ad74bc-2986-4d66-bf79-9fa54af1b91f
begin
	plot()
	plot!(pmf_mean_control, label="Control1")
	plot!(pmf_mean_control2, label="Control2")
	plot!(pmf_mean_treated, label="Treated1")
	plot!(pmf_mean_treated2, label="Treated2")
	plot!()
end

# ╔═╡ 8f1c24bb-3517-4693-acaf-6b2f465890bf
md"## Proof By Simulation"

# ╔═╡ 4ff3434a-1805-47f3-8c15-4ad8a714ecff
dist = Normal(mu, sigma)

# ╔═╡ b4858b90-d4b2-49b2-90f2-f0068c3b9beb
n_samp = 20

# ╔═╡ e8e366f8-a831-4c33-82f7-6095af6369aa
begin
	samples = rand(dist, 1000, n_samp)
	size(samples)
end

# ╔═╡ 7f31c538-ad7f-4f1e-948c-5416f25e115d
sample_means = mean(samples, dims=2);

# ╔═╡ 2ce56d45-8574-4524-804b-92337c1d9300
begin
	low = mean(dist_m) - std(dist_m) * 3
	high = mean(dist_m) + std(dist_m) * 3
	pmf_m = pmf_from_dist(range(low, high, 101), dist_m)
end;

# ╔═╡ 72424251-063a-4e10-839b-10fbff4f2101
pmf_sample_means = kde_from_sample(vec(sample_means), low, high, 101);

# ╔═╡ 55de30dc-ce1c-4421-b08c-b6463f201704
begin
	plot(pmf_m, label="Theoretical distribution")
	plot!(pmf_sample_means, label="KDE of sample means")
end

# ╔═╡ c69230af-f832-4ef8-8b0c-fda95197e38f
md"## Checking Standard Deviation"

# ╔═╡ 8b59e536-7b84-4663-9788-7e8db4f63df3
sample_stds = std(samples, dims=2);

# ╔═╡ 8774d461-c81c-468d-ba13-eedc375a4b76
transformed = n_samp .* sample_stds .^2 ./ sigma^2

# ╔═╡ 67e8d1a4-ce04-411d-9099-6461e9952e60
dist_sq = Chisq(n_samp - 1)

# ╔═╡ 647aa9b2-5dd8-4aba-bb61-ff3d4bca5241
begin
	lowsq = 0
	highsq = mean(dist_sq) + std(dist_sq) * 4
	pmf_sq = pmf_from_dist(range(lowsq, highsq, 101), dist_sq)
end;

# ╔═╡ 49194b60-5b95-4e4e-8f8d-fadb69a5e27c
pmf_sample_stds = kde_from_sample(vec(transformed), lowsq, highsq, 101);

# ╔═╡ 36bfea5d-3aff-49c0-bed6-d27424c603a5
begin
	plot(pmf_sq, label="Theoretical Distribution")
	plot!(pmf_sample_stds, label="KDE of sample std")
end

# ╔═╡ 03a99bf1-4029-4d8a-9be9-954a8cc8a44a
cor(sample_means, sample_stds)

# ╔═╡ 29d6f2eb-5a87-4ebe-99bf-9c3fc02c335e
md"""## Exercises
_exercise 13.1_
"""

# ╔═╡ 12b1ab62-f307-41cd-a193-3c378e435a80
begin
	pmf_std_control = marginal(posterior_control, 2)
	pmf_std_treated = marginal(posterior_treated, 2)
end;

# ╔═╡ 5bd617b3-5fbd-4a60-8b1b-27baaa5a9286
begin
	plot(pmf_std_control, label="control")
	plot!(pmf_std_treated, label="treated")
end

# ╔═╡ bbd53073-4d50-43cd-8d60-559af7880daf
prob_gt(pmf_std_control, pmf_std_treated)

# ╔═╡ ef0bbaf6-c822-41c8-972c-59368f5928c5
pmf_diff2 = sub_dist(pmf_std_control, pmf_std_treated);

# ╔═╡ ba73c7d0-bf14-4d08-8f91-6585ec77fa23
mean(pmf_diff2)

# ╔═╡ 71452494-7caf-48db-8e37-e5fa0ba9eaf0
credible_interval(pmf_diff2, 0.9)

# ╔═╡ 6449db62-e9c7-41ac-a40b-f0055c6ea30d
plot(pmf_diff2)

# ╔═╡ d095ac67-1ec2-4988-96ad-964bfd4465ea
md"_exercise 13.2_"

# ╔═╡ 9061b2f0-c7a2-424a-b5d6-892dd5feacb3
function sample_joint(joint, size)
	(vs, ps) = stack(joint)
	p = pmf_from_seq(vs, ps)
	rand(p, size)
end

# ╔═╡ 06a07103-9f69-4a0f-927f-26ddd91680fd
sample_treated = sample_joint(posterior_treated, 1000)

# ╔═╡ 7da27236-27fd-473d-bc52-1916c6a77952
sample_control = sample_joint(posterior_control, 1000)

# ╔═╡ 8f77739c-81d8-43c4-839f-1c80ea63761e
effect_size = [((u₁ - u₂)/((σ₁ + σ₂)/2)) for ((u₁, σ₁), (u₂, σ₂)) in zip(sample_treated, sample_control)]

# ╔═╡ 4af12148-36a4-469d-97f9-b4e142e242a4
pmf = kde_from_sample(effect_size, minimum(effect_size), maximum(effect_size), 101);

# ╔═╡ 688a9126-ebcf-4c3b-90c8-37526f06fb28
plot(pmf)

# ╔═╡ 520eb429-18c9-4257-b296-456b9c04e9a3
cdf = cdf_from_seq(effect_size);

# ╔═╡ 6d2138b1-857c-4409-bc4c-99c39f7d146c
plot(cdf)

# ╔═╡ 8e430f7a-372e-443e-9312-b189e5fe5e46
mean(cdf), mean(pmf)

# ╔═╡ ee8f32fa-82ee-4bb9-a15a-eeb029b8baf1
md"_exercise 13.3"

# ╔═╡ 7d2d821d-35e2-411b-95d0-4217ff6d6d65
md"""here's the easy, peasy solution

We know the mean is 80. 5 students got over 90 which means 20% of the students.

Find the std that gives just over 20% for ccdf of 90
"""

# ╔═╡ d9842689-1f69-4776-926b-71b8b842f391
mu1 = 81

# ╔═╡ 4fdd9be7-de78-498c-9648-39965a4f0bcd


# ╔═╡ f4d05f19-d140-449e-92c3-5a66a889d60f
pct_over_90 = 5 / 25

# ╔═╡ b7d3a3f8-1b9a-4238-b32a-7bc728fe01a6
for x in range(1, 20, 100)
	d = Normal(mu1, x)
	if ccdf(d, 90) > pct_over_90
		print("the std is about $(x)")
		break
	end
end

# ╔═╡ 9f526288-6c61-450b-aa3b-7c9c5e22f445
md"now the right way"

# ╔═╡ fd9bc2da-f062-4cef-93bc-10f932255bb3
hypos = range(1, 51, 101)

# ╔═╡ c4afb792-7b73-464c-86a4-cf14fd54b899
pgt90 = [ccdf(Normal(mu1, x), 90) for x in hypos]

# ╔═╡ d5805040-fa2f-4746-9f04-2d3f9adfed1c
likelihood_1 = [pdf(Binomial(25, x), 5) for x in pgt90];

# ╔═╡ 115e20cc-fadd-4fe2-b82a-1cd90cad2035
prior_1 = pmf_from_seq(hypos);

# ╔═╡ 2f58da97-7011-4818-83fa-edf16961e63a
posterior_1 = prior_1 * likelihood_1;

# ╔═╡ daf4d854-dc40-4ccb-8ca9-0019b319324c
plot(posterior_1)

# ╔═╡ 58aeb048-62f5-4287-bb27-7761a445fad4
max_prob(posterior_1)

# ╔═╡ 4fc53029-d832-4508-a9b3-096ce1547d5b
pgt60 = [ccdf(Normal(mu1, x), 60) for x in hypos]

# ╔═╡ d5e05822-16c3-4703-8615-efa3352135e6
likelihood_2 = pgt60 .^ 25

# ╔═╡ b0904a71-f8cd-4a14-88a3-a7cf55c5c660
plot(hypos, likelihood_2)

# ╔═╡ 2b96fd3b-2536-4bbd-9100-40249a2864bb
posterior_2 = prior_1 * (likelihood_1 .* likelihood_2);

# ╔═╡ c9cc9041-e279-4333-bbd0-e71b0aca56c7
begin
	plot(posterior_1, label="Posterior 1")
	plot!(posterior_2, label="Posterior 2")
end

# ╔═╡ a16c6b5f-6017-452f-a0ec-161983cf9323
mean(posterior_2)

# ╔═╡ d1909b2d-0ed6-42d8-b7b4-ad7245499e5d
credible_interval(posterior_2, 0.9)

# ╔═╡ 4c5df011-ae82-40a0-be74-c17ffdbdd3e8
md"_exercise 13.3_"

# ╔═╡ ac3b3c2d-4b24-42c8-8ccc-ce0e3e6705be
8.27/178

# ╔═╡ 3836fc2f-d930-4697-80a8-8a62f4c84ea7
function make_posterior(m, s)
	mqs = range(m -0.1, m+0.1, 101)
	prior_mu = pmf_from_seq(mqs)
	sqs = range(s - 0.1, s+0.1, 101)
	prior_sigma = pmf_from_seq(sqs)
	make_joint(*, prior_mu, prior_sigma)	
end

# ╔═╡ 5879cf23-46b9-4f67-b17c-2ea5f391fdae
begin
	men = (n=154407, mean=178, std=8.27)
	men_prior = make_posterior(men.mean, men.std)
end;

# ╔═╡ 284e36c3-0872-492f-9fdb-35bdd2b20cfd
posterior_men = update_norm_summary(men_prior, men);

# ╔═╡ cd34ebd3-9ebd-44a8-93d4-c7d9bc74cb48
contour(posterior_men)

# ╔═╡ 86c7253c-b10c-463e-8351-6be1d53e13d2
begin
	women = (n=254722, mean=163, std=7.75)
	women_prior = make_posterior(women.mean, women.std)
end;

# ╔═╡ 151508c6-3fa8-419b-8a60-2ce6df16b9f3
posterior_women = update_norm_summary(women_prior, women);

# ╔═╡ 4cafb3e8-4ec8-473b-8736-54806359cc52
contour(posterior_women)

# ╔═╡ 41f2e064-550d-484d-8514-3cde08959e26
function get_posterior_cv(joint::Joint)
	pmf_μ = marginal(joint, 1)
	pmf_σ = marginal(joint, 2)
	kde_from_pmf(div_dist(pmf_σ, pmf_μ), 101)
end

# ╔═╡ 886adea9-2578-42d5-8b55-713cb468a20a
begin
	pmf_cv_men = get_posterior_cv(posterior_men)
	pmf_cv_women = get_posterior_cv(posterior_women)
	plot(pmf_cv_men)
	plot!(pmf_cv_women)
	
end

# ╔═╡ c9377f27-b79c-43df-ab63-7359c9451749
begin
	ratio_cv = div_dist(pmf_cv_women, pmf_cv_men)
	max_prob(ratio_cv)
end

# ╔═╡ 37040a37-6ef6-4410-812f-9d5fe2662330
credible_interval(ratio_cv, 0.9)

# ╔═╡ Cell order:
# ╠═99f426c0-d54e-11ec-2e4c-a3fb9fe71e8c
# ╠═a6de7ccf-6512-4c38-b577-ade3fab54c27
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
# ╠═013be8cb-989d-48af-b028-63159d3da220
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
# ╠═9a5512de-cc0c-4f6d-a715-cbf9b7aea978
# ╠═048f526a-376f-43f5-ad3e-352e23066414
# ╠═f8e88d6d-653b-4e8e-9d26-28817ccbf13c
# ╠═1b19b45e-3697-4be3-ab53-95f4c4282a3d
# ╠═e411b5cf-91c6-4d13-8a6e-dc7875a30223
# ╠═34a07d31-e02d-49f7-b4b1-a258068ccf18
# ╠═28d6452f-d9ed-4961-8dc5-9f265b883976
# ╟─439e9db4-2006-4ac8-8a8f-35d7a02f9664
# ╠═c8708e9b-885d-4725-ab8f-54ed4c70ae07
# ╠═d8ad74bc-2986-4d66-bf79-9fa54af1b91f
# ╟─8f1c24bb-3517-4693-acaf-6b2f465890bf
# ╠═4ff3434a-1805-47f3-8c15-4ad8a714ecff
# ╠═b4858b90-d4b2-49b2-90f2-f0068c3b9beb
# ╠═e8e366f8-a831-4c33-82f7-6095af6369aa
# ╠═7f31c538-ad7f-4f1e-948c-5416f25e115d
# ╠═2ce56d45-8574-4524-804b-92337c1d9300
# ╠═72424251-063a-4e10-839b-10fbff4f2101
# ╠═55de30dc-ce1c-4421-b08c-b6463f201704
# ╟─c69230af-f832-4ef8-8b0c-fda95197e38f
# ╠═8b59e536-7b84-4663-9788-7e8db4f63df3
# ╠═8774d461-c81c-468d-ba13-eedc375a4b76
# ╠═67e8d1a4-ce04-411d-9099-6461e9952e60
# ╠═647aa9b2-5dd8-4aba-bb61-ff3d4bca5241
# ╠═49194b60-5b95-4e4e-8f8d-fadb69a5e27c
# ╠═36bfea5d-3aff-49c0-bed6-d27424c603a5
# ╠═03a99bf1-4029-4d8a-9be9-954a8cc8a44a
# ╟─29d6f2eb-5a87-4ebe-99bf-9c3fc02c335e
# ╠═12b1ab62-f307-41cd-a193-3c378e435a80
# ╠═5bd617b3-5fbd-4a60-8b1b-27baaa5a9286
# ╠═bbd53073-4d50-43cd-8d60-559af7880daf
# ╠═ef0bbaf6-c822-41c8-972c-59368f5928c5
# ╠═ba73c7d0-bf14-4d08-8f91-6585ec77fa23
# ╠═71452494-7caf-48db-8e37-e5fa0ba9eaf0
# ╠═6449db62-e9c7-41ac-a40b-f0055c6ea30d
# ╟─d095ac67-1ec2-4988-96ad-964bfd4465ea
# ╠═9061b2f0-c7a2-424a-b5d6-892dd5feacb3
# ╠═06a07103-9f69-4a0f-927f-26ddd91680fd
# ╠═7da27236-27fd-473d-bc52-1916c6a77952
# ╠═8f77739c-81d8-43c4-839f-1c80ea63761e
# ╠═4af12148-36a4-469d-97f9-b4e142e242a4
# ╠═688a9126-ebcf-4c3b-90c8-37526f06fb28
# ╠═520eb429-18c9-4257-b296-456b9c04e9a3
# ╠═6d2138b1-857c-4409-bc4c-99c39f7d146c
# ╠═8e430f7a-372e-443e-9312-b189e5fe5e46
# ╟─ee8f32fa-82ee-4bb9-a15a-eeb029b8baf1
# ╟─7d2d821d-35e2-411b-95d0-4217ff6d6d65
# ╠═d9842689-1f69-4776-926b-71b8b842f391
# ╠═4fdd9be7-de78-498c-9648-39965a4f0bcd
# ╠═f4d05f19-d140-449e-92c3-5a66a889d60f
# ╠═b7d3a3f8-1b9a-4238-b32a-7bc728fe01a6
# ╟─9f526288-6c61-450b-aa3b-7c9c5e22f445
# ╠═fd9bc2da-f062-4cef-93bc-10f932255bb3
# ╠═c4afb792-7b73-464c-86a4-cf14fd54b899
# ╠═d5805040-fa2f-4746-9f04-2d3f9adfed1c
# ╠═115e20cc-fadd-4fe2-b82a-1cd90cad2035
# ╠═2f58da97-7011-4818-83fa-edf16961e63a
# ╠═daf4d854-dc40-4ccb-8ca9-0019b319324c
# ╠═58aeb048-62f5-4287-bb27-7761a445fad4
# ╠═4fc53029-d832-4508-a9b3-096ce1547d5b
# ╠═d5e05822-16c3-4703-8615-efa3352135e6
# ╠═b0904a71-f8cd-4a14-88a3-a7cf55c5c660
# ╠═2b96fd3b-2536-4bbd-9100-40249a2864bb
# ╠═c9cc9041-e279-4333-bbd0-e71b0aca56c7
# ╠═a16c6b5f-6017-452f-a0ec-161983cf9323
# ╠═d1909b2d-0ed6-42d8-b7b4-ad7245499e5d
# ╟─4c5df011-ae82-40a0-be74-c17ffdbdd3e8
# ╠═ac3b3c2d-4b24-42c8-8ccc-ce0e3e6705be
# ╠═3836fc2f-d930-4697-80a8-8a62f4c84ea7
# ╠═5879cf23-46b9-4f67-b17c-2ea5f391fdae
# ╠═284e36c3-0872-492f-9fdb-35bdd2b20cfd
# ╠═cd34ebd3-9ebd-44a8-93d4-c7d9bc74cb48
# ╠═86c7253c-b10c-463e-8351-6be1d53e13d2
# ╠═151508c6-3fa8-419b-8a60-2ce6df16b9f3
# ╠═4cafb3e8-4ec8-473b-8736-54806359cc52
# ╠═41f2e064-550d-484d-8514-3cde08959e26
# ╠═886adea9-2578-42d5-8b55-713cb468a20a
# ╠═c9377f27-b79c-43df-ab63-7359c9451749
# ╠═37040a37-6ef6-4410-812f-9d5fe2662330
