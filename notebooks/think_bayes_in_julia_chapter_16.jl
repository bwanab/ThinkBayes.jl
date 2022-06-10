### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 75ff3f3e-e68d-11ec-3089-63c586f05505
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 20a227a8-dece-403f-af26-92225880e2b8
using DataFrames, CSV, Plots, Distributions, PlutoUI, LinearAlgebra, Statistics, GLM

# ╔═╡ f1325978-b2d0-4ec8-ad45-7761510280f4
TableOfContents()

# ╔═╡ 860ea21a-13fb-44d7-b3dc-1fe39e4ce945
md"## Log Odds"

# ╔═╡ bfc2cf2d-b039-4f55-9bb4-cd5d49a86283
prob(o) = o / (o + 1)

# ╔═╡ ca4f274a-a896-4abe-87c6-afc50f9fd9c6


# ╔═╡ 20d6dee6-244d-420a-86ab-bf080d186db6
begin
	table = DataFrame(Index=["prior", "1 student", "2 students", "3 students"])
	table.odds = [10/3^x for x in 0:3]
	table.prob = 100 .* prob.(table.odds)
	table.prob_diff = vcat([missing], diff(table.prob))
	table
end

# ╔═╡ 7140af96-dfa0-4439-81a8-4bc9b128d0c3
begin
	table.log_odds = log.(table.odds)
	table.log_odds_diff = vcat([missing], diff(table.log_odds))
	table
end

# ╔═╡ fb12a0a8-5e2b-4e84-b9b3-28de8d6371ae
md"## The Space Shuttle Problem"

# ╔═╡ 6eae5309-b4b0-4a5c-850d-f5dd3c8aad1e
data = DataFrame(CSV.File(download("https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv"), footerskip=1, missingstring=["NA"], dateformat="mm/dd/yyyy"));

# ╔═╡ c736605d-3b33-4ae7-9406-9830828b8584
begin
	dropmissing!(data)
	rename!(data, "Damage Incident" => :Damage)
end;

# ╔═╡ e30170b5-302f-434d-a800-0c49039238e1
data

# ╔═╡ cbe9aa3b-8612-4d64-87e4-35e8497c0dc4
nrow(data), sum(data.Damage)

# ╔═╡ bb508993-3f34-4757-9c6b-113a1e45ed6d
begin
	scatter(data.Temperature, data.Damage)
	plot!(xaxis=("Outside temperature"), yaxis=("probability of damage"))
end

# ╔═╡ 4a96cc91-0bf3-42a2-a482-e71e242399be
begin
	offset = round(Int, mean(data.Temperature))
	println(offset)
	data.x .= data.Temperature .- offset 
end

# ╔═╡ 580a7076-f969-45d2-b0fc-399885cf7353
data.y = data.Damage

# ╔═╡ 2f3bc4f5-f0cf-4181-a8df-622f8290b75e
data

# ╔═╡ fa8eaff9-6645-4dcc-bacd-52d7830257b0
ols = glm(@formula(y ~ x), data, Binomial(), LogitLink())

# ╔═╡ 269d4db4-5625-46c7-82ac-f750777b660d
intercept, slope = coef(ols.model)

# ╔═╡ 550a2b10-7334-419e-aff0-ce363733b514
xs = range(53, 82) .- offset

# ╔═╡ 375c3185-f057-44a6-946c-dddb99b18bc1
log_odds = intercept .+ slope .* xs

# ╔═╡ e5095fc1-fa80-49bb-b50c-ca3534cdd168
odds = exp.(log_odds)

# ╔═╡ 30f01508-e430-48ce-9a09-1a1fb69a7643
ps = prob.(odds)

# ╔═╡ fc0a82c6-dc71-4a8b-99e5-7dc235013b86
function expit(vs)
	prob.(exp.(vs))
end

# ╔═╡ 819451ad-2c23-465d-a6fa-1a6b0c9e207c
mean(expit(intercept .+ slope .* xs))

# ╔═╡ bf40c494-caa6-4cc3-aff3-79591fc8b4eb
mean(ps)

# ╔═╡ d52674b5-b0fd-441d-a5ae-ddc05f224331
begin
	plot(xs .+ offset, ps, label="model")
	scatter!(data.Temperature, data.Damage, label="data")
	plot!(xaxis=("Outside temperature"), yaxis=("probability of damage"))
end

# ╔═╡ 7c14ff0e-3816-4e89-bfbf-f5d36ec8df44
md"## Prior Distribution"

# ╔═╡ 30c3f393-1a7a-4adc-9cec-263947c3ba86
prior_inter = pmf_from_seq(range(-5, 1, 101));

# ╔═╡ ecff59b8-728c-4a5a-8273-e84b38a1e160
prior_slope = pmf_from_seq(range(-0.8, 0.1, 101));

# ╔═╡ 608840c5-30c9-4415-832a-929a0939fd72
joint = make_joint(*, prior_inter, prior_slope);

# ╔═╡ 26185c14-e5d9-469d-bd17-b281ddce96f6
joint_index, joint_vals = stack(joint)

# ╔═╡ fb9537b0-2fbf-476c-887a-5ddadf4cc438
joint_pmf = pmf_from_seq(joint_index, joint_vals);

# ╔═╡ e17a38ff-dae3-472a-80d3-ed4f1cca337d
md"## Likelihood"

# ╔═╡ 35e71e71-513c-44f9-85cf-148dab5e59bf
begin
	g=groupby(data, :x)
	grouped = DataFrame([(Index=first(x).x, count=nrow(x), sum=sum(x.y)) for x in g])
end

# ╔═╡ 9aa4d6b9-42a0-411c-9b46-19fc0d5802e9
begin
	ns = grouped.count
	ks = grouped.sum
	xs2 = grouped.Index
end

# ╔═╡ 0df75fd7-c53b-45d3-a3ee-6114ccfd7df8
xs2

# ╔═╡ c491cccf-0075-48eb-936c-31dbf3344cf6
ps2 = expit(intercept .+ slope .* xs2)

# ╔═╡ 177f8214-e5fa-41f0-9592-38f454e0ea0f
likes = [pdf(Binomial(n, p), k) for (n, p, k) in zip(ns, ps2, ks)]

# ╔═╡ a08186b1-8005-4651-8834-989d1648aee2
prod(likes)

# ╔═╡ ee2c849e-30b7-4be8-a4f1-cf7ca691d482
begin
	function compute_likelihood(slope, intercept, ks, ns, xs)
		ps = expit(intercept .+ slope .* xs)
		likes = [pdf(Binomial(n, p), k) for (n, p, k) in zip(ns, ps, ks)]
		prod(likes)
	end
	likelihood = [compute_likelihood(slope, intercept, ks, ns, xs2) for (intercept, slope) in joint_index]
end;

# ╔═╡ 1cb270dc-60b9-4ea9-8b89-e91891a0abba
md"## The Update"

# ╔═╡ 33722f58-3136-4f03-b986-ebdbf5053f6b
posterior_pmf = joint_pmf * likelihood;

# ╔═╡ e83ccc2b-5beb-441a-b4d7-b74860e97ad5
max_prob(posterior_pmf), coef(ols.model)

# ╔═╡ ed549e2e-365a-46c6-aeee-ab36dfd4e4bb
joint_posterior = unstack(values(posterior_pmf), probs(posterior_pmf));

# ╔═╡ 1e58f755-d845-4d51-bacf-f441dc54f199
contour(joint_posterior)

# ╔═╡ 690a7bd9-add4-45b5-bdd7-73580d3bf1eb
md"## Marginal Distributions"

# ╔═╡ 8d255db4-c3b9-41e3-b743-e893a665119d
begin
	marginal_intercept = marginal(joint_posterior, 1)
	marginal_slope = marginal(joint_posterior, 2)
end;

# ╔═╡ 2b926c9e-ec7a-42c8-9064-d1bc02d47900
plot(marginal_intercept, label="intercept", title="Posterior marginal distribution of intercept", xaxis=("Intercept"), yaxis=("PDF"))

# ╔═╡ f7dc6a3e-7525-4faf-8897-9f892faf37cc
plot(marginal_slope, label="slope", title="Posterior marginal distribution of slope", xaxis=("Slope"), yaxis=("PDF"))

# ╔═╡ cf34548f-81eb-4a0e-b82c-6da92c647e36
mean(marginal_intercept), mean(marginal_slope)

# ╔═╡ 1d90f1b6-a06d-4b36-b843-2ae73bbd7dd7
md"## Transforming Distributions"

# ╔═╡ e8ae7512-0128-4f43-9420-95a2aaadbda0
transform(pmf, func) = pmf_from_seq(func(values(pmf)), probs(pmf))

# ╔═╡ 70d371e3-7057-4de6-a17b-cef499f88903
marginal_probs = transform(marginal_intercept, expit);

# ╔═╡ aeab7803-6921-4c7b-b700-2fba2464852e
begin
	plot(marginal_probs)
	plot!(xaxis=("Probability of damage at 70 deg F"),
	         yaxis=("PDF"),
	         title="Posterior marginal distribution of probabilities")
end

# ╔═╡ 0330631b-c70c-4df9-9472-a58654b09f1c
mean_prob = mean(marginal_probs)

# ╔═╡ 2e09c233-b1bf-45d4-873f-14a263add397
marginal_lr = transform(marginal_slope, x->exp.(x));

# ╔═╡ 156a584c-8d4b-4463-a842-ecd64422ca27
begin
	plot(marginal_lr)
	plot!(xaxis=("Likelihood ratio of 1 deg F"),
	      yaxis=("PDF"),
	      title="Posterior marginal distribution of likelihood ratios")
end

# ╔═╡ 0c5fed14-6f32-4554-981e-9485c27f4af6
mean_lr = mean(marginal_lr)

# ╔═╡ 3cf2bcca-96f4-4201-ad2d-0b984d2f89d7
md"## Predictive Distributions"

# ╔═╡ c35fd75c-5d08-4ba3-b194-fd4eb55b8951
md"We have to flip the tuples to give them the same order as in ThinkBayes2."

# ╔═╡ 9e8bb9aa-c92e-497e-b818-006bb9a1f4a4
sample = [(y, x) for (x, y) in rand(posterior_pmf, 101)]

# ╔═╡ dae8f67d-4050-493e-aa96-19f4d6cde5c5
begin
	temps = 31:82
	xs3 = temps .- offset
end

# ╔═╡ 4fa0b559-c088-466a-870f-2f3af6905483
pred = [expit(inter .+ slope .* xs3) for (slope, inter) in sample]

# ╔═╡ 0aeb9f4b-c8d9-4ad3-9e09-680ccfd631cb
begin
	plot(title="Damage to O-Rings vs Temperature", xaxis=("Outside temperature (°F)"), yaxis = ("Probability of damage"), legend=false)
	for ps in pred
		plot!(temps, ps, alpha=0.4)
	end
	plot!()
	scatter!(data.Temperature, data.Damage, label="data")
end

# ╔═╡ 2fcaff07-1243-4a02-a25c-481c459e890f
function percentile(data, p)
	c = cdf_from_seq(data)
	quantile(c, p / 100)
end

# ╔═╡ cbc0e731-ed50-40c2-8239-6920835a5ceb
predM = reshape(collect(Iterators.flatten(pred)), length(temps), length(sample))

# ╔═╡ a8dc6bc2-5fb8-4707-aefc-d5b52e8a6144
pcts = [percentile(predM[i,:], [5, 50, 90]) for i in 1:length(temps)]

# ╔═╡ 953edfd4-b07e-4843-a999-828c73113b18
unzip(a) = [getindex.(a, i) for i in 1:length(a[1])]

# ╔═╡ d9380762-71b2-4844-8d33-e511963f075b
low, median, high = unzip(pcts)

# ╔═╡ 3a97b9cc-ebbe-4091-b895-4d20cbe36348
ribbon(low, median, high) = (median .- low, high .- median)

# ╔═╡ 17fce776-3798-496b-a8fc-a156f8ec907f
plot(temps, median, ribbon=(ribbon(low, median, high)))

# ╔═╡ 96abb620-636f-45f1-af91-2b50444322e7
lmh_df = DataFrame(temp=temps, median=median, low=low, high=high )

# ╔═╡ 1807779a-269d-4942-8c5e-2fc14fae43c2
lmh_df[findfirst(==(80), lmh_df.temp), :]

# ╔═╡ 257e6a96-9887-4e5f-990c-276666fc1a89
lmh_df[findfirst(==(60), lmh_df.temp), :]

# ╔═╡ ec368527-f463-4315-9fc3-cabba30f232f
lmh_df[findfirst(==(31), lmh_df.temp), :]

# ╔═╡ Cell order:
# ╠═20a227a8-dece-403f-af26-92225880e2b8
# ╠═f1325978-b2d0-4ec8-ad45-7761510280f4
# ╠═75ff3f3e-e68d-11ec-3089-63c586f05505
# ╟─860ea21a-13fb-44d7-b3dc-1fe39e4ce945
# ╠═bfc2cf2d-b039-4f55-9bb4-cd5d49a86283
# ╠═ca4f274a-a896-4abe-87c6-afc50f9fd9c6
# ╠═20d6dee6-244d-420a-86ab-bf080d186db6
# ╠═7140af96-dfa0-4439-81a8-4bc9b128d0c3
# ╟─fb12a0a8-5e2b-4e84-b9b3-28de8d6371ae
# ╠═6eae5309-b4b0-4a5c-850d-f5dd3c8aad1e
# ╠═c736605d-3b33-4ae7-9406-9830828b8584
# ╠═e30170b5-302f-434d-a800-0c49039238e1
# ╠═cbe9aa3b-8612-4d64-87e4-35e8497c0dc4
# ╠═bb508993-3f34-4757-9c6b-113a1e45ed6d
# ╠═4a96cc91-0bf3-42a2-a482-e71e242399be
# ╠═580a7076-f969-45d2-b0fc-399885cf7353
# ╠═2f3bc4f5-f0cf-4181-a8df-622f8290b75e
# ╠═fa8eaff9-6645-4dcc-bacd-52d7830257b0
# ╠═269d4db4-5625-46c7-82ac-f750777b660d
# ╠═550a2b10-7334-419e-aff0-ce363733b514
# ╠═375c3185-f057-44a6-946c-dddb99b18bc1
# ╠═e5095fc1-fa80-49bb-b50c-ca3534cdd168
# ╠═30f01508-e430-48ce-9a09-1a1fb69a7643
# ╠═fc0a82c6-dc71-4a8b-99e5-7dc235013b86
# ╠═819451ad-2c23-465d-a6fa-1a6b0c9e207c
# ╠═bf40c494-caa6-4cc3-aff3-79591fc8b4eb
# ╠═d52674b5-b0fd-441d-a5ae-ddc05f224331
# ╟─7c14ff0e-3816-4e89-bfbf-f5d36ec8df44
# ╠═30c3f393-1a7a-4adc-9cec-263947c3ba86
# ╠═ecff59b8-728c-4a5a-8273-e84b38a1e160
# ╠═608840c5-30c9-4415-832a-929a0939fd72
# ╠═26185c14-e5d9-469d-bd17-b281ddce96f6
# ╠═fb9537b0-2fbf-476c-887a-5ddadf4cc438
# ╟─e17a38ff-dae3-472a-80d3-ed4f1cca337d
# ╠═35e71e71-513c-44f9-85cf-148dab5e59bf
# ╠═9aa4d6b9-42a0-411c-9b46-19fc0d5802e9
# ╠═0df75fd7-c53b-45d3-a3ee-6114ccfd7df8
# ╠═c491cccf-0075-48eb-936c-31dbf3344cf6
# ╠═177f8214-e5fa-41f0-9592-38f454e0ea0f
# ╠═a08186b1-8005-4651-8834-989d1648aee2
# ╠═ee2c849e-30b7-4be8-a4f1-cf7ca691d482
# ╟─1cb270dc-60b9-4ea9-8b89-e91891a0abba
# ╠═33722f58-3136-4f03-b986-ebdbf5053f6b
# ╠═e83ccc2b-5beb-441a-b4d7-b74860e97ad5
# ╠═ed549e2e-365a-46c6-aeee-ab36dfd4e4bb
# ╠═1e58f755-d845-4d51-bacf-f441dc54f199
# ╟─690a7bd9-add4-45b5-bdd7-73580d3bf1eb
# ╠═8d255db4-c3b9-41e3-b743-e893a665119d
# ╠═2b926c9e-ec7a-42c8-9064-d1bc02d47900
# ╠═f7dc6a3e-7525-4faf-8897-9f892faf37cc
# ╠═cf34548f-81eb-4a0e-b82c-6da92c647e36
# ╟─1d90f1b6-a06d-4b36-b843-2ae73bbd7dd7
# ╠═e8ae7512-0128-4f43-9420-95a2aaadbda0
# ╠═70d371e3-7057-4de6-a17b-cef499f88903
# ╠═aeab7803-6921-4c7b-b700-2fba2464852e
# ╠═0330631b-c70c-4df9-9472-a58654b09f1c
# ╠═2e09c233-b1bf-45d4-873f-14a263add397
# ╠═156a584c-8d4b-4463-a842-ecd64422ca27
# ╠═0c5fed14-6f32-4554-981e-9485c27f4af6
# ╟─3cf2bcca-96f4-4201-ad2d-0b984d2f89d7
# ╟─c35fd75c-5d08-4ba3-b194-fd4eb55b8951
# ╟─9e8bb9aa-c92e-497e-b818-006bb9a1f4a4
# ╠═dae8f67d-4050-493e-aa96-19f4d6cde5c5
# ╠═4fa0b559-c088-466a-870f-2f3af6905483
# ╠═0aeb9f4b-c8d9-4ad3-9e09-680ccfd631cb
# ╠═2fcaff07-1243-4a02-a25c-481c459e890f
# ╠═cbc0e731-ed50-40c2-8239-6920835a5ceb
# ╠═a8dc6bc2-5fb8-4707-aefc-d5b52e8a6144
# ╠═953edfd4-b07e-4843-a999-828c73113b18
# ╠═d9380762-71b2-4844-8d33-e511963f075b
# ╠═3a97b9cc-ebbe-4091-b895-4d20cbe36348
# ╠═17fce776-3798-496b-a8fc-a156f8ec907f
# ╠═96abb620-636f-45f1-af91-2b50444322e7
# ╠═1807779a-269d-4942-8c5e-2fc14fae43c2
# ╠═257e6a96-9887-4e5f-990c-276666fc1a89
# ╠═ec368527-f463-4315-9fc3-cabba30f232f
