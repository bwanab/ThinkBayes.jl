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
first(data, 5)

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
	function compute_likelihood(slope, intercept, ks, ns, xs; func=pdf)
		ps = expit(intercept .+ slope .* xs)
		likes = [func(Binomial(n, p), k) for (n, p, k) in zip(ns, ps, ks)]
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

# ╔═╡ cbc0e731-ed50-40c2-8239-6920835a5ceb
predM = permutedims(reduce(vcat, transpose.(pred)), (2,1));

# ╔═╡ d9380762-71b2-4844-8d33-e511963f075b
#low, median, high = unzip(pcts)
low, median, high = percentile(predM, [5, 50, 90])

# ╔═╡ 3a97b9cc-ebbe-4091-b895-4d20cbe36348
ribbon(low, median, high) = (median .- low, high .- median)

# ╔═╡ 17fce776-3798-496b-a8fc-a156f8ec907f
plot(temps, median, ribbon=(ribbon(low, median, high)))

# ╔═╡ 96abb620-636f-45f1-af91-2b50444322e7
lmh_df = DataFrame(Index=temps, median=median, low=low, high=high );

# ╔═╡ 3232a8e8-3ef5-4ccb-96d2-084688b83410
loc(lmh_df, 80)

# ╔═╡ 3565c919-1ec8-474a-a428-1125a8005dd4
loc(lmh_df, 60)

# ╔═╡ 1bf989b9-6354-4a51-bc65-77c3e105078d
loc(lmh_df, 31)

# ╔═╡ 59d18bbf-6b5d-4657-b93f-1f06b712ae99
md"""
## Exercises

### _exercise 16.1_
"""

# ╔═╡ 14c13375-7304-422f-b8e4-4692598e0b70
prior_log_odds = log(4)

# ╔═╡ 6ee92193-7121-47d9-98c7-8af5e6552a39
begin
	lr1 = log(7/5)
	lr2 = log(3/5)
	lr3 = log(9/5)
end

# ╔═╡ 9b139eff-6593-4c2f-b629-f15fba0d769f
posterior_log_odds = prior_log_odds + lr1 + lr2 + lr3

# ╔═╡ 4fd1a491-4f4e-464d-9902-f2804934ee4f
md"### _exercise 16.2_"

# ╔═╡ faa0f18a-2bc9-4606-9a31-ea713e81aa1e
begin
	n1 = [32690, 31238, 34405, 34565, 34977, 34415, 
	                   36577, 36319, 35353, 34405, 31285, 31617]
	k1 = [265, 280, 307, 312, 317, 287, 
	                      320, 309, 225, 240, 232, 243]
end

# ╔═╡ b12c3cf0-73bb-49db-9e59-212efc1b7582
roll(l, n) = vcat(collect(Iterators.drop(l, n)), collect(Iterators.take(l,n)))

# ╔═╡ 1af76000-52c3-4a9e-a39a-eba43e522d1d
begin
	x = 0:11
	n = roll(n1, 8)
	k = roll(k1, 8)
end

# ╔═╡ 653b2e59-93a8-43f4-a573-0f31523aaa07


# ╔═╡ 247f8221-08e0-4173-bed8-06f1ac727899
adhd = DataFrame(x=x, k=k, n=n, rate=k ./ n .* 10000)

# ╔═╡ 13ea4cb9-6001-4646-9f0d-7e01b3a9afa3
begin
	scatter(adhd.x, adhd.rate)
	vline!([5.5], color=:gray, alpha=0.2)
	annotate!([(6, 64, text("Younger than average",10, :left))
				(5, 64, text("Older than average", 10, :right))])
	plot!(xaxis=("Bright date, months after cutoff"), yaxis=("Diagnosis rate per 10,000"))
end

# ╔═╡ 442eab20-49b1-4b7a-95e9-f6f6f6e86d88
prior_inter_e2 = pmf_from_seq(range(-5.2, -4.6, 51));

# ╔═╡ 6148e327-880c-43cc-9a1f-faacfc94940e
prior_slope_e2 = pmf_from_seq(range(0.0, 0.08, 51));

# ╔═╡ 2fc61207-0518-489b-a7c3-56e3fae8a404
prior_joint_e2 = make_joint(*, prior_slope_e2, prior_inter_e2);

# ╔═╡ 78369e77-9cef-4656-9a96-c70a06c67f9f
prior_joint_index, prior_joint_vals = stack(prior_joint_e2)

# ╔═╡ 078e0547-efbe-4a12-aae1-83b456b22197
prior_joint_pmf_e2 = pmf_from_seq(prior_joint_index, prior_joint_vals);

# ╔═╡ e0c5c33f-ccf3-492b-ac09-d3c900702450
begin
	ns1_e2 = adhd[1:9, :n]
	ks1_e2 = adhd[1:9, :k]
	xs1_e2 = adhd[1:9, :x]
end

# ╔═╡ 2df993c1-2e0b-4e98-af9e-14ebb086e3b7
likelihood1_e2 = [compute_likelihood(slope, intercept, ks1_e2, ns1_e2, xs1_e2) for (slope, intercept) in prior_joint_index]

# ╔═╡ 591c522f-19ee-4c42-b956-e3eb4c4b4f6e
begin
	ns2_e2 = adhd[10:end, :n]
	ks2_e2 = adhd[10:end, :k]
	xs2_e2 = adhd[10:end, :x]
end

# ╔═╡ 227d38a2-0423-4db3-8f44-aa5c918299b3
adhd[1:9, :]

# ╔═╡ 0fab40a0-b069-4c39-a71e-64f8251e9bbe
adhd[10:end, :]

# ╔═╡ 1e6f9561-c972-4b82-bc6b-2628b6db5d40
likelihood2_e2 = [compute_likelihood(slope, intercept, ks2_e2, ns2_e2, xs2_e2, func=ccdf) for (slope, intercept) in prior_joint_index]

# ╔═╡ 3afdac43-2dd4-4fa8-b548-3b8513b5f371
sum(likelihood1_e2)

# ╔═╡ ae4af115-38d8-46cf-a3b4-54d995ce3a43
sum(likelihood2_e2)

# ╔═╡ cef2e457-3118-4132-ad14-099f7d0f661e
posterior_joint_pmf1_e2 = prior_joint_pmf_e2 * likelihood1_e2;

# ╔═╡ 83fc874d-5d14-46f0-9056-96c8f89232e9
max_prob(posterior_joint_pmf1_e2)

# ╔═╡ 5bec7761-2fcd-4563-8231-6ae609d92a3e
posterior_joint_pmf2_e2 = prior_joint_pmf_e2 * (likelihood1_e2 .* likelihood2_e2);

# ╔═╡ 8ba40a96-3b3f-4bd2-bae3-88b46d169d83
max_prob(posterior_joint_pmf2_e2)

# ╔═╡ 02fb79f6-9d59-48ec-bdb0-2a7c28a81cb0
joint_posterior_e2 = transpose(unstack(values(posterior_joint_pmf2_e2), probs(posterior_joint_pmf2_e2)));

# ╔═╡ 6aef8ef7-8bf4-4b4b-b205-586282824bfd
contour(joint_posterior_e2)

# ╔═╡ fa7a2957-2473-426b-9a4e-898974d6ed99
begin
	marginal_inter_e2 = marginal(joint_posterior_e2, 1)
	marginal_slope_e2 = marginal(joint_posterior_e2, 2)
end;

# ╔═╡ 434b10b8-f514-4912-bae3-1259181128a7
plot(marginal_inter_e2)

# ╔═╡ aca7c105-7f78-4547-be13-e0717bedc62d
plot(marginal_slope_e2)

# ╔═╡ 2285e1ab-fa91-4229-b4cd-6bb072662c84
sample_e2 = rand(posterior_joint_pmf2_e2, 101)

# ╔═╡ 4217d922-a320-4452-8811-9f03859be4ea
xs3_e2 = adhd.x

# ╔═╡ 16a5e8c2-a1eb-43d3-a1db-b1a969fe91b3
ps2_e2 = [expit(inter .+ slope .* xs3_e2) for (slope, inter) in sample_e2]

# ╔═╡ 5b1c2a3c-db89-4787-895a-247480d525f3
ps2_reshape = permutedims(reduce(vcat, transpose.(ps2_e2)), (2,1));

# ╔═╡ 8ebdeda4-a5a1-441e-a871-e2cd65dc7fd5
low_e2, median_e2, high_e2 = percentile(ps2_reshape, [2.5, 50, 97.5])

# ╔═╡ cf491e49-ee32-4be2-8945-2ca860ce765e
median_e2

# ╔═╡ 4a71475b-f686-49a0-ac85-fe78af3b3f53
begin
	plot(xs3_e2, median_e2*10000, ribbon=(ribbon(low_e2*10000, median_e2*10000, high_e2*10000)))
	scatter!(adhd.x, adhd.rate)
end

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
# ╠═cbc0e731-ed50-40c2-8239-6920835a5ceb
# ╠═d9380762-71b2-4844-8d33-e511963f075b
# ╠═3a97b9cc-ebbe-4091-b895-4d20cbe36348
# ╠═17fce776-3798-496b-a8fc-a156f8ec907f
# ╠═96abb620-636f-45f1-af91-2b50444322e7
# ╠═3232a8e8-3ef5-4ccb-96d2-084688b83410
# ╠═3565c919-1ec8-474a-a428-1125a8005dd4
# ╠═1bf989b9-6354-4a51-bc65-77c3e105078d
# ╟─59d18bbf-6b5d-4657-b93f-1f06b712ae99
# ╠═14c13375-7304-422f-b8e4-4692598e0b70
# ╠═6ee92193-7121-47d9-98c7-8af5e6552a39
# ╠═9b139eff-6593-4c2f-b629-f15fba0d769f
# ╟─4fd1a491-4f4e-464d-9902-f2804934ee4f
# ╠═faa0f18a-2bc9-4606-9a31-ea713e81aa1e
# ╠═b12c3cf0-73bb-49db-9e59-212efc1b7582
# ╠═1af76000-52c3-4a9e-a39a-eba43e522d1d
# ╠═653b2e59-93a8-43f4-a573-0f31523aaa07
# ╠═247f8221-08e0-4173-bed8-06f1ac727899
# ╠═13ea4cb9-6001-4646-9f0d-7e01b3a9afa3
# ╠═442eab20-49b1-4b7a-95e9-f6f6f6e86d88
# ╠═6148e327-880c-43cc-9a1f-faacfc94940e
# ╠═2fc61207-0518-489b-a7c3-56e3fae8a404
# ╠═78369e77-9cef-4656-9a96-c70a06c67f9f
# ╠═078e0547-efbe-4a12-aae1-83b456b22197
# ╠═e0c5c33f-ccf3-492b-ac09-d3c900702450
# ╠═2df993c1-2e0b-4e98-af9e-14ebb086e3b7
# ╠═591c522f-19ee-4c42-b956-e3eb4c4b4f6e
# ╠═227d38a2-0423-4db3-8f44-aa5c918299b3
# ╠═0fab40a0-b069-4c39-a71e-64f8251e9bbe
# ╠═1e6f9561-c972-4b82-bc6b-2628b6db5d40
# ╠═3afdac43-2dd4-4fa8-b548-3b8513b5f371
# ╠═ae4af115-38d8-46cf-a3b4-54d995ce3a43
# ╠═cef2e457-3118-4132-ad14-099f7d0f661e
# ╠═83fc874d-5d14-46f0-9056-96c8f89232e9
# ╠═5bec7761-2fcd-4563-8231-6ae609d92a3e
# ╠═8ba40a96-3b3f-4bd2-bae3-88b46d169d83
# ╠═02fb79f6-9d59-48ec-bdb0-2a7c28a81cb0
# ╠═6aef8ef7-8bf4-4b4b-b205-586282824bfd
# ╠═fa7a2957-2473-426b-9a4e-898974d6ed99
# ╠═434b10b8-f514-4912-bae3-1259181128a7
# ╠═aca7c105-7f78-4547-be13-e0717bedc62d
# ╠═2285e1ab-fa91-4229-b4cd-6bb072662c84
# ╠═4217d922-a320-4452-8811-9f03859be4ea
# ╠═16a5e8c2-a1eb-43d3-a1db-b1a969fe91b3
# ╠═5b1c2a3c-db89-4787-895a-247480d525f3
# ╠═8ebdeda4-a5a1-441e-a871-e2cd65dc7fd5
# ╠═cf491e49-ee32-4be2-8945-2ca860ce765e
# ╠═4a71475b-f686-49a0-ac85-fe78af3b3f53
