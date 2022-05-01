### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 4ae2fccd-6676-4609-b692-7a827746820d
using Pkg

# ╔═╡ 97b39b48-0a44-4ba3-bbd9-5a91cf1041fd
# ╠═╡ disabled = true
#=╠═╡
Pkg.activate("/Users/williamallen/.julia/dev/ThinkBayes2")
  ╠═╡ =#

# ╔═╡ 0432f183-1e16-4934-9a7a-13f5050ff1a6
using PlutoUI

# ╔═╡ d6c3293e-e2c9-4233-af04-c521d1c77d91
using StatsPlots

# ╔═╡ d8af51e2-211b-487b-90da-0a41a096652c
using Distributions

# ╔═╡ 2b08b8bc-93f6-4f2c-aa21-f69087e102b4
# ╠═╡ disabled = true
#=╠═╡
using Plots
  ╠═╡ =#

# ╔═╡ 663ae283-a756-4545-b70b-64e9fe93340e
using DataFrames

# ╔═╡ eec67984-1b6c-446e-b06d-26e7f9b5cd39
using ThinkBayes

# ╔═╡ 524496e1-6550-499d-9243-910d5931d28e
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 6e40c123-2acc-4532-9f91-02f150c8f0ca
xs = LinRange(0, 1, 101)

# ╔═╡ 9b975ff7-e9a3-4e5d-8394-a653555d01a0
uniform = pmf_from_seq(xs)

# ╔═╡ 0c4d3d9a-e335-4766-b2f4-507aa7b97a9d
begin
	k = 140
	n = 250
	likelihood = binom_pmf(k, n, xs)
end

# ╔═╡ 19563b94-9de2-4d50-8269-f7a3051e0620
posterior = uniform * likelihood

# ╔═╡ c0c98fc2-c5d2-4e45-8ee6-ef702a1e1c6f
plot(posterior)

# ╔═╡ e819b552-5dec-4b9c-8d1c-490fa4023491
print(mean(posterior), " ", credible_interval(posterior, 0.9))

# ╔═╡ f8689fc2-3e64-49a8-a325-e9064408702f
md"
## Evidence
"

# ╔═╡ 4cc4f230-e91e-45a0-9462-af33c161841c
like_fair = pdf(Binomial(n, 0.5), k)

# ╔═╡ 263a73e0-018a-4360-a433-1a47322dee3d
like_biased = pdf(Binomial(n, 0.56), k)

# ╔═╡ a7051003-c692-439d-bc27-cf07e9e0c255
K = like_biased / like_fair

# ╔═╡ fbfbc736-13b0-4e26-8f0d-aeea50d5efb1
md"## Uniformly Distributed Bias"

# ╔═╡ 766d55ab-6ec4-4d36-9032-3e5f7d6946b3
biased_uniform = copy(uniform)

# ╔═╡ 9f715cfa-d767-45e7-950f-9ad9d59f6fc5
biased_uniform[0.5] = 0

# ╔═╡ 0da308b1-3ef3-4596-8060-644a2637f141
bu_likelihood = binom_pmf(k, n, values(biased_uniform))

# ╔═╡ 0e105c5b-4538-4c41-91c2-f2ed7a2d0d44
like_uniform = sum(probs(biased_uniform) .* bu_likelihood)

# ╔═╡ 6f908076-f7a1-4668-9040-0946cda0fa9e
bu_K = like_fair / like_uniform

# ╔═╡ beecea10-c1c0-47d2-a0b5-a8218e55a969
prior_odds = 1

# ╔═╡ 5cd4777c-3dcc-4ec2-8809-d5c02fbaa403
posterior_odds = prior_odds * bu_K

# ╔═╡ fb03e48b-e09c-42cf-a9a9-15af7dd2ba24
prob(o) = o / (o + 1)

# ╔═╡ a238b43d-b497-4b76-baec-020f53bf4117
posterior_probability = prob(posterior_odds)

# ╔═╡ db1d8c5f-3439-40ad-bd35-26d22bd71efe
a = vcat(0:49, 50:-1:0)


# ╔═╡ 8002873d-91b0-4d8b-be19-eeb186d0a357
triangle = pmf_from_seq(xs, normalize(a))

# ╔═╡ a99ddc6c-6d6f-4234-a99c-e27cd14c6d47
biased_triangle = copy(triangle)

# ╔═╡ a5ceee9e-7871-4dc8-b7a2-5163a1dd4c0c
biased_triangle[0.5] = 0

# ╔═╡ 0b8a260c-e439-44a3-b030-972df44dbb41
plot(biased_uniform, label="uniform_prior")

# ╔═╡ 3d82d934-84d8-4ed0-936e-c57e8296f19b
plot!(biased_triangle, label="triangle_prior")

# ╔═╡ c13ebce2-0b73-4763-8cce-990a91bd570a
bt_likelihood = binom_pmf(k, n, values(biased_triangle))

# ╔═╡ 587d348c-479b-4275-9a47-0f70443dafe5
like_triangle = sum(probs(biased_triangle) .* bt_likelihood)

# ╔═╡ b29bfcb0-adc9-4fe9-b5b3-3c44b0981afc
bt_K = like_fair / like_triangle

# ╔═╡ 97b1f361-da6d-4ec8-a469-d7ce84b1ebc5
bt_posterior_prob = prob(bt_K)

# ╔═╡ dcb876db-4bf6-4f6f-b92a-4486b011231e
md"## my aside"

# ╔═╡ 8b90735d-b9d4-4c46-9f1e-0b4e6a457c31
m = make_normal_pmf(xs, mu=0.5, sigma=0.05)

# ╔═╡ b70689dd-0fcb-492a-83d6-86a4da103607
biased_rev_norm = pmf_from_seq(xs, normalize(abs.(maximum(probs(m)) .- probs(m))))

# ╔═╡ ccd51211-24af-4b9c-84d4-8f6a2021a751
like_rev_norm = sum(probs(biased_rev_norm) .* likelihood)

# ╔═╡ e3062a04-4cc8-4918-970d-c68aaf4e006c
prob(like_fair / like_rev_norm)

# ╔═╡ c244f295-6c75-4060-be9e-54c45eb80aa3
md"## Bayesian Hypothesis Testing"

# ╔═╡ 110f31d4-f181-47b3-8fe8-b57e26621db6
md"## Prior Beliefs"

# ╔═╡ 2e4be690-1a3b-461f-b4c3-f85db0e49d89
cxs = LinRange(0, 1, 101)

# ╔═╡ 58c0678e-0d1f-428e-b610-49cd0661e76c
prior = pmf_from_seq(cxs)

# ╔═╡ 9eb2e94e-7229-4616-9c77-329ddc31ef43
beliefs = [copy(prior) for i in 1:4]

# ╔═╡ c41ab60e-8b02-4186-9922-5be25d24a79f
plot(beliefs)

# ╔═╡ c4668ea7-d8dd-4cdf-b514-c08b8eb42c4a
c_likelihood = Dict('W' => xs, 'L' => 1 .- xs)

# ╔═╡ 69c09862-8d0c-48d3-aef0-5428f1e6abe2
function update(p, data)
	p1 = p
	for d in data
		p1 *= c_likelihood[d]
	end
	p1
end
		

# ╔═╡ accef95d-ffae-4a6a-a414-86171824b265
bandit = copy(prior)

# ╔═╡ d433649f-bf9f-4c4c-aa58-a5d9eb1e7c4a
bandit_post = update(bandit, "WLLLLLLLLL")

# ╔═╡ 1cfcaa78-968f-4794-8cf1-ccb79f55f9c1
plot(bandit_post, xaxis=("Probability of winning"))

# ╔═╡ b7f58b7e-c9de-4915-b7bd-e0bf2d539f11
actual_probs = [0.1, 0.2, 0.3, 0.4]

# ╔═╡ e07d5b2b-bee9-41ff-bbd9-72de2519e580
counter = Dict()

# ╔═╡ 35e856e2-3c9a-431b-a283-154f143a10cf
function play(i)
	counter[i] = get(counter, i, 0) + 1
	p = actual_probs[i]
	if rand() < p
		return "W"
	else
		return "L"
	end
end

# ╔═╡ cdc87aa9-f448-426c-974a-9b410cbbfca5
for i in 1:4
	for j in 1:10
		outcome = play(i)
		beliefs[i] = update(beliefs[i], outcome)
	end
end

# ╔═╡ 922f2399-f995-48ed-9d64-413b68e8faba
plot(beliefs)

# ╔═╡ 059a69e3-1749-4898-87ed-6e8690585002
function summarize_beliefs(beliefs)
	means = fill(0.0, 4)
	ci_low = fill(0.0, 4)
	ci_hi = fill(0.0, 4)
	for (i, b) in enumerate(beliefs)
		means[i] = mean(b)
		(low, high) = credible_interval(b, 0.9)
		ci_low[i] = low
		ci_hi[i] = high
	end
	DataFrame(Actual_P_win=actual_probs, Posterior_mean=means, Credible_interval_low=ci_low, Credible_interval_high=ci_hi)
end

# ╔═╡ 4c4f7cf4-722d-4731-998c-61d82355c0e6
df = summarize_beliefs(beliefs)

# ╔═╡ 7eff2c57-cbe8-4913-aca3-897444249d8b
samples = permutedims(reshape(reduce(vcat, [rand(b, 1000) for b in beliefs]), 1000, 4))

# ╔═╡ d9d47c0d-5f09-4d74-95a3-3eb3d47d0d67
idx = vec([x[1] for x in argmax(samples, dims=1)])

# ╔═╡ a95651fe-1392-4476-93cf-ac0ad29fd3ce
p = pmf_from_seq(idx)

# ╔═╡ 679a8015-8323-4f1f-b36b-73afa6cf4c06
function choose(beliefs)
	ps = [rand(b) for b in beliefs]
	argmax(ps)[1]
end

# ╔═╡ 9dc6b706-2f93-4e7b-9704-af7a7dba5833
choose(beliefs)

# ╔═╡ 574f8530-830f-4d14-a96b-b1f4cd10b9cf
md"## The Strategy"

# ╔═╡ 7f9ded89-d120-4242-9167-601435dfadc4
function choose_play_update(beliefs)
	machine = choose(beliefs)
	outcome = play(machine)
	beliefs[machine] = update(beliefs[machine], outcome)
end

# ╔═╡ 169ae1d1-8ff7-4f44-8506-2c448b126066
begin
	empty!(counter)
	function test_beliefs(prior)
		counter = Dict()
		beliefs = [copy(prior) for i in 1:4];
		num_plays = 1000
		for i in 1:num_plays
			choose_play_update(beliefs)
		end
		beliefs
	end
	beliefs2 = test_beliefs(prior)
end;

# ╔═╡ 93bfd7fe-6f31-4456-91f4-4bcc06e23e56
plot(beliefs2)

# ╔═╡ e0ad0ec5-e3c9-45b1-b654-8ae61143afc4
summarize_beliefs(beliefs2)

# ╔═╡ e5037692-5321-4bcf-b513-d3e36f7a0006
counter

# ╔═╡ 384c4659-99d3-4488-92d4-22945301e805
md"## Exercises"

# ╔═╡ 43b331ff-52b7-4513-a77f-3fce96a01917


# ╔═╡ Cell order:
# ╟─524496e1-6550-499d-9243-910d5931d28e
# ╠═0432f183-1e16-4934-9a7a-13f5050ff1a6
# ╠═d6c3293e-e2c9-4233-af04-c521d1c77d91
# ╠═d8af51e2-211b-487b-90da-0a41a096652c
# ╠═2b08b8bc-93f6-4f2c-aa21-f69087e102b4
# ╠═663ae283-a756-4545-b70b-64e9fe93340e
# ╠═4ae2fccd-6676-4609-b692-7a827746820d
# ╠═97b39b48-0a44-4ba3-bbd9-5a91cf1041fd
# ╠═eec67984-1b6c-446e-b06d-26e7f9b5cd39
# ╠═6e40c123-2acc-4532-9f91-02f150c8f0ca
# ╠═9b975ff7-e9a3-4e5d-8394-a653555d01a0
# ╠═0c4d3d9a-e335-4766-b2f4-507aa7b97a9d
# ╠═19563b94-9de2-4d50-8269-f7a3051e0620
# ╠═c0c98fc2-c5d2-4e45-8ee6-ef702a1e1c6f
# ╠═e819b552-5dec-4b9c-8d1c-490fa4023491
# ╟─f8689fc2-3e64-49a8-a325-e9064408702f
# ╠═4cc4f230-e91e-45a0-9462-af33c161841c
# ╠═263a73e0-018a-4360-a433-1a47322dee3d
# ╠═a7051003-c692-439d-bc27-cf07e9e0c255
# ╟─fbfbc736-13b0-4e26-8f0d-aeea50d5efb1
# ╠═766d55ab-6ec4-4d36-9032-3e5f7d6946b3
# ╠═9f715cfa-d767-45e7-950f-9ad9d59f6fc5
# ╠═0da308b1-3ef3-4596-8060-644a2637f141
# ╠═0e105c5b-4538-4c41-91c2-f2ed7a2d0d44
# ╠═6f908076-f7a1-4668-9040-0946cda0fa9e
# ╠═beecea10-c1c0-47d2-a0b5-a8218e55a969
# ╠═5cd4777c-3dcc-4ec2-8809-d5c02fbaa403
# ╠═fb03e48b-e09c-42cf-a9a9-15af7dd2ba24
# ╠═a238b43d-b497-4b76-baec-020f53bf4117
# ╠═db1d8c5f-3439-40ad-bd35-26d22bd71efe
# ╠═8002873d-91b0-4d8b-be19-eeb186d0a357
# ╠═a99ddc6c-6d6f-4234-a99c-e27cd14c6d47
# ╠═a5ceee9e-7871-4dc8-b7a2-5163a1dd4c0c
# ╠═0b8a260c-e439-44a3-b030-972df44dbb41
# ╠═3d82d934-84d8-4ed0-936e-c57e8296f19b
# ╠═c13ebce2-0b73-4763-8cce-990a91bd570a
# ╠═587d348c-479b-4275-9a47-0f70443dafe5
# ╠═b29bfcb0-adc9-4fe9-b5b3-3c44b0981afc
# ╠═97b1f361-da6d-4ec8-a469-d7ce84b1ebc5
# ╠═dcb876db-4bf6-4f6f-b92a-4486b011231e
# ╠═8b90735d-b9d4-4c46-9f1e-0b4e6a457c31
# ╠═b70689dd-0fcb-492a-83d6-86a4da103607
# ╠═ccd51211-24af-4b9c-84d4-8f6a2021a751
# ╠═e3062a04-4cc8-4918-970d-c68aaf4e006c
# ╟─c244f295-6c75-4060-be9e-54c45eb80aa3
# ╟─110f31d4-f181-47b3-8fe8-b57e26621db6
# ╠═2e4be690-1a3b-461f-b4c3-f85db0e49d89
# ╠═58c0678e-0d1f-428e-b610-49cd0661e76c
# ╠═9eb2e94e-7229-4616-9c77-329ddc31ef43
# ╠═c41ab60e-8b02-4186-9922-5be25d24a79f
# ╠═c4668ea7-d8dd-4cdf-b514-c08b8eb42c4a
# ╠═69c09862-8d0c-48d3-aef0-5428f1e6abe2
# ╠═accef95d-ffae-4a6a-a414-86171824b265
# ╠═d433649f-bf9f-4c4c-aa58-a5d9eb1e7c4a
# ╠═1cfcaa78-968f-4794-8cf1-ccb79f55f9c1
# ╠═b7f58b7e-c9de-4915-b7bd-e0bf2d539f11
# ╠═e07d5b2b-bee9-41ff-bbd9-72de2519e580
# ╠═35e856e2-3c9a-431b-a283-154f143a10cf
# ╠═cdc87aa9-f448-426c-974a-9b410cbbfca5
# ╠═922f2399-f995-48ed-9d64-413b68e8faba
# ╠═059a69e3-1749-4898-87ed-6e8690585002
# ╠═4c4f7cf4-722d-4731-998c-61d82355c0e6
# ╠═7eff2c57-cbe8-4913-aca3-897444249d8b
# ╠═d9d47c0d-5f09-4d74-95a3-3eb3d47d0d67
# ╠═a95651fe-1392-4476-93cf-ac0ad29fd3ce
# ╠═679a8015-8323-4f1f-b36b-73afa6cf4c06
# ╠═9dc6b706-2f93-4e7b-9704-af7a7dba5833
# ╟─574f8530-830f-4d14-a96b-b1f4cd10b9cf
# ╠═7f9ded89-d120-4242-9167-601435dfadc4
# ╠═169ae1d1-8ff7-4f44-8506-2c448b126066
# ╠═93bfd7fe-6f31-4456-91f4-4bcc06e23e56
# ╠═e0ad0ec5-e3c9-45b1-b654-8ae61143afc4
# ╠═e5037692-5321-4bcf-b513-d3e36f7a0006
# ╟─384c4659-99d3-4488-92d4-22945301e805
# ╠═43b331ff-52b7-4513-a77f-3fce96a01917
