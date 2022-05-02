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

# ╔═╡ 574f8530-830f-4d14-a96b-b1f4cd10b9cf
md"## The Strategy"

# ╔═╡ e5037692-5321-4bcf-b513-d3e36f7a0006
counter

# ╔═╡ 384c4659-99d3-4488-92d4-22945301e805
md"""## Exercises
### The Model
"""

# ╔═╡ 43b331ff-52b7-4513-a77f-3fce96a01917
function prob_correct(ability, difficulty)
	a = 100
	c = 0.25
	x = (ability - difficulty) / a
	p = c + (1 - c) / (1 + exp(-x))
end

# ╔═╡ 22e11f32-4c8f-4fcf-9de0-17cdfbaf41b6
begin
	abilities = 100:900
	diff = 500
	ps = [prob_correct(ability, diff) for ability in abilities]
end

# ╔═╡ ecf17617-b1d0-45bf-902d-6c658ebe079b
plot(abilities, ps)

# ╔═╡ 4ea20eb3-9c56-4f85-a926-f018a6f1d5b3
md"### Simulating the Test"

# ╔═╡ bb3add72-161a-4410-aed2-8330e036e4da
function play_sat(ability, difficulty)
	p = prob_correct(ability, difficulty)
	rand() < p
end

# ╔═╡ 0076d91a-083f-4d90-80a2-c589b001e8c0
outcomes = [play_sat(600, 500) for _ in 1:51]

# ╔═╡ 562bc715-92ee-4760-8cc8-adcf60a88473
mean(outcomes)

# ╔═╡ 3be4a7c5-23e0-482d-83a8-032e01f8f8d2
md"### The Prior"

# ╔═╡ bd279551-6d68-46bd-8a67-c44c6c98be90
begin
	mean_ability = 500.0
	std_ability = 300.0
	qs = LinRange(0, 1000, 50)
	prior_ability = make_normal_pmf(qs, mu=mean_ability, sigma=std_ability)
end;

# ╔═╡ e1e47a59-aa3a-41c2-bbba-71f87e0fd909
plot(prior_ability)

# ╔═╡ cb656c48-0af8-4ae1-97ba-aba70b7dd33c
md"### The Update"

# ╔═╡ d2386d59-c373-4624-8f6a-e338d6b16d4b
function update_ability(pmf, data)
	(difficulty, outcome) = data
	abilities = values(pmf)
	ps = [prob_correct(ability, difficulty) for ability in abilities]
	if outcome
		return pmf * ps
	else
		return pmf * (1 .- ps)
	end
end


# ╔═╡ bf1c5610-ab1c-49c9-a3c2-950eab01fdb7
actual_600 = copy(prior_ability);

# ╔═╡ f17874bc-adfe-461d-b830-ee74bc3ff9c5


# ╔═╡ 0ae1b1d7-aa99-4857-bc12-dd5c04ce7db1
begin
	ac = [actual_600]
	for outcome in outcomes
		data = (500, outcome)
		ac[1] = update_ability(ac[1], data)
	end
end

# ╔═╡ e747c36b-6a71-442f-9125-15dfb1cac9ce
plot(ac[1])

# ╔═╡ a9906311-e051-4f7b-ab0e-c825700bb404
mean(ac[1])

# ╔═╡ ff652039-979e-46c7-a437-8fd153bba144
md"### Adaptation"

# ╔═╡ 3179d6fe-b83e-4ca5-bd5d-b9b59f1c90ec
function choose(i, belief)
	return 500
end

# ╔═╡ 9dc6b706-2f93-4e7b-9704-af7a7dba5833
choose(beliefs)

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

# ╔═╡ 554d41b2-bd00-4420-af60-a7c4a6d2c757
function simulate_test(actual_ability, num_questions; choose_func=choose)
	belief = [copy(prior_ability)]
	function st(i)
		difficulty = choose_func(i, belief[1])
		outcome = play_sat(actual_ability, difficulty)
		data = (difficulty, outcome)
		belief[1] = update_ability(belief[1], data)
		(difficulty, outcome)
	end
	vals = [st(i) for i in 1:num_questions]
	trace = DataFrame(difficulty=getindex.(vals, 1), outcome=getindex.(vals, 2))
	return (belief[1], trace)
end

# ╔═╡ c4caf98e-b8c0-4635-b758-6dcf506d0ed9
(belief, trace) = simulate_test(600, 51);

# ╔═╡ f109ebdf-cb54-437c-8024-67124a89262c
sum(trace.outcome)

# ╔═╡ 8382e3d8-47e5-4b85-a897-e5f4e3960e7b
plot(belief)

# ╔═╡ e047cec0-21eb-4009-a346-37e0af426b98
md"### Quantifying Precision"

# ╔═╡ 490f5418-b674-4f52-902c-8de526660bb5
mean(belief), std(belief)

# ╔═╡ 15dc191e-2fca-406a-a4ed-7ff4c8f03813
begin
	actual_abilities = LinRange(200, 800, 50)
	function do_st(ability)
		belief, trace = simulate_test(ability, 51)
		std(belief)
	end
	series = [do_st(ability) for ability in actual_abilities]
end

# ╔═╡ 1b74c680-c03f-49a7-8ce5-4fb54bfbe444
scatter(actual_abilities, series)

# ╔═╡ aa746610-3c77-44d4-8aff-faa8599e4d0c
md"### Discriminatory Power"

# ╔═╡ aeb69ae1-e734-43f8-8cde-ed4150d7638c
function sample_posterior(actual_ability; iters=100, choose_func=choose)
	function do_st()
		belief, trace = simulate_test(actual_ability, 51, choose_func=choose_func)
		mean(belief)
	end
	[do_st() for i in 1:iters]
end

# ╔═╡ 009db74d-aa2d-45e8-a17f-30bffae7e774
sample_500 = sample_posterior(500)

# ╔═╡ fb207dd0-93f2-4247-a7c9-85bc2b193170
sample_600 = sample_posterior(600)

# ╔═╡ 3da1e00c-f1d8-4b5d-b4c1-ba91ad57242c
sample_700 = sample_posterior(700)

# ╔═╡ 4e8676bb-e5be-4a0e-8b6c-726d3cd6da2c
sample_800 = sample_posterior(800)

# ╔═╡ f43e52f9-0bd0-461e-bf68-0134cc7031bd
cdf_500 = cdf_from_seq(sample_500)

# ╔═╡ 51579977-d451-4b2d-94c9-b1c71b27e286
cdf_600 = cdf_from_seq(sample_600)

# ╔═╡ 8f38630e-b042-438e-8f83-5a330743ffe9
cdf_700 = cdf_from_seq(sample_700)

# ╔═╡ 54a85836-a20d-4fdf-b7a0-6e6bcefd6c49
cdf_800 = cdf_from_seq(sample_800)

# ╔═╡ 49702532-7fea-4dde-9e34-4768928c9a3c
begin
	plot(cdf_500, label="500")
	plot!(cdf_600, label="600")
	plot!(cdf_700, label="700")
	plot!(cdf_800, label="800")
end

# ╔═╡ 59658096-7075-4590-89a1-f90d67243585
mean(sample_600 .> sample_500)

# ╔═╡ f75489c9-ba14-45ac-96b0-1bfb969ee04f
mean(sample_700 .> sample_600)

# ╔═╡ 2a346886-fbe7-4b48-9faf-400d9d2d3307
difficulties = LinRange(200, 800, 51)

# ╔═╡ 898a2b2b-a740-4e1b-ac49-003d676a4a08
function choose1(i, belief)
	difficulties[i]
end

# ╔═╡ c83e046f-2517-4915-9f33-2bbedc0adb48
function choose2(i, belief)
	mean(belief)
end

# ╔═╡ 93aae2b9-f35e-44ca-9663-ac624c8569df
begin
	sample_5001 = sample_posterior(500, choose_func=choose1)
	sample_5002 = sample_posterior(500, choose_func=choose2)
	sample_6001 = sample_posterior(600, choose_func=choose1)
	sample_6002 = sample_posterior(600, choose_func=choose2)
	sample_7001 = sample_posterior(700, choose_func=choose1)
	sample_7002 = sample_posterior(700, choose_func=choose2)
	sample_8001 = sample_posterior(800, choose_func=choose1)
	sample_8002 = sample_posterior(800, choose_func=choose2)
	cdf_5001 = cdf_from_seq(sample_5001)
	cdf_5002 = cdf_from_seq(sample_5002)
	cdf_6001 = cdf_from_seq(sample_6001)
	cdf_6002 = cdf_from_seq(sample_6002)
	cdf_7001 = cdf_from_seq(sample_7001)
	cdf_7002 = cdf_from_seq(sample_7002)
	cdf_8001 = cdf_from_seq(sample_8001)
	cdf_8002 = cdf_from_seq(sample_8002)
end

# ╔═╡ 40be469e-cd7e-4a41-a851-68f1bbb61ae7
begin
	plot(cdf_5001, label="5001")
	plot!(cdf_6001, label="6001")
	plot!(cdf_7001, label="7001")
	plot!(cdf_8001, label="8001")
end

# ╔═╡ 40ecc4e1-a822-4a03-8425-f624c79baf9e
begin
	plot(cdf_5002, label="5002")
	plot!(cdf_6002, label="6002")
	plot!(cdf_7002, label="7002")
	plot!(cdf_8002, label="8002")
end

# ╔═╡ daf691fb-6d3e-4262-a9ec-5f96f1b23ab2
mean(sample_6001 .> sample_5001)

# ╔═╡ 4eda627b-0d71-4008-a6de-f74bd813af4e
mean(sample_7001 .> sample_6001)


# ╔═╡ f9d01704-0105-4009-bb4f-a8a114e56b79
mean(sample_8001 .> sample_7001)


# ╔═╡ 17fafffa-bf55-44e8-8597-d4d07f54d32b
mean(sample_6002 .> sample_5002)


# ╔═╡ 846aa029-aeb5-4f40-9443-2cc62d76534b
mean(sample_7002 .> sample_6002)


# ╔═╡ 071d7b7b-2b6f-42a2-80d4-77fdee6c7ca0
mean(sample_8002 .> sample_7002)


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
# ╠═22e11f32-4c8f-4fcf-9de0-17cdfbaf41b6
# ╠═ecf17617-b1d0-45bf-902d-6c658ebe079b
# ╟─4ea20eb3-9c56-4f85-a926-f018a6f1d5b3
# ╠═bb3add72-161a-4410-aed2-8330e036e4da
# ╠═0076d91a-083f-4d90-80a2-c589b001e8c0
# ╠═562bc715-92ee-4760-8cc8-adcf60a88473
# ╟─3be4a7c5-23e0-482d-83a8-032e01f8f8d2
# ╠═bd279551-6d68-46bd-8a67-c44c6c98be90
# ╠═e1e47a59-aa3a-41c2-bbba-71f87e0fd909
# ╟─cb656c48-0af8-4ae1-97ba-aba70b7dd33c
# ╠═d2386d59-c373-4624-8f6a-e338d6b16d4b
# ╠═bf1c5610-ab1c-49c9-a3c2-950eab01fdb7
# ╠═f17874bc-adfe-461d-b830-ee74bc3ff9c5
# ╠═0ae1b1d7-aa99-4857-bc12-dd5c04ce7db1
# ╠═e747c36b-6a71-442f-9125-15dfb1cac9ce
# ╠═a9906311-e051-4f7b-ab0e-c825700bb404
# ╟─ff652039-979e-46c7-a437-8fd153bba144
# ╠═3179d6fe-b83e-4ca5-bd5d-b9b59f1c90ec
# ╠═554d41b2-bd00-4420-af60-a7c4a6d2c757
# ╠═c4caf98e-b8c0-4635-b758-6dcf506d0ed9
# ╠═f109ebdf-cb54-437c-8024-67124a89262c
# ╠═8382e3d8-47e5-4b85-a897-e5f4e3960e7b
# ╟─e047cec0-21eb-4009-a346-37e0af426b98
# ╠═490f5418-b674-4f52-902c-8de526660bb5
# ╠═15dc191e-2fca-406a-a4ed-7ff4c8f03813
# ╠═1b74c680-c03f-49a7-8ce5-4fb54bfbe444
# ╟─aa746610-3c77-44d4-8aff-faa8599e4d0c
# ╠═aeb69ae1-e734-43f8-8cde-ed4150d7638c
# ╠═009db74d-aa2d-45e8-a17f-30bffae7e774
# ╠═fb207dd0-93f2-4247-a7c9-85bc2b193170
# ╠═3da1e00c-f1d8-4b5d-b4c1-ba91ad57242c
# ╠═4e8676bb-e5be-4a0e-8b6c-726d3cd6da2c
# ╠═f43e52f9-0bd0-461e-bf68-0134cc7031bd
# ╠═51579977-d451-4b2d-94c9-b1c71b27e286
# ╠═8f38630e-b042-438e-8f83-5a330743ffe9
# ╠═54a85836-a20d-4fdf-b7a0-6e6bcefd6c49
# ╠═49702532-7fea-4dde-9e34-4768928c9a3c
# ╠═59658096-7075-4590-89a1-f90d67243585
# ╠═f75489c9-ba14-45ac-96b0-1bfb969ee04f
# ╠═2a346886-fbe7-4b48-9faf-400d9d2d3307
# ╠═898a2b2b-a740-4e1b-ac49-003d676a4a08
# ╠═c83e046f-2517-4915-9f33-2bbedc0adb48
# ╠═93aae2b9-f35e-44ca-9663-ac624c8569df
# ╠═40be469e-cd7e-4a41-a851-68f1bbb61ae7
# ╠═40ecc4e1-a822-4a03-8425-f624c79baf9e
# ╠═daf691fb-6d3e-4262-a9ec-5f96f1b23ab2
# ╠═4eda627b-0d71-4008-a6de-f74bd813af4e
# ╠═f9d01704-0105-4009-bb4f-a8a114e56b79
# ╠═17fafffa-bf55-44e8-8597-d4d07f54d32b
# ╠═846aa029-aeb5-4f40-9443-2cc62d76534b
# ╠═071d7b7b-2b6f-42a2-80d4-77fdee6c7ca0
