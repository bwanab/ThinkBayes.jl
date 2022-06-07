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

# ╔═╡ 55641895-d5ee-40b4-92b2-cafbd61375e4
md""" ## Implementation note:

A very common pattern that is used in ThinkBayes2 is 

1. Create a mesh of two or three vectors
2. Compute densities from the resulting meshs.
3. Compute a likelihood from those densities

After some searching, I found that mesh is considered unjulian that it's better to
use matrix operations directly. With some experimentation I've determined the julian pattern to be:

1. create a function on two variables that does a single computation of 3 (above).
2. Use the outer function to build the full data.

Thus, where ThinkBayes2 would have:

>val = 100

>amesh, bmesh = np.meshgrid(as, bs)

>likelihoods = normal(amesh, bmesh).pdf(val)

This implementation uses

>nfunc(mu, sigma) = pdf(Normal(mu, sigma))

>likelihoods = outer(nfunc, as, bs)

Or, in a more complex example:

>amesh, bmesh, cmesh = np.meshgrid(as, bs, cs)

>densities = normal(amesh, bmesh).pdf(cmesh) # results in a 3 dimensional array.

>likelihoods = densities.prod(axis=2)

This implementation uses:

>nfunc(mu, sigma) = prod([pdf(Normal(mu, sigma), x) for x in cs]

>likelihoods = outer(nfunc, as, bs)
"""

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
d = [0.80497283, 2.11577082, 0.43308797, 0.10862644, 5.17334866,
     3.25745053, 3.05555883, 2.47401062, 0.05340806, 1.08386395]

# ╔═╡ 55a7646f-69ef-4d06-ab08-bb7d86c43fb2
function update_weibull(prior, data; f=pdf)
	wd_func(l, k) = prod([f(Weibull(k, l), x) for x in data])
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

# ╔═╡ 2ee0660c-19b1-4748-bb24-52091c6da8ab
begin
	# u = Uniform(0,8)
	# start = rand(u, 10)
	# here, again, I'm using numbers from ThinkBayes2 for verification of numbers
	start = [0.78026881, 6.08999773, 1.97550379, 1.1050535 , 2.65157251,
       0.66399652, 5.37581665, 6.45275039, 7.86193532, 5.08528588]
end

# ╔═╡ 9249885f-ed8b-4002-ab40-df08780698ae
begin
	# duration = rand(dist, 10)
	# and again
	duration = [0.80497283, 2.11577082, 0.43308797, 0.10862644, 5.17334866,
       3.25745053, 3.05555883, 2.47401062, 0.05340806, 1.08386395]
end

# ╔═╡ 8ec0dca6-ee3a-4b19-8fbf-5e14c53dd210
obs = DataFrame(start = start, endt = start.+duration)

# ╔═╡ 3e186339-d999-49ae-92ab-22ebca75b676
begin
	sort!(obs, [:start])
	obs.Index = 1:length(start)
end

# ╔═╡ 15dbc762-1ced-4ca3-8e39-403e008e405a
obs[findall(>(8), obs.endt), :endt] .= 8

# ╔═╡ 6451666e-0ca1-410d-9b0d-0c55a23087b8
begin
	obs.status .= 1
	obs[findall(==(8.0), obs.endt), :status] .= 0
end

# ╔═╡ a638a5c8-0224-4a29-a9dc-3f377a56e7bf
s, e = obs[1, [:start, :endt]]

# ╔═╡ 14207112-1205-43dc-8d5a-44b7aa0745ba
e

# ╔═╡ defc3f45-f9f0-4803-8637-ec3caec27ee7
function plot_lifelines(obs)
	plot()
	for i in 1:nrow(obs)
		start, endt, status = obs[i,[:start, :endt, :status]]
		if status == 0
			plot!([start, endt], [i, i], label=nothing, c=:green)
		else
			plot!([start, endt], [i, i], label=nothing, c=:orange)
			scatter!([endt], [i], label=nothing, c=:orange)
		end
	end
	yflip!()
	plot!(title="Lifelines showing censored and uncensored observations", xaxis=("Time (weeks)"), yaxis=("Dog index"))
	#legend!(:right)
end

# ╔═╡ 73a71198-5be9-4372-841e-f70d3a03fd61
plot_lifelines(obs)

# ╔═╡ 9b8db260-8c0c-4791-bb53-9863e093169f
obs.T = obs.endt - obs.start

# ╔═╡ 7e381341-208b-4ce5-9d9d-d4754f07c671
md"## Using Incomplete Data"

# ╔═╡ 0e3db40d-a131-42e1-82a5-80570edf86f7
begin
	data1 = obs[findall(==(1), obs.status), [:Index, :T]]
	data2 = obs[findall(==(0), obs.status), [:Index, :T]]
end

# ╔═╡ b4194bd6-5bb3-4bca-8110-c379b35ee3a7
posterior1 = update_weibull(prior, data1.T);

# ╔═╡ 07d332ed-76f5-4b01-ab39-509ef9612295
posterior2 = update_weibull(posterior1, data2.T, f=ccdf);

# ╔═╡ 1ea0af0b-d59a-4a06-9c8e-91054f3b5e84
contour(posterior2, size=[800, 500])

# ╔═╡ 0e4063d0-e22f-44bf-afe9-d995b3089bce
begin
	posterior_lam2 = marginal(posterior2, 1)
	posterior_k2 = marginal(posterior2, 2)
end;

# ╔═╡ 7f9e5a78-ee23-4cb6-ad2a-9b0c7e40a525
begin
	plot()
	plot!(posterior_lam, label="All complete")
	plot!(posterior_lam2, label="Some censored")
end

# ╔═╡ 46cd7867-535f-4529-bade-ed291af492cf
begin
	plot()
	plot!(posterior_k, label="All Complete")
	plot!(posterior_k2, label="Some censored")
end

# ╔═╡ da8dbaf3-f60c-4724-8dd9-931824b8dd9a
md"## Light Bulbs"

# ╔═╡ fec484b1-489f-460c-b261-a0b910619e41
begin
	df = DataFrame(CSV.File(download("https://gist.github.com/epogrebnyak/7933e16c0ad215742c4c104be4fbdeb1/raw/c932bc5b6aa6317770c4cbf43eb591511fec08f9/lamps.csv")))
	first(df, 6)
end

# ╔═╡ 5a33aae0-dadb-444a-94fb-abc6bc80693a
pmf_bulb = pmf_from_seq(df.h, counts=df.f);

# ╔═╡ c0f6ed4a-918b-40b8-a99e-a05244977daa
mean(pmf_bulb)

# ╔═╡ bdb2e4d3-f70e-49a4-8bf8-22504b9bd239
prior_lb_lam = pmf_from_seq(range(1000, 2000, 51));

# ╔═╡ 690e00d0-b95a-4ac9-ae76-9989d2a94805
prior_lb_k = pmf_from_seq(range(1, 10, 51));

# ╔═╡ c35cb2f7-a085-4fda-b6a4-8f8a97b0ef99
prior_bulb = make_joint(*, prior_lb_lam, prior_lb_k);

# ╔═╡ c8e8a8c8-a98d-4738-b5f6-a78ab9c32de8
data_bulb = reduce(vcat, fill.(df.h, df.f))

# ╔═╡ 4e3475de-3bb0-4bd1-ba10-1eaabd80effb
posterior_bulb = update_weibull(prior_bulb, data_bulb);

# ╔═╡ 27fab0c7-519a-4545-8163-2de5dc8fcde5
contour(posterior_bulb, size=(700, 500))

# ╔═╡ 82c06b5b-d25c-42fb-b371-1bc4853e6aa1
md"## Posterior Means"

# ╔═╡ 0829edc2-16b7-46dd-b8ba-1d88253b3f50
begin
	post_mean(λ, k) = mean(Weibull(k, λ))
	means = outer(post_mean, index(prior_bulb), columns(prior_bulb))
end;

# ╔═╡ 1afe1b94-6a2a-487b-bf2e-bbd5dfb3a3a9
prod_lb = posterior_bulb.M .* means;

# ╔═╡ e0e0eb36-0b74-4321-8b4c-b46c37a07a45
sum(prod_lb)

# ╔═╡ 444fd409-36a9-4474-a0b6-764bc63d7349
function joint_weibull_mean(joint::Joint)
	post_mean(λ, k) = mean(Weibull(k, λ))
	means = outer(post_mean, index(joint), columns(joint))
	prod = joint.M .* means
	sum(prod)
end

# ╔═╡ c62a924e-8850-4b49-aa9a-73522a97e853
joint_weibull_mean(posterior_bulb)

# ╔═╡ 96e27f0f-cb7c-4b09-9102-00049ffa41c5
md"## Incomplete Information"

# ╔═╡ b4160ea6-36be-4901-b2d3-1e876690c46b
function update_weibull_between(prior, data)
	function wd_func(l, k)
		dist = Weibull(k, l)
		cdf1 = [cdf(dist, x) for x in data]
		cdf2 = [cdf(dist, x-12) for x in data]
		prod(cdf1 .- cdf2)
	end
	likelihood = outer(wd_func, index(prior), columns(prior))
	prior * likelihood
end

# ╔═╡ fc0a31d1-4448-4118-beb8-d759ffb02dc4
posterior_bulb2 = update_weibull_between(prior_bulb, data_bulb);

# ╔═╡ a3a46bf3-0f74-4e48-92ec-bdbf018dd0b8
contour(posterior_bulb, title="joint posterior distribution, light bulbs", xaxis=("λ"), yaxis=("k"))

# ╔═╡ c905b96b-86ee-41d7-bd4c-1842d7fc6ee4
joint_weibull_mean(posterior_bulb2), joint_weibull_mean(posterior_bulb)

# ╔═╡ 64bc5799-75a1-4d27-acd2-5066643533dc
md"## Posterior Predictive Distribution"

# ╔═╡ 6b723760-1509-4b4b-bc68-dffb275900f3
md"Given a λ = 1550, k = 4.25, how many bulbs will be dead after 1000 hours?"

# ╔═╡ 93dddef3-08d3-4f14-8fbd-ca59ade80276
prob_dead = cdf(Weibull(4.25, 1550), 1000)

# ╔═╡ d82aff82-1359-429b-a702-e4a45ae13046
md"Given n = 100 bulbs with this probability of dying:"

# ╔═╡ 88a6a86f-6ac4-440b-be02-3642a4218e32
dist_num_dead = make_binomial(100, prob_dead);

# ╔═╡ d6c992db-ea62-4d71-b886-79ee13bd3213
plot(dist_num_dead, label="known parameters", xaxis=("Number of dead bulbs"), yaxis=("PMF"), plot_title="Predictive distribution with known parameters")

# ╔═╡ 24a5e8e2-b4ad-4ed7-9925-37a57623dc51
ps_index, ps_vals = stack(posterior_bulb);

# ╔═╡ 65e2c032-e16c-4140-b0eb-1725f50a87a2


# ╔═╡ f3d44c45-c421-4826-aaf0-3b79390c76c0
begin
	function make_pmf_seq(ps, t, n)
		λ, k = ps
		prob_dead = cdf(Weibull(k, λ), t)
		make_binomial(n, prob_dead)
	end
	pmf_seq = [make_pmf_seq(x, 1000, 100) for x in ps_index]
end;

# ╔═╡ ccb060f3-b6bf-4d03-8db5-02ab7fbe0cbb
post_pred = make_mixture(pmf_from_seq(ps_index, ps_vals), pmf_seq);

# ╔═╡ c2cc93b6-a1da-47e7-81b2-c89465c5beb1
begin
	plot(dist_num_dead, label="known parameters")
	plot!(pmf_from_seq(1:101, probs(post_pred)), label="unknown parameters")
	plot!(plot_title="Posterior predictive distribution", xaxis=("Number of dead bulbs"), yaxis=("PMF"))
end

# ╔═╡ 371039d5-41c2-4c3b-b9fb-5c35008b8b0d
md"""## Exercises

_exercise 14.1_"""

# ╔═╡ 62cb708a-4373-4e94-9fe3-51d532f32116
begin
	t = 1000
	prob_dead_func(λ, k) = cdf(Weibull(k, λ), t)
	pd = outer(prob_dead_func, index(prior_bulb), columns(prior_bulb))
end

# ╔═╡ bd2cf7c6-c61a-44da-8431-4fe32e581661
begin
	k = 20
	n = 100
	likelihood = [pdf(Binomial(n, x), k) for x in pd]
end

# ╔═╡ 4eda111d-1b65-4ace-a2c8-d370041b3a24
begin
	posterior_bulb3 = posterior_bulb * likelihood
	contour(posterior_bulb3, size=(700,500))
end

# ╔═╡ 2f972e0e-3be5-41de-92f9-2a82ef6a8b59
joint_weibull_mean(posterior_bulb3)

# ╔═╡ 715c3888-a0ab-4c6b-9b97-288b0d41a5c8
md"_exercise 14.2_"

# ╔═╡ 67f69dd0-ae4b-4eb8-8ee7-e4deb3df5a82
begin
	weather = DataFrame(CSV.File(download("https://github.com/AllenDowney/ThinkBayes2/raw/master/data/2203951.csv")))
	first(weather, 6)
end

# ╔═╡ b1699203-4599-466b-baf9-0be73d28e000
weather.rained = weather.PRCP .> 0

# ╔═╡ afaae38a-b91a-4f1c-b3ae-0ee9231da299
sum(weather.rained)

# ╔═╡ 9ec5d38a-c9f7-4880-bdfc-39e82eb648ab
begin
	prcp = weather[weather.rained, "PRCP"]
	println("mean:", mean(prcp))
	println("std: ", std(prcp))
	println("min: ", minimum(prcp))
	println("max: ", maximum(prcp))
end

# ╔═╡ cc1b2a22-3cb0-4c12-86ae-08b175d6daff
begin
	cdf_data = cdf_from_seq(prcp)
	plot(cdf_data, xaxis=("total rainfall (in)"), yaxis=("CDF"), plot_title="Distribution of rainfall on days it rained")
end

# ╔═╡ 96bd3ade-94e5-4bf3-97af-6d7a1d0f737b
g = fit(Gamma, prcp)

# ╔═╡ ea7fc339-7325-4e00-8311-e40015a3a836
md"My first attempt was pretty lame :("

# ╔═╡ 841a4949-3e90-4be4-b410-a9f456529432
prior_rain_dist = rand(g, 51)

# ╔═╡ cf09e2f1-790a-4f16-bc7e-00ba7db7da17
mean(prior_rain_dist), std(prior_rain_dist)

# ╔═╡ 706eb874-fba7-4db9-8b37-4353997b8701
prior_rain_pmf = kde_from_sample(prior_rain_dist, 0.0, 1.5, 14);

# ╔═╡ 3d9878e9-1f2d-40c6-8186-aa6ed3ab7083
posterior_rain = prior_rain_pmf * prcp;

# ╔═╡ 12e2f574-0e32-46f4-a386-2abb2ed6f636
prob_ge(posterior_rain, 1.5)

# ╔═╡ a750ba38-28bf-4685-b2a6-b35ea45030e8
md"End of my first attempt - here's how Allen did it:"

# ╔═╡ 6feec10e-a79f-49f9-8713-49d54c5be705
prior_α = pmf_from_seq(range(0.01, 2, 51));

# ╔═╡ 9152632f-224c-4bb1-8974-9b006d79535f
prior_θ = pmf_from_seq(range(0.01, 1.5, 51));

# ╔═╡ fe200df5-61bd-4af5-8af2-05b8fcce5d3c
rain_prior = make_joint(*, prior_α, prior_θ);

# ╔═╡ 85888f00-4f0f-40a6-a301-78cf2b74d5f0
begin
	gamma_func(α, θ) = prod([pdf(Gamma(α, θ), d) for d in prcp])
	rain_likelihood = outer(gamma_func, index(rain_prior), columns(rain_prior))
end;

# ╔═╡ 35ea1d78-123a-47ef-a1ba-31aed02a1c4a
sum(rain_likelihood)

# ╔═╡ 68484602-712f-4668-8be5-dc55f7a6e5fd
posterior_rain2 = rain_prior * rain_likelihood;

# ╔═╡ e74412cc-3940-44f8-a5d4-43ae6ecc0f86
contour(posterior_rain2, yaxis=("θ"), xaxis=("α"), title="Posterior distribution, parameters of a gamma distribution", size=(700, 500))

# ╔═╡ fcf7e103-0dec-4597-968f-14e5ab8e9860
begin
	posterior_α = marginal(posterior_rain2, 1)
	posterior_θ = marginal(posterior_rain2, 2)
end;

# ╔═╡ 4a0aa532-e4b2-4249-82f4-d92fb7921474
plot(posterior_α, xaxis=("α"), yaxis=("PDF"), plot_title="Posterior marginal distribution of α")

# ╔═╡ aa1edd0f-4239-4ad5-b971-4e3e58b23fff
mean(posterior_α), credible_interval(posterior_α, 0.9)

# ╔═╡ b17b5d0b-04a9-48d9-9895-d24d802b3984
plot(posterior_θ, xaxis=("θ"), yaxis=("PDF"), plot_title="Posterior marginal distribution of θ")

# ╔═╡ 1886a63f-2b35-4222-a612-6f7e552f2f2d
mean(posterior_θ), credible_interval(posterior_θ, 0.9)

# ╔═╡ 2a8a410d-7c1c-4dfb-9206-dd9efbed2a9e
rain_index, rain_vals = stack(posterior_rain2)

# ╔═╡ 81e3b595-8025-40bb-8598-9a2ee9b65ed6
begin
	function make_pmf_gamma_seq(ps, qs)
		α, θ = ps
		dist = Gamma(α, θ)
		pmf_from_dist(qs, dist)
	end
	rain_qs = range(0.01, 2, 101)
	pmf_gamma_seq = [make_pmf_gamma_seq(x, rain_qs) for x in rain_index]
end;

# ╔═╡ ed6e1952-681a-4303-ba24-5d408fb9bedb
rain_post_pred = make_mixture(pmf_from_seq(rain_index, rain_vals), pmf_gamma_seq);

# ╔═╡ 070d7fc2-30d2-4150-9dd4-acc99dbfc3ea
begin
	rain_pmf = pmf_from_seq(rain_qs, probs(rain_post_pred))
	rain_cdf = make_cdf(rain_pmf)
	plot(rain_cdf)
	plot!(plot_title="Posterior predictive distribution of rainfall")
	plot!(xaxis=("Total rainfall (in)"), yaxis=("CDF"))
end

# ╔═╡ f4107cb8-68c0-49de-a2bb-3a670314837d
p_gt = prob_gt(rain_pmf, 1.5)

# ╔═╡ 580df68f-0165-4be7-84fa-23258f63d741
1 / p_gt

# ╔═╡ Cell order:
# ╠═11bfe214-db8f-11ec-24d9-21c6b862ea54
# ╠═a0bf7ab3-541d-4182-ba67-5de432a53ff9
# ╠═b240adda-091b-45ac-878d-302f7b18d610
# ╟─55641895-d5ee-40b4-92b2-cafbd61375e4
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
# ╠═2ee0660c-19b1-4748-bb24-52091c6da8ab
# ╠═9249885f-ed8b-4002-ab40-df08780698ae
# ╠═8ec0dca6-ee3a-4b19-8fbf-5e14c53dd210
# ╠═3e186339-d999-49ae-92ab-22ebca75b676
# ╠═15dbc762-1ced-4ca3-8e39-403e008e405a
# ╠═6451666e-0ca1-410d-9b0d-0c55a23087b8
# ╠═a638a5c8-0224-4a29-a9dc-3f377a56e7bf
# ╠═14207112-1205-43dc-8d5a-44b7aa0745ba
# ╠═defc3f45-f9f0-4803-8637-ec3caec27ee7
# ╠═73a71198-5be9-4372-841e-f70d3a03fd61
# ╠═9b8db260-8c0c-4791-bb53-9863e093169f
# ╟─7e381341-208b-4ce5-9d9d-d4754f07c671
# ╠═0e3db40d-a131-42e1-82a5-80570edf86f7
# ╠═b4194bd6-5bb3-4bca-8110-c379b35ee3a7
# ╠═07d332ed-76f5-4b01-ab39-509ef9612295
# ╠═1ea0af0b-d59a-4a06-9c8e-91054f3b5e84
# ╠═0e4063d0-e22f-44bf-afe9-d995b3089bce
# ╠═7f9e5a78-ee23-4cb6-ad2a-9b0c7e40a525
# ╠═46cd7867-535f-4529-bade-ed291af492cf
# ╟─da8dbaf3-f60c-4724-8dd9-931824b8dd9a
# ╠═fec484b1-489f-460c-b261-a0b910619e41
# ╠═5a33aae0-dadb-444a-94fb-abc6bc80693a
# ╠═c0f6ed4a-918b-40b8-a99e-a05244977daa
# ╠═bdb2e4d3-f70e-49a4-8bf8-22504b9bd239
# ╠═690e00d0-b95a-4ac9-ae76-9989d2a94805
# ╠═c35cb2f7-a085-4fda-b6a4-8f8a97b0ef99
# ╠═c8e8a8c8-a98d-4738-b5f6-a78ab9c32de8
# ╠═4e3475de-3bb0-4bd1-ba10-1eaabd80effb
# ╠═27fab0c7-519a-4545-8163-2de5dc8fcde5
# ╟─82c06b5b-d25c-42fb-b371-1bc4853e6aa1
# ╠═0829edc2-16b7-46dd-b8ba-1d88253b3f50
# ╠═1afe1b94-6a2a-487b-bf2e-bbd5dfb3a3a9
# ╠═e0e0eb36-0b74-4321-8b4c-b46c37a07a45
# ╠═444fd409-36a9-4474-a0b6-764bc63d7349
# ╠═c62a924e-8850-4b49-aa9a-73522a97e853
# ╟─96e27f0f-cb7c-4b09-9102-00049ffa41c5
# ╠═b4160ea6-36be-4901-b2d3-1e876690c46b
# ╠═fc0a31d1-4448-4118-beb8-d759ffb02dc4
# ╠═a3a46bf3-0f74-4e48-92ec-bdbf018dd0b8
# ╠═c905b96b-86ee-41d7-bd4c-1842d7fc6ee4
# ╟─64bc5799-75a1-4d27-acd2-5066643533dc
# ╟─6b723760-1509-4b4b-bc68-dffb275900f3
# ╠═93dddef3-08d3-4f14-8fbd-ca59ade80276
# ╟─d82aff82-1359-429b-a702-e4a45ae13046
# ╠═88a6a86f-6ac4-440b-be02-3642a4218e32
# ╠═d6c992db-ea62-4d71-b886-79ee13bd3213
# ╠═24a5e8e2-b4ad-4ed7-9925-37a57623dc51
# ╠═65e2c032-e16c-4140-b0eb-1725f50a87a2
# ╠═f3d44c45-c421-4826-aaf0-3b79390c76c0
# ╠═ccb060f3-b6bf-4d03-8db5-02ab7fbe0cbb
# ╠═c2cc93b6-a1da-47e7-81b2-c89465c5beb1
# ╟─371039d5-41c2-4c3b-b9fb-5c35008b8b0d
# ╠═62cb708a-4373-4e94-9fe3-51d532f32116
# ╠═bd2cf7c6-c61a-44da-8431-4fe32e581661
# ╠═4eda111d-1b65-4ace-a2c8-d370041b3a24
# ╠═2f972e0e-3be5-41de-92f9-2a82ef6a8b59
# ╟─715c3888-a0ab-4c6b-9b97-288b0d41a5c8
# ╠═67f69dd0-ae4b-4eb8-8ee7-e4deb3df5a82
# ╠═b1699203-4599-466b-baf9-0be73d28e000
# ╠═afaae38a-b91a-4f1c-b3ae-0ee9231da299
# ╠═9ec5d38a-c9f7-4880-bdfc-39e82eb648ab
# ╠═cc1b2a22-3cb0-4c12-86ae-08b175d6daff
# ╠═96bd3ade-94e5-4bf3-97af-6d7a1d0f737b
# ╟─ea7fc339-7325-4e00-8311-e40015a3a836
# ╠═841a4949-3e90-4be4-b410-a9f456529432
# ╠═cf09e2f1-790a-4f16-bc7e-00ba7db7da17
# ╠═706eb874-fba7-4db9-8b37-4353997b8701
# ╠═3d9878e9-1f2d-40c6-8186-aa6ed3ab7083
# ╠═12e2f574-0e32-46f4-a386-2abb2ed6f636
# ╟─a750ba38-28bf-4685-b2a6-b35ea45030e8
# ╠═6feec10e-a79f-49f9-8713-49d54c5be705
# ╠═9152632f-224c-4bb1-8974-9b006d79535f
# ╠═fe200df5-61bd-4af5-8af2-05b8fcce5d3c
# ╠═85888f00-4f0f-40a6-a301-78cf2b74d5f0
# ╠═35ea1d78-123a-47ef-a1ba-31aed02a1c4a
# ╠═68484602-712f-4668-8be5-dc55f7a6e5fd
# ╠═e74412cc-3940-44f8-a5d4-43ae6ecc0f86
# ╠═fcf7e103-0dec-4597-968f-14e5ab8e9860
# ╠═4a0aa532-e4b2-4249-82f4-d92fb7921474
# ╠═aa1edd0f-4239-4ad5-b971-4e3e58b23fff
# ╠═b17b5d0b-04a9-48d9-9895-d24d802b3984
# ╠═1886a63f-2b35-4222-a612-6f7e552f2f2d
# ╠═2a8a410d-7c1c-4dfb-9206-dd9efbed2a9e
# ╠═81e3b595-8025-40bb-8598-9a2ee9b65ed6
# ╠═ed6e1952-681a-4303-ba24-5d408fb9bedb
# ╠═070d7fc2-30d2-4150-9dd4-acc99dbfc3ea
# ╠═f4107cb8-68c0-49de-a2bb-3a670314837d
# ╠═580df68f-0165-4be7-84fa-23258f63d741
