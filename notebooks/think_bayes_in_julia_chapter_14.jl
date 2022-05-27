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
prior_bulb = make_joint(*, prior_lb_lam, prior_lb_k)

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
end

# ╔═╡ 1afe1b94-6a2a-487b-bf2e-bbd5dfb3a3a9
prod_lb = posterior_bulb.M .* means

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

# ╔═╡ f3d44c45-c421-4826-aaf0-3b79390c76c0
begin
	function make_pmf_seq(ps, t, n)
		λ, k = ps
		prob_dead = cdf(Weibull(k, λ), t)
		make_binomial(n, prob_dead)
	end
	pmf_seq = [make_pmf_seq(x, 1000, 100) for x in ps_index]
end

# ╔═╡ ccb060f3-b6bf-4d03-8db5-02ab7fbe0cbb
post_pred = make_mixture(pmf_from_seq(ps_index, ps_vals), pmf_seq);

# ╔═╡ c2cc93b6-a1da-47e7-81b2-c89465c5beb1
begin
	plot(dist_num_dead, label="known parameters")
	plot!(pmf_from_seq(1:101, probs(post_pred)), label="unknown parameters")
	plot!(plot_title="Posterior predictive distribution", xaxis=("Number of dead bulbs"), yaxis=("PMF"))
end

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
# ╠═f3d44c45-c421-4826-aaf0-3b79390c76c0
# ╠═ccb060f3-b6bf-4d03-8db5-02ab7fbe0cbb
# ╠═c2cc93b6-a1da-47e7-81b2-c89465c5beb1
