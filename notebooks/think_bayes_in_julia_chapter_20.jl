### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ ef34dcc9-bef6-46b0-a3ab-24b502c1ca09
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ 75a6c6c2-0917-11ed-3657-67a42568b31d
using Plots, Distributions, DataFrames, Random, Interpolations, PlutoUI

# ╔═╡ 02888228-3da5-4163-91c5-92085f6e5464
TableOfContents()

# ╔═╡ f98f6341-4595-4556-afc6-95aeb7b5b705
md"# The Kidney Tumor Problem"

# ╔═╡ 12801cc6-c07e-4372-bcb8-270d5a57acab
md"## A simple growth model"

# ╔═╡ 754db0fd-2991-4c3d-a637-50bd5530691b
function calc_volume(diameter)
	factor = 4 * π / 3
	factor * (diameter / 2.0) ^3
end

# ╔═╡ b2cbf003-dd5f-4caf-90d9-f35d261b88cb
begin
	d1 = 1
	v1 = calc_volume(d1)
end

# ╔═╡ 901ac56f-2629-44a0-b2b3-b5e97cdbf6d2
begin
	median_doubling_time = 811
	rdt = 365 / median_doubling_time
	
end

# ╔═╡ 3a58be95-5d60-407c-bff5-8ab9064be398
begin
	interval = 9.0
	doublings= interval * rdt
end

# ╔═╡ 1b8a5c02-6955-46a7-b461-1cf783c35131
v2 = v1 * 2 ^ doublings

# ╔═╡ 4aff0728-9e69-4572-b34d-f5218afee49e
function calc_diameter(volume)
	factor = 3 / π / 4
	2 * (factor * volume) ^ (1/3)
end

# ╔═╡ b42942d8-ba01-4016-89a1-fd22f6371b28
d2 = calc_diameter(v2)

# ╔═╡ 08333b64-7300-4065-ac8b-c5ab0d03be99
md"## A more general model"

# ╔═╡ 5630198f-1198-4978-a224-e6e1fcaf1e24
begin
	counts = [2, 29, 11, 6, 3, 1, 1]
	rdts1 = -1:6 .+ 0.01
	pmf_rdt = pmf_from_seq(rdts1, counts=counts)
end

# ╔═╡ 3c1208b6-2bff-4496-b429-f05c2d112599
rdt_sample = [5.089,  3.572,  3.242,  2.642,  1.982,  1.847,  1.908,  1.798,
        1.798,  1.761,  2.703, -0.416,  0.024,  0.869,  0.746,  0.257,
        0.269,  0.086,  0.086,  1.321,  1.052,  1.076,  0.758,  0.587,
        0.367,  0.416,  0.073,  0.538,  0.281,  0.122, -0.869, -1.431,
        0.012,  0.037, -0.135,  0.122,  0.208,  0.245,  0.404,  0.648,
        0.673,  0.673,  0.563,  0.391,  0.049,  0.538,  0.514,  0.404,
        0.404,  0.33,  -0.061,  0.538,  0.306]



# ╔═╡ 88e40789-48a1-4239-aced-6b411059930a
pmf_rdt1 = kde_from_sample(rdt_sample, -2, 6, 201)

# ╔═╡ 400c14e3-a450-4b8a-bd7e-1d56ed31358e
1 / cdf(make_cdf(pmf_rdt1), 0.5) * 365

# ╔═╡ 821be72f-ea72-4c1d-8147-eb4fe993520d
plot(pmf_rdt1)

# ╔═╡ 563b2cc2-76e1-46ca-9778-242149fbb772
md"## Simulation"

# ╔═╡ 4a35debb-737c-4ff4-81c1-90e57b228d4c
begin
	interval1 = 245 / 365
	min_diameter = 0.3
	max_diameter = 20
end

# ╔═╡ f32e8919-8910-4c9a-9fc3-158d4f2339d6
begin
	v0 = calc_volume(min_diameter)
	vmax = calc_volume(max_diameter)
	v0, vmax
end

# ╔═╡ edb8e3bc-c8d9-482e-a3bb-df16d0b3ea58
function simulate_growth(pmf_rdt)
	age = 0
	volume = v0
	res = []
	push!(res, (age=age, volume=volume))
	while volume < vmax
		rdt = rand(pmf_rdt1)
		age += interval1
		doublings = rdt * interval1
		volume *= 2^doublings
		push!(res, (age=age, volume=volume))
	end
	sim = DataFrame(res)
	sim.diameter = calc_diameter.(sim.volume)
	sim
end

# ╔═╡ 1b01cec5-f7ca-49d3-a5db-7c83db9dce7c
Random.seed!(17)

# ╔═╡ 7a24b1d3-5c8d-46b9-bd82-7c5411dce6c4
sim = simulate_growth(pmf_rdt1)

# ╔═╡ a5730dad-050c-40c8-85f3-dee90087a47c
first(sim, 3)

# ╔═╡ 8c7908ff-5a97-4bb4-9936-0921e2c28223
last(sim, 3)

# ╔═╡ 7f1682b9-5f70-4277-a094-c271374402dc
sims = [simulate_growth(pmf_rdt1) for _ in 1:101]

# ╔═╡ dafe5dfb-66ac-4268-b612-709529719016
begin
	diameters = [4, 8, 16]
	plot(legend=false)
	hline!(diameters, linestyle=:dashdot)
	for sim in sims
		plot!(sim.age, sim.diameter, yscale=:log10, yaxis=([0.2, 0.5, 1, 2, 5, 10, 20]))
	end
	plot!(yformatter=x->string(round(x)))
end

# ╔═╡ c79f4012-bcf5-4b9b-b267-e5b65b1a8949
function interpolate_ages(sims, diameter)
	ages = []
	for (i, sim) in enumerate(sims)
		interp = LinearInterpolation(Interpolations.deduplicate_knots!(sim.diameter), sim.age)
		age = interp(diameter)
		#println(age)
		push!(ages, age)
	end
	ages
end

# ╔═╡ 32016ec8-669a-48bc-aeeb-c7980a0ff37a
# ╠═╡ show_logs = false
begin
	ages = interpolate_ages(sims, 15)
	cdf1 = cdf_from_seq(ages)
	prob_le(cdf1, 9), credible_interval(cdf1, 0.9)
end

# ╔═╡ 2cbc5253-34b2-47b2-8a51-02cf78d509c7
begin
	plot()
	for diameter in diameters
		ages = interpolate_ages(sims, diameter)
		cdf = cdf_from_seq(ages)
		plot!(cdf, label=string(diameter)*" cm")
	end
	plot!()
end

# ╔═╡ a4d471a0-9086-4db0-86ba-31c1e66c9f8d
md"# Approximate Baysian Calculation"

# ╔═╡ 1b3e4e95-1940-40c4-b97e-7e3990d0fe74
md"## Counting Cells"

# ╔═╡ 0be3b768-5b83-4944-9562-7911accdbd87
md"waiting until I figure out a pymc3 equivalent for julia"

# ╔═╡ 3f96ce00-4a15-4ad0-9e26-08a02712df12
md"# Exercises"

# ╔═╡ 77358e2d-b49d-4e09-a061-9eddaedee58e
md"## Socks"

# ╔═╡ 4018a0a1-f2c9-4ad8-9477-5e3140317855
begin
	mu = 30
	p = 0.866666
	r = mu * (1-p) / p
	prior_n_socks = NegativeBinomial(r, 1-p)
	mean(prior_n_socks), std(prior_n_socks)
end

# ╔═╡ a7c2b70e-ed65-429a-90d8-2b160b871d07
begin
	prior_prop_pair = Beta(15, 2)
	mean(prior_prop_pair)
end

# ╔═╡ 704f0521-8b8a-4ebb-bac9-9e2dc4dbb321
pmf1 = pmf_from_seq(1:90, normalize([pdf(prior_n_socks, x) for x in 1:90]));

# ╔═╡ 4d667831-54ce-4d7d-b942-955bbcfc3f98
plot(pmf1, yaxis="PMF", xaxis=("Number of socks"), label="prior")

# ╔═╡ f973a165-79a5-4b73-9857-0dc46e5fd186
pmf2 = pmf_from_dist(range(0, 1, 101), prior_prop_pair);

# ╔═╡ a7350a87-b3ac-49e6-8a8b-16f472bcd7a2
plot(pmf2, yaxis="PDF", xaxis="Proportion of socks in pairs", label="prior")

# ╔═╡ dde3923a-a343-48e7-aacf-7d0f0c6cd3fe
begin
	n_socks = rand(prior_n_socks, 1000)
	prop_pairs = rand(prior_prop_pair, 1000)
end

# ╔═╡ dc80d3d7-c968-444a-98d8-1a7e2c8f926f
begin
	n_pairs = round.(n_socks ./ 2 .* prop_pairs)
	n_odds = n_socks .- n_pairs .* 2
	n_pairs, n_odds
end

# ╔═╡ e63332bf-5704-46dd-ac46-7ce65feb20ce
mean(n_pairs),std(n_pairs), mean(n_odds), std(n_odds)

# ╔═╡ 311846a3-56d6-4689-92ca-745c698b690e
begin
	n_pairs1 = 9
	n_odds1 = 5
	socks = vcat(1:n_pairs1, 1:n_odds1+n_pairs1)
end

# ╔═╡ 8bbc0d2c-39fb-4979-ad25-e4ed929cc289
begin
	shuffle!(socks)
	picked_socks = socks[1:11]
end

# ╔═╡ 8ea94030-5c69-48e3-b0e9-388c20d182b5
function find_counts(v)
	u = unique(v)
	c = [length(findall(y -> y == x, v)) for x in u]
	u,c
end

# ╔═╡ f85b85b6-2f7a-4e08-b613-4bd100a2a4e7
values, cts = find_counts(picked_socks)

# ╔═╡ b8c5d74d-f385-4cca-9fd5-4fab3acb62e9
begin
	solo = sum(cts .== 1)
	pairs = sum(cts .== 2)
	solo, pairs
end

# ╔═╡ 944fe613-e3b7-41d3-8aec-11fe0c439566
function pick_socks(n_pairs, n_odds, n_pick)
	socks = vcat(1:n_pairs1, 1:n_odds1+n_pairs1)
	shuffle!(socks)
	picked_socks = socks[1:n_pick]
	values, cts = find_counts(picked_socks)
	pairs = sum(cts .== 2)
	odds = sum(cts .== 1)
	pairs, odds
end

# ╔═╡ a513fa1a-a4e0-438e-854f-bb79c55372b4
pick_socks(n_pairs, n_odds, 11)

# ╔═╡ d7b18331-7582-431c-9ca0-793605c5ea1b
begin
	data = (0, 11)
	res = []
	for i in 1:10000
		n_socks = rand(prior_n_socks)
		if n_socks > 10
			prop_pairs = rand(prior_prop_pair)
			n_pairs = round(n_socks / 2 * prop_pairs)
			n_odds = n_socks - n_pairs * 2
			result = pick_socks(n_pairs, n_odds, 11)
			if result == data
				push!(res, (n_socks=n_socks, n_pairs=n_pairs, n_odds=n_odds))
			end
		end
	end
	length(res)
			
end

# ╔═╡ f89edfb0-6238-45f8-afa0-521b031ba638
begin
	results = DataFrame(res)
	first(results, 6)
end

# ╔═╡ 49e99fba-4c4a-44f8-b186-63ea08fba827
posterior_n_socks = pmf_from_seq(results.n_socks);

# ╔═╡ 5f9bb2ea-6268-4844-8b5c-7078bc2093d9
mean(posterior_n_socks), credible_interval(posterior_n_socks, 0.9)

# ╔═╡ 2e291ed6-69ac-4fbf-9d1f-bb15bf5cf878
plot(posterior_n_socks, yaxis="PMF", xaxis="Number of socks", label="posterior")

# ╔═╡ Cell order:
# ╠═ef34dcc9-bef6-46b0-a3ab-24b502c1ca09
# ╠═75a6c6c2-0917-11ed-3657-67a42568b31d
# ╠═02888228-3da5-4163-91c5-92085f6e5464
# ╟─f98f6341-4595-4556-afc6-95aeb7b5b705
# ╟─12801cc6-c07e-4372-bcb8-270d5a57acab
# ╠═754db0fd-2991-4c3d-a637-50bd5530691b
# ╠═b2cbf003-dd5f-4caf-90d9-f35d261b88cb
# ╠═901ac56f-2629-44a0-b2b3-b5e97cdbf6d2
# ╠═3a58be95-5d60-407c-bff5-8ab9064be398
# ╠═1b8a5c02-6955-46a7-b461-1cf783c35131
# ╠═4aff0728-9e69-4572-b34d-f5218afee49e
# ╠═b42942d8-ba01-4016-89a1-fd22f6371b28
# ╟─08333b64-7300-4065-ac8b-c5ab0d03be99
# ╠═5630198f-1198-4978-a224-e6e1fcaf1e24
# ╠═3c1208b6-2bff-4496-b429-f05c2d112599
# ╠═88e40789-48a1-4239-aced-6b411059930a
# ╠═400c14e3-a450-4b8a-bd7e-1d56ed31358e
# ╠═821be72f-ea72-4c1d-8147-eb4fe993520d
# ╟─563b2cc2-76e1-46ca-9778-242149fbb772
# ╠═4a35debb-737c-4ff4-81c1-90e57b228d4c
# ╠═f32e8919-8910-4c9a-9fc3-158d4f2339d6
# ╠═edb8e3bc-c8d9-482e-a3bb-df16d0b3ea58
# ╠═1b01cec5-f7ca-49d3-a5db-7c83db9dce7c
# ╠═7a24b1d3-5c8d-46b9-bd82-7c5411dce6c4
# ╠═a5730dad-050c-40c8-85f3-dee90087a47c
# ╠═8c7908ff-5a97-4bb4-9936-0921e2c28223
# ╠═7f1682b9-5f70-4277-a094-c271374402dc
# ╠═dafe5dfb-66ac-4268-b612-709529719016
# ╠═c79f4012-bcf5-4b9b-b267-e5b65b1a8949
# ╠═32016ec8-669a-48bc-aeeb-c7980a0ff37a
# ╠═2cbc5253-34b2-47b2-8a51-02cf78d509c7
# ╟─a4d471a0-9086-4db0-86ba-31c1e66c9f8d
# ╟─1b3e4e95-1940-40c4-b97e-7e3990d0fe74
# ╟─0be3b768-5b83-4944-9562-7911accdbd87
# ╟─3f96ce00-4a15-4ad0-9e26-08a02712df12
# ╟─77358e2d-b49d-4e09-a061-9eddaedee58e
# ╠═4018a0a1-f2c9-4ad8-9477-5e3140317855
# ╠═a7c2b70e-ed65-429a-90d8-2b160b871d07
# ╠═704f0521-8b8a-4ebb-bac9-9e2dc4dbb321
# ╠═4d667831-54ce-4d7d-b942-955bbcfc3f98
# ╠═f973a165-79a5-4b73-9857-0dc46e5fd186
# ╠═a7350a87-b3ac-49e6-8a8b-16f472bcd7a2
# ╠═dde3923a-a343-48e7-aacf-7d0f0c6cd3fe
# ╠═dc80d3d7-c968-444a-98d8-1a7e2c8f926f
# ╠═e63332bf-5704-46dd-ac46-7ce65feb20ce
# ╠═311846a3-56d6-4689-92ca-745c698b690e
# ╠═8bbc0d2c-39fb-4979-ad25-e4ed929cc289
# ╠═8ea94030-5c69-48e3-b0e9-388c20d182b5
# ╠═f85b85b6-2f7a-4e08-b613-4bd100a2a4e7
# ╠═b8c5d74d-f385-4cca-9fd5-4fab3acb62e9
# ╠═944fe613-e3b7-41d3-8aec-11fe0c439566
# ╠═a513fa1a-a4e0-438e-854f-bb79c55372b4
# ╠═d7b18331-7582-431c-9ca0-793605c5ea1b
# ╠═f89edfb0-6238-45f8-afa0-521b031ba638
# ╠═49e99fba-4c4a-44f8-b186-63ea08fba827
# ╠═5f9bb2ea-6268-4844-8b5c-7078bc2093d9
# ╠═2e291ed6-69ac-4fbf-9d1f-bb15bf5cf878
