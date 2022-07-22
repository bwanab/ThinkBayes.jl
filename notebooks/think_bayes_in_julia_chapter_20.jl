### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ ef34dcc9-bef6-46b0-a3ab-24b502c1ca09
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ 75a6c6c2-0917-11ed-3657-67a42568b31d
using Plots, Distributions, DataFrames, Random, Interpolations

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
	plot(legend=false, yaxis=([0.2, 0.5, 1, 2, 5, 10, 20]))
	for sim in sims
		plot!(sim.age, sim.diameter, yscale=:log10)
	end
	vline!([16])
	hline!([20])
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


# ╔═╡ Cell order:
# ╠═ef34dcc9-bef6-46b0-a3ab-24b502c1ca09
# ╠═75a6c6c2-0917-11ed-3657-67a42568b31d
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
