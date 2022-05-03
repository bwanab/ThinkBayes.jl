### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 678a4dce-6f32-457a-b9c7-30c327991d8a
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 6670ce99-8c18-4094-93a3-14afdb164e23
using DataFrames, StatsPlots

# ╔═╡ 1b872baa-ca5b-11ec-31aa-b359cc37aa04
md"## Chapter 11 - Comparison"

# ╔═╡ ca757f1b-cff6-46b1-98b5-bd6a25042cf0
x = [1, 3, 5]

# ╔═╡ abb7c6ab-8d68-43bf-8997-8a50b720514a
y = [2, 4]

# ╔═╡ b1e31c89-0163-4adf-b358-398d999ddd81
d = x' .* y

# ╔═╡ 72420f5e-111f-4a17-b05b-e73cf1c23706
size(d)

# ╔═╡ 2f4e3509-63ed-411c-bae6-158351e54fc1
function outer(f, x, y)
	d = f.(x', y)
	[vec(d[i,:]) for i in 1:length(y)]
end

# ╔═╡ c2972026-51d1-4a44-a0f4-7b5c95fbae7d
md"## How Tall is A?"

# ╔═╡ dcdfeb38-37dc-4433-b79c-ea2bfa76bbda
mean_height = 178.0

# ╔═╡ 60053e03-277e-458f-9276-b16e03df1728
qs = mean_height-24:0.5:mean_height+23.5

# ╔═╡ 4451893d-e863-46b6-8a1c-c8363f3fe496
std_height=7.7

# ╔═╡ 0fbc2535-2e0e-4550-8217-e96fd0ae3310
prior = make_normal_pmf(qs, mu=mean_height, sigma=std_height);

# ╔═╡ 417cd3a1-2cca-41f5-8f52-5717b568fd67
plot(prior)

# ╔═╡ 9e982496-4c0d-42ef-9dd2-5b6e306f70f5
md"## Joint Distribution"

# ╔═╡ 1a2e51c8-2131-4ab0-83f5-01166c2eb806
function make_joint(f, x_pmf, y_pmf)
	x_p = probs(x_pmf)
	y_p = probs(y_pmf)
	v = outer(f, x_p, y_p)
	df = DataFrame()
	for i in 1:length(values(y_pmf))
		df[!, string(values(y_pmf)[i])] = v[i]
	end
	df
end

# ╔═╡ 1a8c56ac-1ee3-48a2-b8a8-f9a2424d3993
joint = make_joint(*,prior, prior)

# ╔═╡ 5898dd53-701d-485f-980f-bc1cf3b5970e
sum(sum(eachcol(joint)))

# ╔═╡ c0368bcc-7062-4a88-91fe-39949f5c9868
md"## Visualizing the Joint Distribution"

# ╔═╡ b31faa4c-ec12-47f0-b271-4452285f2031


# ╔═╡ Cell order:
# ╟─1b872baa-ca5b-11ec-31aa-b359cc37aa04
# ╠═678a4dce-6f32-457a-b9c7-30c327991d8a
# ╠═6670ce99-8c18-4094-93a3-14afdb164e23
# ╠═ca757f1b-cff6-46b1-98b5-bd6a25042cf0
# ╠═abb7c6ab-8d68-43bf-8997-8a50b720514a
# ╠═b1e31c89-0163-4adf-b358-398d999ddd81
# ╠═72420f5e-111f-4a17-b05b-e73cf1c23706
# ╠═2f4e3509-63ed-411c-bae6-158351e54fc1
# ╟─c2972026-51d1-4a44-a0f4-7b5c95fbae7d
# ╠═dcdfeb38-37dc-4433-b79c-ea2bfa76bbda
# ╠═60053e03-277e-458f-9276-b16e03df1728
# ╠═4451893d-e863-46b6-8a1c-c8363f3fe496
# ╠═0fbc2535-2e0e-4550-8217-e96fd0ae3310
# ╠═417cd3a1-2cca-41f5-8f52-5717b568fd67
# ╟─9e982496-4c0d-42ef-9dd2-5b6e306f70f5
# ╠═1a2e51c8-2131-4ab0-83f5-01166c2eb806
# ╠═1a8c56ac-1ee3-48a2-b8a8-f9a2424d3993
# ╠═5898dd53-701d-485f-980f-bc1cf3b5970e
# ╟─c0368bcc-7062-4a88-91fe-39949f5c9868
# ╠═b31faa4c-ec12-47f0-b271-4452285f2031
