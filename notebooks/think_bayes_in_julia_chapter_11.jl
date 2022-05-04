### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 678a4dce-6f32-457a-b9c7-30c327991d8a
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
	#Pkg.add("Colors")
	#Pkg.add("ImageTransformations")
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

# ╔═╡ 1a8c56ac-1ee3-48a2-b8a8-f9a2424d3993
joint = make_joint(*,prior, prior)

# ╔═╡ c0368bcc-7062-4a88-91fe-39949f5c9868
md"## Visualizing the Joint Distribution"

# ╔═╡ 31b02921-ac32-4333-8352-424164c05265
visualize_joint(joint)

# ╔═╡ 04593a8c-c378-4f99-905b-dc61a4d4bcbc
md"## Likelihood"

# ╔═╡ 44d43d0b-860b-42fa-8007-11f692b8222a
A_taller = outer(>, values(prior), values(prior))

# ╔═╡ 14f3f366-2741-446c-99f5-d3506c2cd234
likelihood = DataFrame(A_taller, string.(values(prior)))

# ╔═╡ 9e81cf89-ab9c-4462-9e24-8546ea35bf03
visualize_joint(likelihood)

# ╔═╡ 590b6403-aa96-477f-8747-6056fe237772
posterior = DataFrame(df_to_matrix(joint) .* df_to_matrix(likelihood), string.(values(prior)))

# ╔═╡ 9a445bc3-2c17-412b-a4dd-1c281e7dd722
visualize_joint(posterior)

# ╔═╡ 49db0e62-af5d-40b1-b479-594354c0a211
md"## Marginal Distributions"

# ╔═╡ 76ecda71-3943-4834-a9a1-52387da1a139


# ╔═╡ Cell order:
# ╟─1b872baa-ca5b-11ec-31aa-b359cc37aa04
# ╠═678a4dce-6f32-457a-b9c7-30c327991d8a
# ╠═6670ce99-8c18-4094-93a3-14afdb164e23
# ╠═ca757f1b-cff6-46b1-98b5-bd6a25042cf0
# ╠═abb7c6ab-8d68-43bf-8997-8a50b720514a
# ╠═b1e31c89-0163-4adf-b358-398d999ddd81
# ╠═72420f5e-111f-4a17-b05b-e73cf1c23706
# ╟─c2972026-51d1-4a44-a0f4-7b5c95fbae7d
# ╠═dcdfeb38-37dc-4433-b79c-ea2bfa76bbda
# ╠═60053e03-277e-458f-9276-b16e03df1728
# ╠═4451893d-e863-46b6-8a1c-c8363f3fe496
# ╠═0fbc2535-2e0e-4550-8217-e96fd0ae3310
# ╠═417cd3a1-2cca-41f5-8f52-5717b568fd67
# ╟─9e982496-4c0d-42ef-9dd2-5b6e306f70f5
# ╠═1a8c56ac-1ee3-48a2-b8a8-f9a2424d3993
# ╟─c0368bcc-7062-4a88-91fe-39949f5c9868
# ╠═31b02921-ac32-4333-8352-424164c05265
# ╟─04593a8c-c378-4f99-905b-dc61a4d4bcbc
# ╠═44d43d0b-860b-42fa-8007-11f692b8222a
# ╠═14f3f366-2741-446c-99f5-d3506c2cd234
# ╠═9e81cf89-ab9c-4462-9e24-8546ea35bf03
# ╠═590b6403-aa96-477f-8747-6056fe237772
# ╠═9a445bc3-2c17-412b-a4dd-1c281e7dd722
# ╟─49db0e62-af5d-40b1-b479-594354c0a211
# ╠═76ecda71-3943-4834-a9a1-52387da1a139
