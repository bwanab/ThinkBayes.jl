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

# ╔═╡ 6b53e7e7-afc9-4ffa-994a-98392d1297c7
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
visualize_joint(joint, normalize=true)

# ╔═╡ 04593a8c-c378-4f99-905b-dc61a4d4bcbc
md"## Likelihood"

# ╔═╡ 44d43d0b-860b-42fa-8007-11f692b8222a
A_taller = outer(>, values(prior), values(prior))

# ╔═╡ 14f3f366-2741-446c-99f5-d3506c2cd234
likelihood = Joint(A_taller, values(prior), values(prior))

# ╔═╡ 9e81cf89-ab9c-4462-9e24-8546ea35bf03
visualize_joint(likelihood)

# ╔═╡ 590b6403-aa96-477f-8747-6056fe237772
posterior = Joint(normalize(joint.M .* likelihood.M), values(prior), values(prior))

# ╔═╡ 9a445bc3-2c17-412b-a4dd-1c281e7dd722
visualize_joint(posterior, normalize=true)

# ╔═╡ 49db0e62-af5d-40b1-b479-594354c0a211
md"## Marginal Distributions"

# ╔═╡ b1ee65ac-f132-4d5a-94b1-bcba548b650a
col = column(posterior, 180.0)

# ╔═╡ 8663c52f-e0ab-42c6-b20c-e0f88697d468
sum(col)

# ╔═╡ 9421f9d3-33cd-4b77-9084-515af9238ae6
column_sums = [(c, sum(column(posterior, c))) for c in posterior.xs]

# ╔═╡ e265b1d2-33bf-431a-827b-0ebcb384301e
marginal_A = pmf_from_tuples(column_sums);

# ╔═╡ b7ee6f3c-be26-464b-b484-338143f0628d
plot(marginal_A)

# ╔═╡ 09143c94-efec-467e-83f0-580d35035400
row_sums = [(r, sum(row(posterior, r))) for r in posterior.ys]

# ╔═╡ 65090587-9061-448e-ab4b-495f5d261cb1
marginal_B = pmf_from_tuples(row_sums)

# ╔═╡ e2e8f155-3465-481d-bb48-37dca0c12d77
plot(marginal_B)

# ╔═╡ 49448fcf-49e1-44d0-86d1-a60eb0330770
function marginal(joint, dim)
	sums = sum(joint.M, dims=dim)
	pmf_from_seq(dim==1 ? joint.xs : joint.ys, vec(sums))
end

# ╔═╡ e8a44221-f3e2-4df9-bb0b-c73e6f6066b8
marg_A = marginal(posterior, 1);

# ╔═╡ 11a135b5-4275-4286-804d-5c895d95799b
marg_B = marginal(posterior, 2);

# ╔═╡ bb54c318-284f-4547-8af3-f0e37ad88fb6
begin
	plot(prior, label="prior")
	plot!(marg_A, label="posterior A")
	plot!(marg_B, label="posterior B")
end

# ╔═╡ b01831ef-517d-4c4a-9acd-d1d6273e091a
mean(prior), mean(marg_A), mean(marg_B)

# ╔═╡ 8ad5a1ad-b62e-4982-aee1-78b5a24982f4
std(prior), std(marg_A), std(marg_B)

# ╔═╡ b8d21789-4f2a-4891-8263-1036a929c5dd
md"## Conditional Posteriors"

# ╔═╡ cbab5bb8-b9d7-414e-ad72-381b2afe65d4
column_170 = column(posterior, 170.0)

# ╔═╡ 7e1fdc4b-e1e2-4036-8d2d-2fed440dedc7
cond_B = pmf_from_seq(values(marg_B), normalize(column_170));

# ╔═╡ d59d33f6-adc7-414b-b11c-f3edbe8441cd
begin
	plot(prior, label="prior")
	plot!(marg_B, label="posterior B")
	plot!(cond_B, label="conditional posterior B")
end

# ╔═╡ abb26a6f-4347-4552-a770-687335d0ac40
md"""## Exercises
_exercise 11.1_"""

# ╔═╡ 7b2003f0-ddd3-4af2-9fe8-008c53e16d2f
begin
	# find the index of 180 in the vaules of prior, then get that row from posterior.
	vals = row(posterior, 180.0)
	cond_A = pmf_from_seq(values(marg_A), normalize(vals));
end;

# ╔═╡ db018953-b2d2-4c02-8b1a-80d37a57a0ae
begin
	plot(marg_A, label="posterior A")
	plot!(cond_A, label="conditional posterior A")
end

# ╔═╡ a65bca47-5479-4aff-bf46-fbc64522d05d
md"_exercise 11.2_"

# ╔═╡ 630aadaa-07b1-4562-b30b-e67cdb72443c
md"_exercise 11.3_"

# ╔═╡ 5e06a69f-b541-402c-bbe3-4c752764e5e6
prior_A = make_normal_pmf(1300:10:1890, mu=1600.0, sigma=100.0)

# ╔═╡ 53a5f782-af57-4fc4-8497-a66dc45284f6
prior_B = make_normal_pmf(1500:10:2090, mu=1800.0, sigma=100.0)

# ╔═╡ 89bb5b0c-00ad-433c-b028-f0bf29621d20
begin
	plot(prior_A, label="prior A")
	plot!(prior_B, label="prior B")
end

# ╔═╡ 73a2d469-904a-4284-af79-4957155d11ef
joint_elo = make_joint(*, prior_A, prior_B)

# ╔═╡ 2cf76a80-15aa-40c7-a370-694c76298fd9
size(joint_elo.M)

# ╔═╡ 42b9cb1b-3f2c-4c1c-871d-d8ad5a123488
visualize_joint(joint_elo, xaxis="A Rating", yaxis="B Rating")

# ╔═╡ 02ebb04c-1ab8-4a17-beff-fff62fe6deb1
begin
	X = values(prior_A)
	Y = values(prior_B)
	diffs = outer(-, X, Y)
	diff = reduce(hcat, diffs)
end;

# ╔═╡ 11f09c1d-4129-42e1-9c37-6892eff6b64b


# ╔═╡ f038c99c-b31f-41f3-ae7a-ec6621bb30bc
a = 1 ./ (1 .+ 10 .^(-diffs ./ 400));

# ╔═╡ 3d883128-bdde-4f47-9c63-f84241f567e1
visualize_joint(a, xs=X, ys=Y)

# ╔═╡ 185588a5-ec77-4f6d-8793-4e4d87e360a6
posterior_elo = Joint(normalize(joint_elo.M .* a), X, Y);

# ╔═╡ 2af4f254-12ac-48ae-85c6-05c5418d2ea1
visualize_joint(posterior_elo)

# ╔═╡ 7e08acef-7c36-4bd7-88da-3f0cb4a158ad
marginal_A_elo = marginal(posterior_elo, 1);

# ╔═╡ c98053a9-090c-417b-a02b-f0fba219af22
marginal_B_elo = marginal(posterior_elo, 2);

# ╔═╡ 36467d5a-6b00-4a9e-8511-5b06e568eba2
begin
	plot(marginal_A_elo, label="posterior A")
	plot!(marginal_B_elo, label="posterior B")
end

# ╔═╡ 0f95cfb4-c178-4cdd-99be-017200089be8
mean(marginal_A_elo), mean(marginal_B_elo)

# ╔═╡ 937f913d-dec4-4607-989a-de5fc88bae8e
std(marginal_A_elo), std(marginal_B_elo)

# ╔═╡ Cell order:
# ╟─1b872baa-ca5b-11ec-31aa-b359cc37aa04
# ╟─6b53e7e7-afc9-4ffa-994a-98392d1297c7
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
# ╠═b1ee65ac-f132-4d5a-94b1-bcba548b650a
# ╠═8663c52f-e0ab-42c6-b20c-e0f88697d468
# ╠═9421f9d3-33cd-4b77-9084-515af9238ae6
# ╠═e265b1d2-33bf-431a-827b-0ebcb384301e
# ╠═b7ee6f3c-be26-464b-b484-338143f0628d
# ╠═09143c94-efec-467e-83f0-580d35035400
# ╠═65090587-9061-448e-ab4b-495f5d261cb1
# ╠═e2e8f155-3465-481d-bb48-37dca0c12d77
# ╠═49448fcf-49e1-44d0-86d1-a60eb0330770
# ╠═e8a44221-f3e2-4df9-bb0b-c73e6f6066b8
# ╠═11a135b5-4275-4286-804d-5c895d95799b
# ╠═bb54c318-284f-4547-8af3-f0e37ad88fb6
# ╠═b01831ef-517d-4c4a-9acd-d1d6273e091a
# ╠═8ad5a1ad-b62e-4982-aee1-78b5a24982f4
# ╟─b8d21789-4f2a-4891-8263-1036a929c5dd
# ╠═cbab5bb8-b9d7-414e-ad72-381b2afe65d4
# ╠═7e1fdc4b-e1e2-4036-8d2d-2fed440dedc7
# ╠═d59d33f6-adc7-414b-b11c-f3edbe8441cd
# ╟─abb26a6f-4347-4552-a770-687335d0ac40
# ╠═7b2003f0-ddd3-4af2-9fe8-008c53e16d2f
# ╠═db018953-b2d2-4c02-8b1a-80d37a57a0ae
# ╟─a65bca47-5479-4aff-bf46-fbc64522d05d
# ╟─630aadaa-07b1-4562-b30b-e67cdb72443c
# ╠═5e06a69f-b541-402c-bbe3-4c752764e5e6
# ╠═53a5f782-af57-4fc4-8497-a66dc45284f6
# ╠═89bb5b0c-00ad-433c-b028-f0bf29621d20
# ╠═73a2d469-904a-4284-af79-4957155d11ef
# ╠═2cf76a80-15aa-40c7-a370-694c76298fd9
# ╠═42b9cb1b-3f2c-4c1c-871d-d8ad5a123488
# ╠═02ebb04c-1ab8-4a17-beff-fff62fe6deb1
# ╠═11f09c1d-4129-42e1-9c37-6892eff6b64b
# ╠═f038c99c-b31f-41f3-ae7a-ec6621bb30bc
# ╠═3d883128-bdde-4f47-9c63-f84241f567e1
# ╠═185588a5-ec77-4f6d-8793-4e4d87e360a6
# ╠═2af4f254-12ac-48ae-85c6-05c5418d2ea1
# ╠═7e08acef-7c36-4bd7-88da-3f0cb4a158ad
# ╠═c98053a9-090c-417b-a02b-f0fba219af22
# ╠═36467d5a-6b00-4a9e-8511-5b06e568eba2
# ╠═0f95cfb4-c178-4cdd-99be-017200089be8
# ╠═937f913d-dec4-4607-989a-de5fc88bae8e
