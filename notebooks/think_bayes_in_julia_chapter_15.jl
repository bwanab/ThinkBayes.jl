### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 6b6640d0-e1de-11ec-3baf-01ded65b80ff
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 20a227a8-dece-403f-af26-92225880e2b8
using DataFrames, CSV, Plots, Distributions, PlutoUI

# ╔═╡ f1325978-b2d0-4ec8-ad45-7761510280f4
TableOfContents()

# ╔═╡ f749819f-6904-4a1e-a0cf-6c301d710118
md"## The Grizzly Bear Problem"

# ╔═╡ 018690c4-4737-4387-9e28-9c18a3527e66
hypergeom(N, K, n) = Hypergeometric(K, N-K, n)

# ╔═╡ ce5f433a-3cd1-4b3b-84a9-fad8635c159f
begin
	N = 100
	K = 23
	n = 19
	
	ks = 0:11
	h = hypergeom(N, K, n)
	ps = [pdf(h, x) for x in ks]
end

# ╔═╡ fc125273-cad0-4be6-8c3f-67755ef46dc3
begin
	bar(ks, ps)
	plot!(xaxis=("Number of bears observed twice"), yaxis=("PMF"), title="Hypergeometric distribution of k (known population of 100)", size=(700, 500))
end

# ╔═╡ 05b15a27-5540-4903-b8e8-ff291410211e
md"## The Update"

# ╔═╡ b4a4991e-9431-4f4b-940b-a4dc573b8703
prior_N = pmf_from_seq(50:500);

# ╔═╡ 63ab927a-0963-42f5-84a5-0feaa255a0f9
begin
	k = 4
	likelihood = [pdf(hypergeom(N, K, n), k) for N in values(prior_N)]
end

# ╔═╡ 619634a0-acdf-4bc3-b7e2-ba4551053631
posterior_N = prior_N * likelihood;

# ╔═╡ eaafa265-b110-4a16-a0b6-d8e43e4c55f0
plot(posterior_N, title="Posterior distribution of N", yaxis=("PDF"), xaxis=("Population of Bears"))

# ╔═╡ 310f4f4b-0832-4bb0-8f98-2ace5bb48c13
max_prob(posterior_N)

# ╔═╡ c906584d-496d-4881-a0f9-777730ff229b
mean(posterior_N)

# ╔═╡ a6f060e7-1d4f-4f61-b742-6acb70363e79
credible_interval(posterior_N, 0.9)

# ╔═╡ 88423616-07e2-4139-9d58-cd46ab25f051
md"## Two Parameter Model"

# ╔═╡ 93763f67-4500-48c4-8660-07b8e514916d
K, n, k

# ╔═╡ e8db554e-0b31-4878-b812-d7c31b0b6574
begin
	k10 = K - k
	k01 = n - k
	k11 = k
end

# ╔═╡ a83afb07-7347-4945-a0f2-9f4279beac53
N

# ╔═╡ cbefa829-c545-4f87-81c9-14d9e1b526c9
begin
	observed = k01 + k10 + k11
	k00 = N - observed
end

# ╔═╡ 5edf395b-4428-44bb-91e8-ca1a76cf9c34
x = [k00, k01, k10, k11]

# ╔═╡ 9b23f4ba-6426-4594-b9d6-93b715235cf8
begin
	p = 0.2
	q = 1 - p
	y = [q*q, q*p, p*q, p*p]
end

# ╔═╡ 8595805b-8654-44c3-875a-e08c5e467a3d
begin
	m = Multinomial(N, y)
	m_likelihood = pdf(m, x)
end

# ╔═╡ 14f9ac87-c8af-4a6b-8657-58606c8db5b8
md"## The Prior"

# ╔═╡ be381749-61d7-4dd9-a88c-95e4fa0426a9
prior_p = pmf_from_seq(range(0, 0.99, 100));

# ╔═╡ cc192b61-bfec-4bad-b2a0-3d5cea362db0
joint_prior = make_joint(*, prior_N, prior_p);

# ╔═╡ 44c6bbe8-630a-4960-8607-0eddfc309f8c
begin
	joint_prior_index, joint_prior_probs = stack(joint_prior)
	joint_pmf = pmf_from_seq(joint_prior_index, joint_prior_probs)
end;

# ╔═╡ 182cdbbe-9b78-45a7-aa2b-0d293e6ca863
joint_prior_index

# ╔═╡ a7cacc0f-dde5-4989-88bc-727d79b19983
md"## The Update"

# ╔═╡ 69a460c0-5812-41ab-a9bf-3f0ddb8a05d7
begin
	function make_joint_likelihood(N, p, data)
		observed = sum(data)
		k00 = N - observed
		x = vcat([k00], data)
		q = 1-p
		y = [q*q, q*p, p*q, p*p]
		m = Multinomial(N, y)
		pdf(m, x)
	end
	kdata = [k01, k10, k11]
	j_likelihood = [make_joint_likelihood(N, p, kdata) for (N, p) in joint_prior_index]
end

# ╔═╡ 6e6e1df1-c607-4a01-9616-a412dd7cacf0
posterior_pmf = joint_pmf * j_likelihood;

# ╔═╡ 806a0b0b-b171-451f-9fef-925a90b663e6
posterior_joint = transpose(unstack(values(posterior_pmf), probs(posterior_pmf)));

# ╔═╡ fb921a3e-61db-4d36-9edd-808477ebdfd6


# ╔═╡ 18a2765a-c354-4f9a-87cb-b9317ed27887
contour(posterior_joint, size=(800, 500))

# ╔═╡ 2317838c-e9cd-4d1b-a833-d86f5e76853e
posterior2_p = marginal(posterior_joint, 1);

# ╔═╡ 101d7bb6-4135-40f4-953a-abb5f8c69bb7
posterior2_N = marginal(posterior_joint, 2);

# ╔═╡ 01dc87f9-6248-4911-9548-38613c78b386
plot(posterior2_p, title="Posterior marginal distribution of p", xaxis=("Probability of observing a bear"), yaxis=("PDF"))

# ╔═╡ 2b1bef2f-7bc3-48ca-b272-b35177686e82
begin
	plot(posterior_N, title="Posterior marginal distribution of N", xaxis=("Population of Bears"), yaxis=("PDF"), label="one parameter model")
	plot!(posterior2_N, label="two parameter model")
end

# ╔═╡ 21855bf2-9321-4933-ab26-02defa105850
mean(posterior_N), credible_interval(posterior_N, 0.9)

# ╔═╡ 52a61815-beaa-475e-ae04-a5d1a9f75ac7
mean(posterior2_N), credible_interval(posterior2_N, 0.9)

# ╔═╡ 33756054-fe68-48e7-9e09-345f9e560fe5
N1 = 138

# ╔═╡ 9854763f-c8fa-4992-84e1-ffd9769d90f2
begin
	mean1 = (23 + 19) / 2
	p1 = mean1/N1
end

# ╔═╡ 0f887cc4-33dc-44f1-97a9-de480eb2b11e
std(Binomial(N1, p1))

# ╔═╡ e97e23e6-7109-4145-9c72-9c0ec50101ed
begin
	N2 = 173
	p2 = mean1 / N2
end

# ╔═╡ 0ae406e7-baf0-4200-8439-4432939e4fe3
std(Binomial(N2, p2))

# ╔═╡ 454a5e06-72c9-4774-8e7f-a2e1a694b049
begin
	plot1 = plot(posterior2_p, label="two parameter model", size=(800, 100))
	plot2 = contour(posterior_joint, size=(800, 500), legend=false)
	plot(plot1, plot2, layout=(2,1), legend=false, size=(800, 700))
end

# ╔═╡ da9310fa-38ba-410a-979f-584c914885ab
md"## The Lincoln Index Problem"

# ╔═╡ 0a00d82a-a41a-4098-a8c7-90796ca5162f
begin
	lk11 = 3   # bugs found in common
	lk10 = 20 - lk11 # first tester finds 20 bugs
	lk01 = 15 - lk11 # 2nd tester finds 15
	lp0, lp1 = 0.2, 0.15
end

# ╔═╡ fcb7010f-4784-4d99-9867-764b44007a6b
function compute_probs(p0, p1)
	q0 = 1-p0
	q1 = 1-p1
	[q0*q1, q0*p1, p0*q1, p0*p1]
end

# ╔═╡ 42202dbd-a39f-4b26-9f89-43244e4a3a02
ly = compute_probs(lp0, lp1)

# ╔═╡ 2baeb3d3-2a07-413a-a068-444336d615fb
begin
	lqs = range(32, 350, step=5)
	lprior_N = pmf_from_seq(lqs)
end;

# ╔═╡ ff284b61-158a-4072-a5d9-a1ffc4b1977b
data = [lk01, lk10, lk11]

# ╔═╡ 7e5b0c5b-a973-4004-a3bb-07692af16092


# ╔═╡ afafe1e6-687b-4b96-b98c-781a26765f3d
begin
	function compute_likelihood(N, y, data)
		x = vcat([N - sum(data)], data)
		pdf(Multinomial(N, y), x)
	end
	llikelihood = [compute_likelihood(N, ly, data) for N in values(lprior_N)]
end

# ╔═╡ 09a77f31-f7b0-49b2-b0c2-11abb0cacd7f
lposterior_N = lprior_N * llikelihood;

# ╔═╡ 2bcf5371-f6e8-49bf-af0d-90a7354771ef
plot(lposterior_N, title="Posterior marginal distribution of N with known p1, p2", xaxis=("Number of bugs (N)"), yaxis=("PMF"), size=(800, 500))

# ╔═╡ 1e6083af-7676-4471-8da4-bb31f494d853
mean(lposterior_N), credible_interval(lposterior_N, 0.9)

# ╔═╡ 69064d67-4895-4e97-827d-bcdcde90c84c
md"## Three-Parameter Model"

# ╔═╡ 726fa3f6-d08f-404a-b065-13b8f5408298
begin
	prior_p0 = pmf_from_seq(range(0,1,51))
	prior_p1 = pmf_from_seq(range(0,1,51))
end;

# ╔═╡ 66260a01-0f24-4241-9dec-343223258e93
joint2 = make_joint(*, prior_p0, prior_p1);

# ╔═╡ dd2d05f8-9d14-4291-9bb9-fcf9dbd074fc
begin
	lindex, lvals = stack(joint2)
	joint2_pmf = pmf_from_seq(lindex, lvals)
end;

# ╔═╡ c7372eab-1dd5-4885-a79e-9e8be67bf82d
joint3 = make_joint(*, lprior_N, joint2_pmf);

# ╔═╡ bc1ebb0a-3fec-4470-8b1d-7961504c66a5
begin
	lindex3, lvals3 = stack(joint3)
	joint3_pmf = pmf_from_seq(lindex3, lvals3)
end;

# ╔═╡ ab20a463-784e-468d-80b7-e9efa63891fa
begin
	function compute_likelihood2(N, p0, p1, data)
		y = compute_probs(p0, p1)
		x = vcat([N - sum(data)], data)
		pdf(Multinomial(N, y), x)
	end
	llikelihood2 = [compute_likelihood2(N, p0, p1, data) for (N, (p0, p1)) in values(joint3_pmf)]
end

# ╔═╡ 4e02876f-71fb-4921-8cfd-1a9f86b497ee
posterior3_pmf = joint3_pmf * llikelihood2;

# ╔═╡ b3e48792-0038-491a-aec9-8d065f5eb3ea
posterior3_joint = unstack(values(posterior3_pmf), probs(posterior3_pmf));

# ╔═╡ 8b3ceabc-8b62-4dca-8280-7511d704c987


# ╔═╡ d9f01217-87a7-4a77-b589-2b8b445c3168
posterior3_N = marginal(posterior3_joint, 1);

# ╔═╡ bfe33f89-1f88-4751-afff-77a3d958b0f0
plot(posterior3_N)

# ╔═╡ 206a27e5-c96d-4108-871e-2251eeb33d7f
mean(posterior3_N)

# ╔═╡ 163dc932-d487-4fbe-bf23-0cc7acb843da
md"""Because of the way that posterior3_joint's index is, we've got to disentangle it. 

The index is [(p1₁, p2₁),(p1₂, p2₂),...(p1ₙ, p2ₙ)]. What we need is a Joint with the p1s as the index and the p2s as the columns. The values of each (p1ₓ, p2ₓ) pair is the sum of the values of the matrix on the 2nd dimension.

This is what's going on below"""

# ╔═╡ bc2da982-203e-42a1-92e9-1514371eec8a
begin
	indices = columns(posterior3_joint)
	vals = sum(posterior3_joint.M, dims=2)
	posterior_px_joint = unstack(indices, vals)
	posterior_p1 = marginal(posterior_px_joint, 1)
	posterior_p2 = marginal(posterior_px_joint, 2)
end;

# ╔═╡ a9be33a5-d33e-4d88-97f0-e153ec4dc369
begin
	plot(posterior_p1, label="p1")
	plot!(posterior_p2, label="p2")
end

# ╔═╡ 54edbbed-c32e-45b0-92cd-45fe6e98f837
mean(posterior_p1), credible_interval(posterior_p1, 0.9)

# ╔═╡ 2c97d098-3a08-4916-aea6-e2b406e02760
mean(posterior_p2), credible_interval(posterior_p2, 0.9)

# ╔═╡ cd84bda4-9e60-4ae0-9997-955780495130
md"""## Exercises

### _exercise 15.1_
"""

# ╔═╡ eefbbed5-e2c9-41b4-90f2-51089eaad235
begin
	on_both_lists = 49
	list1 = 135
	list2 = 122
	hdata=[list2 - on_both_lists, list1 - on_both_lists, on_both_lists]
end

# ╔═╡ 68252c1d-7e03-411e-97e1-ac4d4cc7ba6b
begin
	hprior_N = pmf_from_seq(range(200, 500, step=5))
	hprior_p = pmf_from_seq(range(0, 0.98, 50))
end;

# ╔═╡ 2418c72d-338b-4229-a844-164f21c4fed7
hjoint_prior = make_joint(*, hprior_N, hprior_p);

# ╔═╡ 227b20bf-7783-4e59-8db4-4c8e67029341
begin
	hjoint_prior_index, hjoint_prior_probs = stack(hjoint_prior)
	hjoint_prior_pmf = pmf_from_seq(hjoint_prior_index, hjoint_prior_probs)
	hobserved = sum(hdata)
end

# ╔═╡ f919f645-0eee-4e3d-ba52-d36c8667448b
hlikelihood = [make_joint_likelihood(N, p, hdata) for (N, p) in hjoint_prior_index]

# ╔═╡ 249b36a2-7ce8-49c3-8ebf-0e4449672f1d
hposterior_pmf = hjoint_prior_pmf * hlikelihood;

# ╔═╡ e4d459a8-0c41-4a78-a4a2-8547a329e5ea
hposterior_joint = transpose(unstack(values(hposterior_pmf), probs(hposterior_pmf)));

# ╔═╡ 324def7f-7219-4893-a9fb-73fc16ca7f79
contour(hposterior_joint)

# ╔═╡ 04eb9c4d-da77-400d-b7b8-deb81e4922e2
begin
	hmarginal_p = marginal(hposterior_joint, 1)
	hmarginal_n = marginal(hposterior_joint, 2)
end;

# ╔═╡ 6ba93371-938c-4b5e-b638-49ae9cf88e36
plot(hmarginal_p)

# ╔═╡ 3d6b7840-4572-43fe-b62b-2ac5e2510dde
plot(hmarginal_n)

# ╔═╡ dfd8a137-f014-4f58-823c-8a6970dd10fd
mean(hmarginal_n), credible_interval(hmarginal_n, 0.9)

# ╔═╡ 237ce393-7c60-4e3e-8cf0-e210aabb7113
md"### _exercise 15.2_"

# ╔═╡ 51c53b82-83c8-4d72-83e2-618eb7abb282
data3 = [63, 55, 18, 69, 17, 21, 28]

# ╔═╡ f6ec7d2c-d729-40ee-aa7c-90c4f974a279
begin
	hp = 0.2
	ht = (1-p, p)
end

# ╔═╡ 0287fd43-0e64-4495-8ff2-d526f3a557d7
cartesian_product(args...) = prod.(vec(collect(Iterators.product(args...))))

# ╔═╡ e5ec1550-bd3b-4fb1-8767-c4cd41d4adae
cartesian_product(ht, ht, ht)

# ╔═╡ e1476ace-1439-4c01-a1ac-90243d4e229d
begin
	function make_joint_likelihood2(N, p, hdata)
		observed = sum(hdata)
		k00 = N - observed
		x = vcat([k00], hdata)
		t = (1-p, p)
		y = cartesian_product(t, t, t)
		m = Multinomial(N, y)
		pdf(m, x)
	end
	hlikelihood3 = [make_joint_likelihood2(N, p, data3) for (N, p) in hjoint_prior_index] 
end

# ╔═╡ 74dff101-d7b7-4b47-b0b0-2ab4ead3ba20
hposterior3_pmf = hjoint_prior_pmf * hlikelihood3;

# ╔═╡ 99508a56-f7e7-4a76-8cd7-decb1533b2c9
hposterior3_joint = transpose(unstack(values(hposterior3_pmf), probs(hposterior3_pmf)));

# ╔═╡ bbc07509-2043-4262-9c78-513c10811769
contour(hposterior3_joint)

# ╔═╡ ecc71fcd-773b-483e-9e9f-3b7ce45ec53a
begin
	hmarginal3_p = marginal(hposterior3_joint, 1)
	hmarginal3_n = marginal(hposterior3_joint, 2)
end;

# ╔═╡ ff45e83e-84a0-45ac-8d54-adc93c530bcc
begin
	plot(hmarginal3_n, label="3 lists")
	plot!(hmarginal_n, label="2 lists")
end

# ╔═╡ 23211301-f933-4f59-8bf0-968c11e24cf0
mean(hmarginal3_n), credible_interval(hmarginal3_n, 0.9)

# ╔═╡ e9c661aa-6562-4d98-a5a7-e4ee95851473
mean(hmarginal_n), credible_interval(hmarginal_n, 0.9)

# ╔═╡ 4974d5aa-dc0e-4a20-b5f2-81da481937bc
plot(hmarginal3_p)

# ╔═╡ d97990e2-0473-46ee-bb14-7312778b9bca
mean(hmarginal3_n), credible_interval(hmarginal3_p, 0.9)

# ╔═╡ Cell order:
# ╠═6b6640d0-e1de-11ec-3baf-01ded65b80ff
# ╠═20a227a8-dece-403f-af26-92225880e2b8
# ╠═f1325978-b2d0-4ec8-ad45-7761510280f4
# ╟─f749819f-6904-4a1e-a0cf-6c301d710118
# ╠═018690c4-4737-4387-9e28-9c18a3527e66
# ╠═ce5f433a-3cd1-4b3b-84a9-fad8635c159f
# ╠═fc125273-cad0-4be6-8c3f-67755ef46dc3
# ╟─05b15a27-5540-4903-b8e8-ff291410211e
# ╠═b4a4991e-9431-4f4b-940b-a4dc573b8703
# ╠═63ab927a-0963-42f5-84a5-0feaa255a0f9
# ╠═619634a0-acdf-4bc3-b7e2-ba4551053631
# ╠═eaafa265-b110-4a16-a0b6-d8e43e4c55f0
# ╠═310f4f4b-0832-4bb0-8f98-2ace5bb48c13
# ╠═c906584d-496d-4881-a0f9-777730ff229b
# ╠═a6f060e7-1d4f-4f61-b742-6acb70363e79
# ╟─88423616-07e2-4139-9d58-cd46ab25f051
# ╠═93763f67-4500-48c4-8660-07b8e514916d
# ╠═e8db554e-0b31-4878-b812-d7c31b0b6574
# ╠═a83afb07-7347-4945-a0f2-9f4279beac53
# ╠═cbefa829-c545-4f87-81c9-14d9e1b526c9
# ╠═5edf395b-4428-44bb-91e8-ca1a76cf9c34
# ╠═9b23f4ba-6426-4594-b9d6-93b715235cf8
# ╠═8595805b-8654-44c3-875a-e08c5e467a3d
# ╟─14f9ac87-c8af-4a6b-8657-58606c8db5b8
# ╠═be381749-61d7-4dd9-a88c-95e4fa0426a9
# ╠═cc192b61-bfec-4bad-b2a0-3d5cea362db0
# ╠═44c6bbe8-630a-4960-8607-0eddfc309f8c
# ╠═182cdbbe-9b78-45a7-aa2b-0d293e6ca863
# ╟─a7cacc0f-dde5-4989-88bc-727d79b19983
# ╠═69a460c0-5812-41ab-a9bf-3f0ddb8a05d7
# ╠═6e6e1df1-c607-4a01-9616-a412dd7cacf0
# ╠═806a0b0b-b171-451f-9fef-925a90b663e6
# ╠═fb921a3e-61db-4d36-9edd-808477ebdfd6
# ╠═18a2765a-c354-4f9a-87cb-b9317ed27887
# ╠═2317838c-e9cd-4d1b-a833-d86f5e76853e
# ╠═101d7bb6-4135-40f4-953a-abb5f8c69bb7
# ╠═01dc87f9-6248-4911-9548-38613c78b386
# ╠═2b1bef2f-7bc3-48ca-b272-b35177686e82
# ╠═21855bf2-9321-4933-ab26-02defa105850
# ╠═52a61815-beaa-475e-ae04-a5d1a9f75ac7
# ╠═33756054-fe68-48e7-9e09-345f9e560fe5
# ╠═9854763f-c8fa-4992-84e1-ffd9769d90f2
# ╠═0f887cc4-33dc-44f1-97a9-de480eb2b11e
# ╠═e97e23e6-7109-4145-9c72-9c0ec50101ed
# ╠═0ae406e7-baf0-4200-8439-4432939e4fe3
# ╠═454a5e06-72c9-4774-8e7f-a2e1a694b049
# ╟─da9310fa-38ba-410a-979f-584c914885ab
# ╠═0a00d82a-a41a-4098-a8c7-90796ca5162f
# ╠═fcb7010f-4784-4d99-9867-764b44007a6b
# ╠═42202dbd-a39f-4b26-9f89-43244e4a3a02
# ╠═2baeb3d3-2a07-413a-a068-444336d615fb
# ╠═ff284b61-158a-4072-a5d9-a1ffc4b1977b
# ╠═7e5b0c5b-a973-4004-a3bb-07692af16092
# ╠═afafe1e6-687b-4b96-b98c-781a26765f3d
# ╠═09a77f31-f7b0-49b2-b0c2-11abb0cacd7f
# ╠═2bcf5371-f6e8-49bf-af0d-90a7354771ef
# ╠═1e6083af-7676-4471-8da4-bb31f494d853
# ╟─69064d67-4895-4e97-827d-bcdcde90c84c
# ╠═726fa3f6-d08f-404a-b065-13b8f5408298
# ╠═66260a01-0f24-4241-9dec-343223258e93
# ╠═dd2d05f8-9d14-4291-9bb9-fcf9dbd074fc
# ╠═c7372eab-1dd5-4885-a79e-9e8be67bf82d
# ╠═bc1ebb0a-3fec-4470-8b1d-7961504c66a5
# ╠═ab20a463-784e-468d-80b7-e9efa63891fa
# ╠═4e02876f-71fb-4921-8cfd-1a9f86b497ee
# ╠═b3e48792-0038-491a-aec9-8d065f5eb3ea
# ╠═8b3ceabc-8b62-4dca-8280-7511d704c987
# ╠═d9f01217-87a7-4a77-b589-2b8b445c3168
# ╠═bfe33f89-1f88-4751-afff-77a3d958b0f0
# ╠═206a27e5-c96d-4108-871e-2251eeb33d7f
# ╟─163dc932-d487-4fbe-bf23-0cc7acb843da
# ╠═bc2da982-203e-42a1-92e9-1514371eec8a
# ╠═a9be33a5-d33e-4d88-97f0-e153ec4dc369
# ╠═54edbbed-c32e-45b0-92cd-45fe6e98f837
# ╠═2c97d098-3a08-4916-aea6-e2b406e02760
# ╟─cd84bda4-9e60-4ae0-9997-955780495130
# ╠═eefbbed5-e2c9-41b4-90f2-51089eaad235
# ╠═68252c1d-7e03-411e-97e1-ac4d4cc7ba6b
# ╠═2418c72d-338b-4229-a844-164f21c4fed7
# ╠═227b20bf-7783-4e59-8db4-4c8e67029341
# ╠═f919f645-0eee-4e3d-ba52-d36c8667448b
# ╠═249b36a2-7ce8-49c3-8ebf-0e4449672f1d
# ╠═e4d459a8-0c41-4a78-a4a2-8547a329e5ea
# ╠═324def7f-7219-4893-a9fb-73fc16ca7f79
# ╠═04eb9c4d-da77-400d-b7b8-deb81e4922e2
# ╠═6ba93371-938c-4b5e-b638-49ae9cf88e36
# ╠═3d6b7840-4572-43fe-b62b-2ac5e2510dde
# ╠═dfd8a137-f014-4f58-823c-8a6970dd10fd
# ╟─237ce393-7c60-4e3e-8cf0-e210aabb7113
# ╠═51c53b82-83c8-4d72-83e2-618eb7abb282
# ╠═f6ec7d2c-d729-40ee-aa7c-90c4f974a279
# ╠═0287fd43-0e64-4495-8ff2-d526f3a557d7
# ╠═e5ec1550-bd3b-4fb1-8767-c4cd41d4adae
# ╠═e1476ace-1439-4c01-a1ac-90243d4e229d
# ╠═74dff101-d7b7-4b47-b0b0-2ab4ead3ba20
# ╠═99508a56-f7e7-4a76-8cd7-decb1533b2c9
# ╠═bbc07509-2043-4262-9c78-513c10811769
# ╠═ecc71fcd-773b-483e-9e9f-3b7ce45ec53a
# ╠═ff45e83e-84a0-45ac-8d54-adc93c530bcc
# ╠═23211301-f933-4f59-8bf0-968c11e24cf0
# ╠═e9c661aa-6562-4d98-a5a7-e4ee95851473
# ╠═4974d5aa-dc0e-4a20-b5f2-81da481937bc
# ╠═d97990e2-0473-46ee-bb14-7312778b9bca
