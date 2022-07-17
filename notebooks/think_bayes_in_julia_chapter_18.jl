### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ df3ded8c-015a-11ed-0d15-cde2a5ec3e72
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ 50084207-f757-4b87-b6de-1e8de705b0f6
using DataFrames, CSV, Plots, Distributions, PlutoUI

# ╔═╡ 2214cdea-0676-4e9d-8574-c55a37592378
TableOfContents()

# ╔═╡ 31e1d1f9-c58e-428b-95f5-934692da1ef9
md"# The World Cup Problem Revisited"

# ╔═╡ a00ae43a-50a5-4a21-8030-a1f2c2b30e4b
begin
	alpha = 1.4
	dist = Gamma(alpha)
end

# ╔═╡ ec3da7aa-47c8-426c-a86c-27acf15e1c04
lams = LinRange(0, 10, 101)

# ╔═╡ d9d42113-7a2f-4804-b879-0a4342b66cb8
prior = pmf_from_dist(lams, dist);

# ╔═╡ f4c757ab-cf06-43da-b97e-b98875cad9cd
begin
	k = 4
	likelihood = [pdf(Poisson(λ), k) for λ in lams]
end

# ╔═╡ df8a0171-38f4-4da8-a966-1a727ff400b4
posterior = prior * likelihood;

# ╔═╡ 626cd5a5-aeae-4162-93b6-86aa21c63434
make_gamma_dist(alpha, beta) = Gamma(alpha, 1/beta)

# ╔═╡ 0e07b99e-0fde-42ca-a5bd-a083b699b257
prior_gamma = make_gamma_dist(1.4, 1)

# ╔═╡ 1c774d1f-7ba6-4525-91c8-2177c92a44e5
beta(g::Gamma) = 1 / scale(g)

# ╔═╡ ff9c7695-efcc-4775-ab23-8153a2e1d07d
function update_gamma(prior, data)
	k, t = data
	alpha = shape(prior) + k
	β = beta(prior) + t
	make_gamma_dist(alpha, β)
end

# ╔═╡ f7e83f33-7809-4417-a800-a7e2a13d0000
begin
	data = (4, 1)
	posterior_gamma = update_gamma(prior_gamma, data)
end

# ╔═╡ 436f9b5b-dd27-42d2-a7ed-870f362e4d9c
posterior_conjugate = pmf_from_dist(lams, posterior_gamma);

# ╔═╡ d839ccfa-6f9f-44f5-98c8-85e6d13966f4
begin
	plot(posterior_conjugate, alpha=0.4)
	plot!(posterior, linestyle=:dashdot)
end

# ╔═╡ de002d38-9bd6-44bf-8c31-a40cdc0526ea
posterior ≈ posterior_conjugate

# ╔═╡ 4941deee-8d36-4925-9ab9-58348dbafcdf
md"# Binomial Likelihood"

# ╔═╡ cb03afdd-847f-4ff1-ad5c-918e43d7ecc0
begin
	xs = LinRange(0, 1, 101)
	uniform = pmf_from_seq(xs)
end;

# ╔═╡ 62280e53-a6ca-4372-8163-70a15409482d
likelihood2 = binom_pmf(140, 250, xs)

# ╔═╡ 59ee56d4-a838-4c56-9120-3f7cab430e96
posterior2 = uniform * likelihood2;

# ╔═╡ 731bbb7b-18e3-492e-9088-9c1a22720393
prior_beta = Beta(1,1)

# ╔═╡ 1a664b00-6744-4750-bc5f-c19594e74df7
function update_beta(prior, data)
	k, n  = data
	alpha, beta = params(prior) .+ [k, n-k]
	Beta(alpha, beta)
end

# ╔═╡ 8d5b221c-d13e-4e7a-86eb-54468167dc7f
begin
	data2 = 140, 250
	posterior_beta = update_beta(prior_beta, data2)
end

# ╔═╡ 9f7c4ae8-b227-4740-92b3-f965d0a7c5b1
posterior_conjugate2 = pmf_from_dist(xs, posterior_beta);

# ╔═╡ faeace00-ad5e-4d45-8279-18e9dc1c6163
begin
	plot(posterior2, alpha=0.4)
	plot!(posterior_conjugate2, linestyle=:dashdot)
end

# ╔═╡ e176a56e-c1b8-4485-ac4d-52be2c613034
posterior2 ≈ posterior_conjugate2

# ╔═╡ b7904013-4670-4326-96d8-1a47c0b6ba1d
md"# Lions and Tigers and Bears"

# ╔═╡ 67b2da60-23ce-4508-ad32-d4322095dfb9
begin
	data3 = [3, 2, 1]
	n = sum(data3)
	ps = [0.4, 0.3, 0.3]
	pdf(Multinomial(n, ps), data3)
end

# ╔═╡ 7fc0ff03-693e-448c-b077-a6c62556e55a
md"## The Dirichlet Distribution"

# ╔═╡ 59477258-02b1-47cf-b09b-3553c2351d1b
begin
	alphas = [1,2,3]
	dist3 = Dirichlet(alphas)
end

# ╔═╡ e79ceed0-073d-4ccf-946d-8d6ab73d1baa
sample = rand(dist3, 1000)

# ╔═╡ 108984eb-5c9a-4a8f-89d9-2ef19f1c469c
cdfs = [cdf_from_seq(sample[i,:]) for i in 1:size(sample)[1] ];

# ╔═╡ da3ac2c3-a4c5-43b5-b94c-64768fb6548e
begin
	plot()
	for (i,c) in enumerate(cdfs)
		plot!(c, label="Column "*string(i))
	end
	plot!(legend_position=:topleft)
end

# ╔═╡ d9bb524a-6bd9-4ada-aabb-ac26c6d22953
marginal_beta(alpha, i) = Beta(alpha[i], sum(alpha)-alpha[i])

# ╔═╡ 3e8fa94b-2fec-453a-8a69-34a3f7973ea0
marginals = [marginal_beta(alphas, i) for i in 1:length(alphas)]

# ╔═╡ 81999499-2136-4cbc-8381-b766b362026c
begin
	plot()
	for i in 1:length(alphas)
		plot!(cdf_from_dist(xs, marginals[i]))
		plot!(cdfs[i], label="Column "*string(i), linestyle=:dash)
	end
	plot!()
end

# ╔═╡ 0a1a4c0a-b4e6-45be-bf20-a08a053e01dc
md"# Exercises

### exercise 18.1"

# ╔═╡ 8170ae7a-485c-41ec-a0db-893ab3b7c3f8
begin
	wc_data1 = 1, 11/90
	wc_posterior1 = update_gamma(prior_gamma, wc_data1)
end

# ╔═╡ 6a18e57c-527a-4515-9a77-722fe577be1c
begin
	wc_data2 = 1, 12/90
	wc_posterior2 = update_gamma(wc_posterior1, wc_data2)
end

# ╔═╡ 99d9c1e7-bf78-4b84-bf9c-dc06b6023152
mean(prior_gamma), mean(wc_posterior1), mean(wc_posterior2)

# ╔═╡ dd468a0f-e592-480a-99dd-6380710f436e
begin
	plot(pmf_from_dist(lams, prior_gamma), label="prior")
	plot!(pmf_from_dist(lams, wc_posterior1), label="after 1 goal")
	plot!(pmf_from_dist(lams, wc_posterior2), label="after 2 goals")
	plot!(xaxis=("Goal scoring rate", 0:2:10), yaxis="PMF", title="World Cup Problem, Germany v Brazil")
end

# ╔═╡ e0604da3-6dbb-41c5-a19f-747a10ec5fa0
md"### exercise 18.2"

# ╔═╡ ef6c2668-59b4-4a29-a10b-350f80873095
begin
	ramp_up = 0.0:49
	ramp_down = 50:-1:0
	a = normalize(vcat(ramp_up, ramp_down))
	triangle = pmf_from_seq(collect(xs), a)
end;

# ╔═╡ cb83bf4e-b44f-4510-94a7-1cfd1c2185c1
begin
	likelihood4 = binom_pmf(140, 250, xs)
	posterior4 = triangle * likelihood4
end;

# ╔═╡ cb268d9f-37fb-4028-95e1-a6810300a6e8
begin
	plot(triangle)
	plot!(pmf_from_dist(xs, prior_beta))
end

# ╔═╡ aeb1f3a5-1bf6-4e64-89aa-42874eafd965
posterior_beta2 = update_beta(prior_beta, (140, 250))

# ╔═╡ 5c5bf20b-766b-47f5-9a37-c2d84d2a1355
posterior_conjugate3 = pmf_from_dist(xs, posterior_beta2);

# ╔═╡ aa9941ad-bb8d-41db-8daf-68071e972240
begin
	plot(posterior4)
	plot!(posterior_conjugate3)
end

# ╔═╡ 3c4dba09-812f-4655-a4d6-1b3aeffb5b09
posterior4 ≈ posterior_conjugate3

# ╔═╡ e497af10-cfcc-4f19-8891-a45596c58903
md"### exercise 18.3"

# ╔═╡ b0f4a37b-d157-47d5-ade8-151904526b19
begin
	xs2 = LinRange(0.005, 0.995, 199)
	beta3 = Beta(8, 2)
	prior_beta3 = pmf_from_dist(xs2, beta3)
	plot(prior_beta3)
end

# ╔═╡ a4c4378c-b06b-4014-b937-39fd1202c7b1
begin
	sdata1 = (10, 10)
	sdata2 = (48, 50)
	sdata3 = (186, 200)
	seller1 = update_beta(beta3, sdata1)
	seller2 = update_beta(beta3, sdata2)
	seller3 = update_beta(beta3, sdata3)
	seller1_pmf = pmf_from_dist(xs2, seller1)
	seller2_pmf = pmf_from_dist(xs2, seller2)
	seller3_pmf = pmf_from_dist(xs2, seller3)
	plot(seller1_pmf, label="seller 1")
	plot!(seller2_pmf, label="seller 2")
	plot!(seller3_pmf, label="seller 3")
	plot!(xaxis=("Probability of positive rating", (0.65, 1.0)))
end

# ╔═╡ 0c2ddea0-73c5-49eb-96e6-1648662ccf09
mean(seller1), mean(seller2), mean(seller3)

# ╔═╡ de42e124-e3b7-4bca-bd0d-4daa98ec72b9
a2 = reduce(hcat, [rand(x, 10000) for x in [seller1, seller2, seller3]])

# ╔═╡ c7f8332c-b1d6-43b4-b877-fe097674fc9a
best = argmax(a2, dims=2);

# ╔═╡ 08e61b50-6417-4e9b-8779-7a72d97c7f0e
pmf_from_seq(reshape([b[2] for b in best], length(best)))

# ╔═╡ df0560bd-35d7-41c0-a5e7-295ee93ba169
md"### exercise 18.4"

# ╔═╡ 3db532b2-3af7-4bfc-9301-f7d294f316e9
begin
	prior_alpha = [1,1,1]
	data4 = [3,2,1]
	posterior_alpha = prior_alpha .+ data4
end

# ╔═╡ 78867dc1-b4e0-46bb-b38b-c3cda1a48e0f
	marginal_bear = marginal_beta(posterior_alpha, 3)

# ╔═╡ 060a2091-48f8-4160-9ef1-6a83d277570e
mean(marginal_bear)

# ╔═╡ 3c4a22e2-c91f-4fa4-bacb-853b5ae32289
dist2 = Dirichlet(posterior_alpha)

# ╔═╡ be7a1daa-2179-4735-9a60-50d63e112af5
mean(dist2)

# ╔═╡ Cell order:
# ╠═df3ded8c-015a-11ed-0d15-cde2a5ec3e72
# ╠═50084207-f757-4b87-b6de-1e8de705b0f6
# ╠═2214cdea-0676-4e9d-8574-c55a37592378
# ╟─31e1d1f9-c58e-428b-95f5-934692da1ef9
# ╠═a00ae43a-50a5-4a21-8030-a1f2c2b30e4b
# ╠═ec3da7aa-47c8-426c-a86c-27acf15e1c04
# ╠═d9d42113-7a2f-4804-b879-0a4342b66cb8
# ╠═f4c757ab-cf06-43da-b97e-b98875cad9cd
# ╠═df8a0171-38f4-4da8-a966-1a727ff400b4
# ╠═626cd5a5-aeae-4162-93b6-86aa21c63434
# ╠═0e07b99e-0fde-42ca-a5bd-a083b699b257
# ╠═1c774d1f-7ba6-4525-91c8-2177c92a44e5
# ╠═ff9c7695-efcc-4775-ab23-8153a2e1d07d
# ╠═f7e83f33-7809-4417-a800-a7e2a13d0000
# ╠═436f9b5b-dd27-42d2-a7ed-870f362e4d9c
# ╠═d839ccfa-6f9f-44f5-98c8-85e6d13966f4
# ╠═de002d38-9bd6-44bf-8c31-a40cdc0526ea
# ╟─4941deee-8d36-4925-9ab9-58348dbafcdf
# ╠═cb03afdd-847f-4ff1-ad5c-918e43d7ecc0
# ╠═62280e53-a6ca-4372-8163-70a15409482d
# ╠═59ee56d4-a838-4c56-9120-3f7cab430e96
# ╠═731bbb7b-18e3-492e-9088-9c1a22720393
# ╠═1a664b00-6744-4750-bc5f-c19594e74df7
# ╠═8d5b221c-d13e-4e7a-86eb-54468167dc7f
# ╠═9f7c4ae8-b227-4740-92b3-f965d0a7c5b1
# ╠═faeace00-ad5e-4d45-8279-18e9dc1c6163
# ╠═e176a56e-c1b8-4485-ac4d-52be2c613034
# ╟─b7904013-4670-4326-96d8-1a47c0b6ba1d
# ╠═67b2da60-23ce-4508-ad32-d4322095dfb9
# ╟─7fc0ff03-693e-448c-b077-a6c62556e55a
# ╠═59477258-02b1-47cf-b09b-3553c2351d1b
# ╠═e79ceed0-073d-4ccf-946d-8d6ab73d1baa
# ╠═108984eb-5c9a-4a8f-89d9-2ef19f1c469c
# ╠═da3ac2c3-a4c5-43b5-b94c-64768fb6548e
# ╠═d9bb524a-6bd9-4ada-aabb-ac26c6d22953
# ╠═3e8fa94b-2fec-453a-8a69-34a3f7973ea0
# ╠═81999499-2136-4cbc-8381-b766b362026c
# ╟─0a1a4c0a-b4e6-45be-bf20-a08a053e01dc
# ╠═8170ae7a-485c-41ec-a0db-893ab3b7c3f8
# ╠═6a18e57c-527a-4515-9a77-722fe577be1c
# ╠═99d9c1e7-bf78-4b84-bf9c-dc06b6023152
# ╠═dd468a0f-e592-480a-99dd-6380710f436e
# ╟─e0604da3-6dbb-41c5-a19f-747a10ec5fa0
# ╠═ef6c2668-59b4-4a29-a10b-350f80873095
# ╠═cb83bf4e-b44f-4510-94a7-1cfd1c2185c1
# ╠═cb268d9f-37fb-4028-95e1-a6810300a6e8
# ╠═aeb1f3a5-1bf6-4e64-89aa-42874eafd965
# ╠═5c5bf20b-766b-47f5-9a37-c2d84d2a1355
# ╠═aa9941ad-bb8d-41db-8daf-68071e972240
# ╠═3c4dba09-812f-4655-a4d6-1b3aeffb5b09
# ╟─e497af10-cfcc-4f19-8891-a45596c58903
# ╠═b0f4a37b-d157-47d5-ade8-151904526b19
# ╠═a4c4378c-b06b-4014-b937-39fd1202c7b1
# ╠═0c2ddea0-73c5-49eb-96e6-1648662ccf09
# ╠═de42e124-e3b7-4bca-bd0d-4daa98ec72b9
# ╠═c7f8332c-b1d6-43b4-b877-fe097674fc9a
# ╠═08e61b50-6417-4e9b-8779-7a72d97c7f0e
# ╟─df0560bd-35d7-41c0-a5e7-295ee93ba169
# ╠═3db532b2-3af7-4bfc-9301-f7d294f316e9
# ╠═78867dc1-b4e0-46bb-b38b-c3cda1a48e0f
# ╠═060a2091-48f8-4160-9ef1-6a83d277570e
# ╠═3c4a22e2-c91f-4fa4-bacb-853b5ae32289
# ╠═be7a1daa-2179-4735-9a60-50d63e112af5
